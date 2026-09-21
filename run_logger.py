"""Crash-resistant, append-only logging for experiment entry points."""
from __future__ import annotations

import atexit
import faulthandler
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, TextIO


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return repr(value)


class _Tee(TextIO):
    def __init__(self, terminal: TextIO, sinks: list[TextIO], lock: threading.RLock):
        self._terminal = terminal
        self._sinks = sinks
        self._lock = lock

    @property
    def encoding(self):
        return getattr(self._terminal, "encoding", "utf-8")

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        return bool(getattr(self._terminal, "isatty", lambda: False)())

    def fileno(self) -> int:
        return self._terminal.fileno()

    def write(self, text: str) -> int:
        with self._lock:
            self._terminal.write(text)
            for sink in self._sinks:
                sink.write(text)
        return len(text)

    def flush(self) -> None:
        with self._lock:
            self._terminal.flush()
            for sink in self._sinks:
                sink.flush()


class RunLogger:
    """Mirror stdout/stderr to a unique raw log and a legacy named log.

    Call ``start`` before parsing arguments. ``launch_run.sh`` is the outer
    boundary for syntax/import failures that occur before this class can start.
    """

    def __init__(
        self,
        entrypoint: str | Path,
        *,
        raw_root: str | Path | None = None,
        source_paths: Iterable[str | Path] = (),
    ) -> None:
        self.entrypoint = Path(entrypoint).resolve()
        self.raw_root = Path(raw_root or os.environ.get("MOLECULE_RUN_ROOT", "./runs")).resolve()
        self.source_paths = [Path(path).resolve() for path in source_paths]
        self.started_at = datetime.now().astimezone()
        stamp = self.started_at.strftime("%Y%m%dT%H%M%S.%f%z")
        default_run_id = f"{stamp}_{self.entrypoint.stem}_pid{os.getpid()}"
        self.run_id = os.environ.get("MOLECULE_RUN_ID", default_run_id)
        env_run_dir = os.environ.get("MOLECULE_RUN_DIR")
        self.run_dir = Path(env_run_dir).resolve() if env_run_dir else self.raw_root / self.run_id
        self.raw_log = self.run_dir / f"{stamp}.log"
        self.manifest_path = self.run_dir / "runtime_manifest.json"
        self._lock = threading.RLock()
        self._raw_handle: Optional[TextIO] = None
        self._legacy_handle: Optional[TextIO] = None
        self._original_stdout: Optional[TextIO] = None
        self._original_stderr: Optional[TextIO] = None
        self._original_excepthook = None
        self._manifest: dict[str, Any] = {}
        self._closed = False

    def _git_metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=self.entrypoint.parent,
                capture_output=True, text=True, timeout=5, check=False,
            )
            if commit.returncode == 0:
                result["commit"] = commit.stdout.strip()
            dirty = subprocess.run(
                ["git", "status", "--porcelain"], cwd=self.entrypoint.parent,
                capture_output=True, text=True, timeout=5, check=False,
            )
            if dirty.returncode == 0:
                lines = [line for line in dirty.stdout.splitlines() if line]
                result["dirty"] = bool(lines)
                result["dirty_paths"] = lines
        except (OSError, subprocess.SubprocessError):
            result["error"] = "git metadata unavailable"
        return result

    def _source_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for path in [self.entrypoint, *self.source_paths]:
            try:
                stat = path.stat()
                metadata[str(path)] = {
                    "sha256": _sha256(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                }
            except OSError as exc:
                metadata[str(path)] = {"error": str(exc)}
        return metadata

    def _write_manifest(self) -> None:
        temp = self.manifest_path.with_suffix(f".tmp.{os.getpid()}")
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(self._manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, self.manifest_path)

    def start(self) -> "RunLogger":
        # launch_run.sh creates MOLECULE_RUN_DIR before Python starts. A direct
        # invocation must still refuse to reuse an old run directory.
        if os.environ.get("MOLECULE_RUN_DIR"):
            self.run_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.run_dir.mkdir(parents=True, exist_ok=False)
        self._raw_handle = self.raw_log.open("x", encoding="utf-8", buffering=1)
        self._manifest = {
            "schema_version": 1,
            "run_id": self.run_id,
            "status": "running",
            "started_at": self.started_at.isoformat(),
            "pid": os.getpid(), "ppid": os.getppid(), "hostname": socket.gethostname(),
            "cwd": str(Path.cwd().resolve()), "argv": sys.argv,
            "python": sys.version, "platform": platform.platform(),
            "raw_log": str(self.raw_log),
            "source_files": self._source_metadata(), "git": self._git_metadata(),
        }
        self._write_manifest()
        self._original_stdout, self._original_stderr = sys.stdout, sys.stderr
        sinks = [self._raw_handle]
        sys.stdout = _Tee(self._original_stdout, sinks, self._lock)
        sys.stderr = _Tee(self._original_stderr, sinks, self._lock)
        faulthandler.enable(file=self._raw_handle, all_threads=True)
        self._original_excepthook = sys.excepthook
        sys.excepthook = self._handle_exception
        atexit.register(self._atexit)
        print(f"[run-logger] run_id={self.run_id}")
        print(f"[run-logger] raw_log={self.raw_log}")
        print(f"[run-logger] manifest={self.manifest_path}")
        return self

    def attach_legacy_log(self, path: str | Path | None) -> None:
        if not path:
            return
        legacy_path = Path(path).resolve()
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._legacy_handle = legacy_path.open("a", encoding="utf-8", buffering=1)
        except OSError:
            print(f"[run-logger] WARNING: legacy log could not be opened: {legacy_path}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return
        assert isinstance(sys.stdout, _Tee) and isinstance(sys.stderr, _Tee)
        sys.stdout._sinks.append(self._legacy_handle)
        sys.stderr._sinks.append(self._legacy_handle)
        self._manifest["legacy_log"] = str(legacy_path)
        self._write_manifest()
        print(f"[run-logger] legacy_log={legacy_path}")

    def record_arguments(self, args: Mapping[str, Any]) -> None:
        self._manifest["arguments"] = _json_value(dict(args))
        for name in ("resume", "init_weights"):
            value = args.get(name)
            if not value:
                continue
            path = Path(str(value)).resolve()
            try:
                self._manifest[f"{name}_checkpoint"] = {
                    "path": str(path), "sha256": _sha256(path), "size": path.stat().st_size,
                }
            except OSError as exc:
                self._manifest[f"{name}_checkpoint"] = {"path": str(path), "error": str(exc)}
        self._write_manifest()

    def _handle_exception(self, exc_type, exc_value, exc_traceback) -> None:
        self._manifest["status"] = "failed"
        self._manifest["exception"] = {
            "type": getattr(exc_type, "__name__", str(exc_type)), "message": str(exc_value),
        }
        self._manifest["ended_at"] = datetime.now().astimezone().isoformat()
        try:
            self._write_manifest()
        finally:
            assert self._original_excepthook is not None
            self._original_excepthook(exc_type, exc_value, exc_traceback)

    def finish(self, status: str = "completed", **details: Any) -> None:
        if self._closed:
            return
        self._manifest["status"] = status
        self._manifest["ended_at"] = datetime.now().astimezone().isoformat()
        if details:
            self._manifest["result"] = _json_value(details)
        self._write_manifest()
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            sys.stdout.flush(); sys.stderr.flush()
        finally:
            if self._original_stdout is not None:
                sys.stdout = self._original_stdout
            if self._original_stderr is not None:
                sys.stderr = self._original_stderr
            if self._original_excepthook is not None:
                sys.excepthook = self._original_excepthook
            faulthandler.disable()
            if self._legacy_handle is not None:
                self._legacy_handle.close()
            if self._raw_handle is not None:
                self._raw_handle.close()

    def _atexit(self) -> None:
        if self._closed:
            return
        if self._manifest.get("status") == "running":
            self._manifest["status"] = "exited_without_finish"
            self._manifest["ended_at"] = datetime.now().astimezone().isoformat()
            try:
                self._write_manifest()
            except OSError:
                pass
        self.close()
