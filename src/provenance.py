"""Capture reproducibility metadata for forecasting runs."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import shlex
import subprocess
import sys


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_info(path_value: str | Path | None) -> dict | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        return {"path": str(path_value), "available": False}
    if path.is_file():
        return {
            "path": path.name,
            "kind": "file",
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    digest = hashlib.sha256()
    files = []
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = item.relative_to(path).as_posix()
        file_hash = _sha256_file(item)
        files.append({"path": relative, "bytes": item.stat().st_size, "sha256": file_hash})
        digest.update(relative.encode("utf-8"))
        digest.update(bytes.fromhex(file_hash))
    return {
        "path": path.name,
        "kind": "directory",
        "sha256": digest.hexdigest(),
        "files": files,
    }


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cached_model_revision(model_id: str) -> str | None:
    local = Path(model_id).expanduser()
    if local.is_dir():
        config_path = local / "config.json"
        if config_path.exists():
            try:
                revision = json.loads(config_path.read_text()).get("_commit_hash")
                if revision:
                    return str(revision)
            except (OSError, json.JSONDecodeError):
                pass
    try:
        from huggingface_hub import try_to_load_from_cache
        cached_config = try_to_load_from_cache(model_id, "config.json")
        if isinstance(cached_config, str):
            parts = Path(cached_config).parts
            if "snapshots" in parts:
                return parts[parts.index("snapshots") + 1]
            config = json.loads(Path(cached_config).read_text())
            return config.get("_commit_hash")
    except (ImportError, OSError, ValueError, IndexError):
        pass
    return None


def _package_versions() -> dict[str, str | None]:
    packages = ("numpy", "pandas", "matplotlib", "torch", "transformers",
                "accelerate", "peft", "huggingface-hub")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def build_run_manifest(data_path: str | Path, series, splits, args: dict,
                       model_id: str, adapter_path: str | Path | None,
                       operation: str = "full_model_comparison",
                       extra: dict | None = None) -> dict:
    """Build a portable manifest; paths are stored as basenames, not machine paths."""
    data = artifact_info(data_path)
    if data is not None:
        data["rows_after_regularization"] = int(len(series))
        data["start"] = str(series.index.min())
        data["end"] = str(series.index.max())
        data["timezone"] = str(series.index.tz) if series.index.tz is not None else None
    partitions = {}
    for name in ("train", "validation", "test"):
        part = getattr(splits, name)
        partitions[name] = {
            "rows": int(len(part)),
            "start": str(part.index.min()) if len(part) else None,
            "end": str(part.index.max()) if len(part) else None,
        }

    try:
        import torch
        hardware = {
            "torch_cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        hardware = {"torch_cuda_available": None, "cuda_version": None, "gpu": None}

    portable_args = dict(args)
    for key in ("data", "adapter", "output_dir", "resume_adapter", "predictions"):
        value = portable_args.get(key)
        if value:
            portable_args[key] = Path(value).name

    manifest = {
        "schema_version": 1,
        "operation": operation,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "arguments": portable_args,
        "git": {
            "commit": _git_value("rev-parse", "HEAD"),
            "branch": _git_value("branch", "--show-current"),
            "working_tree_dirty": bool(_git_value("status", "--porcelain")),
        },
        "data": data,
        "split": {
            "evaluation_partition": args.get("split"),
            "train_end_exclusive": args.get("train_end"),
            "validation_end_exclusive": args.get("validation_end"),
            "test_end_exclusive": args.get("test_end"),
            "partitions": partitions,
        },
        "forecast": {
            "horizon_hours": args.get("horizon"),
            "context_length_hours": args.get("context_length"),
            "windows_requested": args.get("windows"),
            "inference_batch_size": args.get("inference_batch_size"),
            "preprocessing": "hourly mean resampling; causal forward-fill up to 3 hours",
        },
        "model": {
            "id": model_id,
            "cached_revision": _cached_model_revision(model_id),
            "adapter": artifact_info(adapter_path),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": _package_versions(),
            "hardware": hardware,
        },
    }
    if extra:
        manifest.update(extra)
    return manifest
