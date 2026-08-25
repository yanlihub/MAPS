"""Shared helpers for the MAPS pipeline scripts."""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_paths() -> dict:
    """Resolve the data tree. MAPS_DATA_ROOT wins over configs/paths.yaml, and a relative
    root is taken relative to the repository rather than the working directory."""
    with open(REPO_ROOT / "configs" / "paths.yaml") as fh:
        raw = yaml.safe_load(fh)
    root = os.environ.get("MAPS_DATA_ROOT", raw["data_root"])
    root = str(Path(root) if Path(root).is_absolute() else (REPO_ROOT / root).resolve())
    out = {k: Path(str(v).replace("${data_root}", root)) for k, v in raw.items()}
    out["data_root"] = Path(root)
    return out


def load_config(dataset: str) -> dict:
    with open(REPO_ROOT / "configs" / f"{dataset}.yaml") as fh:
        return yaml.safe_load(fh)


class _GpuSampler(threading.Thread):
    """Poll nvidia-smi in the background so utilisation can be averaged over a stage.

    A stage can hold a large allocation while barely using the device, so peak memory alone
    is not enough. Utilisation has to be sampled over time, which torch does not expose.
    """

    def __init__(self, device: int, interval: float = 1.0):
        super().__init__(daemon=True)
        self.device = device
        self.interval = interval
        self.samples: list[float] = []
        # Thread._stop is an internal method of threading.Thread; do not shadow it.
        self._halt = threading.Event()

    def run(self) -> None:
        while not self._halt.wait(self.interval):
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits", f"--id={self.device}"],
                    capture_output=True, text=True, timeout=5,
                )
                if out.returncode == 0 and out.stdout.strip():
                    self.samples.append(float(out.stdout.strip().splitlines()[0]))
            except Exception:
                pass

    def stop(self) -> None:
        self._halt.set()


class Timer:
    """Wall clock, CPU time, peak GPU memory and mean GPU utilisation for one stage."""

    def __init__(self, label: str, sample_gpu: bool = True):
        self.label = label
        self.sample_gpu = sample_gpu
        self.record: dict = {"label": label}

    def __enter__(self):
        try:
            import torch

            self._torch = torch if torch.cuda.is_available() else None
            if self._torch:
                self._torch.cuda.reset_peak_memory_stats()
        except Exception:
            self._torch = None

        self._sampler = None
        if self._torch and self.sample_gpu:
            # CUDA_VISIBLE_DEVICES remaps device ids, so ask nvidia-smi about the
            # physical device this process was pinned to.
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
            try:
                self._sampler = _GpuSampler(int(visible))
                self._sampler.start()
            except ValueError:
                self._sampler = None

        self._cpu0 = time.process_time()
        self._t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.record["wall_seconds"] = round(time.time() - self._t0, 2)
        self.record["cpu_seconds"] = round(time.process_time() - self._cpu0, 2)
        if self._sampler:
            self._sampler.stop()
            self._sampler.join(timeout=3)
            if self._sampler.samples:
                self.record["gpu_util_mean_pct"] = round(
                    sum(self._sampler.samples) / len(self._sampler.samples), 1
                )
                self.record["gpu_util_max_pct"] = round(max(self._sampler.samples), 1)
                self.record["gpu_util_n_samples"] = len(self._sampler.samples)
        if self._torch:
            self.record["peak_gpu_mem_gb"] = round(
                self._torch.cuda.max_memory_allocated() / 1024**3, 3
            )
            self.record["gpu_name"] = self._torch.cuda.get_device_name(0)
        self.record["peak_rss_gb"] = round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2, 3
        )
        print(f"[timing] {self.label}: {self.record['wall_seconds']}s wall, "
              f"{self.record['cpu_seconds']}s cpu, "
              f"gpu {self.record.get('gpu_util_mean_pct', '-')}% mean, "
              f"{self.record.get('peak_gpu_mem_gb', '-')}GB peak, "
              f"{self.record['peak_rss_gb']}GB rss")
        return False


def write_json(obj, path: Path) -> None:
    # Written to a sibling temporary file and renamed into place. Two jobs writing the
    # same result concurrently once left a valid document with the tail of a longer
    # previous write appended to it, which no longer parsed; os.replace is atomic on
    # POSIX, so a reader sees either the old file or the new one and never a splice.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)
    print(f"[write] {path}")


# Figures land inside the repository unless MAPS_FIGURE_DIR says otherwise.
FIGURE_DIR = Path(os.environ.get("MAPS_FIGURE_DIR", REPO_ROOT / "figures"))


def save_figure(fig, stem: str, dpi: int = 300, tight: bool = True):
    """Write one figure in the formats the submission needs.

    JBI wants TIFF or EPS at publication resolution; PNG is kept alongside because it is
    what is actually readable while iterating, and PDF because LaTeX handles it best.
    LZW keeps the TIFFs from being enormous without touching the pixels.
    """
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for ext, kw in (("png", {}), ("pdf", {}), ("tif", {"pil_kwargs": {"compression": "tiff_lzw"}})):
        out = FIGURE_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=dpi,
                    bbox_inches="tight" if tight else None, **kw)
        written.append(out)
    print(f"[figure] {stem}: " + ", ".join(w.suffix for w in written)
          + f"  -> {FIGURE_DIR}")
    return written
