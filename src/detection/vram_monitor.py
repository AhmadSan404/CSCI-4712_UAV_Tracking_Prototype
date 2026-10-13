"""GPU memory readout, refreshed at most once per second."""

import time

import torch

REFRESH_SECONDS = 1.0
BYTES_PER_MEGABYTE = 1024 * 1024


class VramMonitor:
    """Reads how much GPU memory is in use. On the CPU both values stay None."""

    def __init__(self, device: str) -> None:
        self._is_enabled = device.startswith("cuda")
        self._used_mb: float | None = None
        self._total_mb: float | None = None
        self._last_check = 0.0

    @property
    def used_mb(self) -> float | None:
        """GPU memory in use in MB (for the whole GPU, not only this program), or None on the CPU."""
        self._refresh()
        return self._used_mb

    @property
    def total_mb(self) -> float | None:
        """Total GPU memory in MB, or None on the CPU."""
        self._refresh()
        return self._total_mb

    def _refresh(self) -> None:
        """Queries the GPU, but not more often than `REFRESH_SECONDS`."""
        if not self._is_enabled:
            return
        now = time.perf_counter()
        if now - self._last_check < REFRESH_SECONDS:
            return
        self._last_check = now
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        self._total_mb = total_bytes / BYTES_PER_MEGABYTE
        self._used_mb = (total_bytes - free_bytes) / BYTES_PER_MEGABYTE
