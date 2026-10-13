"""Facts about the machine and libraries a run happened on."""

import platform
import sys

CPU_REGISTRY_KEY = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
CPU_REGISTRY_VALUE = "ProcessorNameString"


class EnvironmentInfo:
    """Collects device, CPU and library version information."""

    @staticmethod
    def read_cpu_name() -> str:
        """The CPU's marketing name from the Windows registry, falling back to `platform.processor()`."""
        if sys.platform == "win32":
            import winreg

            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, CPU_REGISTRY_KEY) as key:
                    return str(winreg.QueryValueEx(key, CPU_REGISTRY_VALUE)[0]).strip()
            except OSError:
                pass
        return platform.processor() or "unknown"

    @staticmethod
    def collect(device_name: str, precision: str, total_vram_mb: float | None) -> dict:
        """Gathers the environment as a JSON-friendly dict. torch and ultralytics are imported only here."""
        import cv2
        import torch
        import ultralytics

        return {
            "device_name": device_name,
            "precision": precision,
            "total_vram_mb": round(total_vram_mb) if total_vram_mb else None,
            "cpu_name": EnvironmentInfo.read_cpu_name(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "ultralytics_version": ultralytics.__version__,
            "opencv_version": cv2.__version__,
        }
