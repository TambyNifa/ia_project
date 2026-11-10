"""Vérifie un environnement GPU et affiche la procédure d'installation."""

import importlib.util
import shutil
import sys


def main() -> None:
    """Contrôle les composants sans installer ni télécharger quoi que ce soit."""
    print(f"Python: {sys.version}")
    for package in ("torch", "transformers", "datasets", "accelerate"):
        if importlib.util.find_spec(package):
            module = __import__(package)
            print(f"{package}: installed ({getattr(module, '__version__', 'unknown')})")
        else:
            print(f"{package}: missing")
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        print(
            f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}"
        )
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"VRAM: {free / 1024**3:.2f} / {total / 1024**3:.2f} GB free")
    print(f"Free disk GB: {shutil.disk_usage('.').free / 1024**3:.2f}")
    print(
        "\nInstall in Python 3.11 GPU environment with:\n"
        "python -m pip install -r requirements-transformers.txt"
    )


if __name__ == "__main__":
    main()
