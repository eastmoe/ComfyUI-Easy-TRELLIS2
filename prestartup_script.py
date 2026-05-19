import os
from pathlib import Path
from shutil import copytree

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

SCRIPT_DIR = Path(__file__).resolve().parent
COMFYUI_DIR = SCRIPT_DIR.parent.parent


def copy_files(src: Path, dst: Path) -> None:
    if src.exists():
        copytree(src, dst, dirs_exist_ok=True)


# Copy sample assets into ComfyUI's input folder for the example workflows.
copy_files(SCRIPT_DIR / "assets", COMFYUI_DIR / "input")
copy_files(SCRIPT_DIR / "assets" / "3d", COMFYUI_DIR / "input" / "3d")
