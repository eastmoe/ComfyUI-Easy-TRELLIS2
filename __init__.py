import logging
import pathlib

log = logging.getLogger("trellis2")
log.info("loading...")


def _stage_sparse_attention_modules():
    import comfy_sparse_attn
    from comfy_sparse_attn import setup_link

    package_dir = pathlib.Path(comfy_sparse_attn.__file__).parent
    setup_link(package_dir / "sparse.py", "sparse.py")
    setup_link(package_dir / "ops_sparse.py", "ops_sparse.py")
    setup_link(package_dir / "attention_sparse.py", "attention_sparse.py")


def _register_trellis2_model_configs():
    import comfy.supported_models
    from .nodes.trellis2.supported_models import TRELLIS2SparseStructure, TRELLIS2SLat

    models = comfy.supported_models.models
    for index, model_config in enumerate((TRELLIS2SparseStructure, TRELLIS2SLat)):
        if model_config not in models:
            models.insert(index, model_config)

    log.info("registered TRELLIS2 model configs with ComfyUI")


_stage_sparse_attention_modules()

try:
    _register_trellis2_model_configs()
except Exception as e:
    log.warning(f"failed to register TRELLIS2 model configs: {e}")

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402
log.info("registered nodes")


WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
