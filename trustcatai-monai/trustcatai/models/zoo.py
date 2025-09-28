from typing import Callable, Dict

from .unet3d import build_unet3d

_MODEL_REGISTRY: Dict[str, Callable[..., object]] = {
    "unet3d": build_unet3d,
}


def get_model(name: str, **kwargs):
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    return _MODEL_REGISTRY[name](**kwargs)
