from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class ModelRegistration:
    name: str
    cls: type
    metadata: dict[str, Any]


MODEL_REGISTRY: dict[str, ModelRegistration] = {}


def register(name: str, **metadata: Any) -> Callable[[type], type]:
    """Register local checkpoint model classes without changing class creation."""

    def decorator(cls: type) -> type:
        MODEL_REGISTRY[name] = ModelRegistration(name=name, cls=cls, metadata=dict(metadata))
        return cls

    return decorator
