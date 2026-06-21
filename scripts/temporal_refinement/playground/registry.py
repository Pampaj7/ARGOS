from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Small explicit registry for composable research modules."""

    def __init__(self, name: str):
        self.name = name
        self._builders: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(builder: Callable[..., T]) -> Callable[..., T]:
            if name in self._builders:
                raise KeyError(f"{name!r} is already registered in {self.name}")
            self._builders[name] = builder
            return builder

        return decorator

    def build(self, name: str, **params: Any) -> T:
        if name not in self._builders:
            available = ", ".join(sorted(self._builders))
            raise KeyError(f"Unknown {self.name} module {name!r}. Available: {available}")
        return self._builders[name](**params)

    def names(self) -> list[str]:
        return sorted(self._builders)
