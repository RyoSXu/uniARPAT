"""Public model names, loaded only when requested.

Importing a standalone local component must not load the DOS model, trainer
utilities or historical Encoder. Existing ``from model import ...`` names
continue to resolve to the original classes.
"""

from importlib import import_module

__all__ = ["basemodel", "Transformer"]


def __getattr__(name):
    modules = {"basemodel": ".model", "Transformer": ".transformer"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
