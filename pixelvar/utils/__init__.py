"""Utility helpers for PixelVAR."""

__all__ = ["load_yaml", "save_rgba_grid", "tokens_to_rgba"]


def __getattr__(name):
    if name == "load_yaml":
        from pixelvar.utils.config import load_yaml

        return load_yaml
    if name in {"save_rgba_grid", "tokens_to_rgba"}:
        from pixelvar.utils.render import save_rgba_grid, tokens_to_rgba

        return {"save_rgba_grid": save_rgba_grid, "tokens_to_rgba": tokens_to_rgba}[name]
    raise AttributeError(name)
