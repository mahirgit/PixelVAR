"""Smoke test: verify the package and its dependencies import correctly."""

import importlib


def test_pixelvar_imports():
    assert importlib.import_module("pixelvar") is not None


def test_pixelvar_data_imports():
    assert importlib.import_module("pixelvar.data") is not None
    assert importlib.import_module("pixelvar.data.palette") is not None
    assert importlib.import_module("pixelvar.data.dataset") is not None


def test_third_party_imports():
    for module in ["torch", "torchvision", "numpy", "PIL", "sklearn", "matplotlib"]:
        assert importlib.import_module(module) is not None, f"Failed to import {module}"


def test_palette_extractor_instantiation():
    from pixelvar.data.palette import PaletteExtractor

    extractor = PaletteExtractor(palette_size=16)
    assert extractor.palette_size == 16
    assert extractor.palette is None
