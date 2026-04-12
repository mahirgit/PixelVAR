.PHONY: install test lint smoke clean

install:
	pip install -e ".[dev]"

test:
	python3 -m pytest tests/ -v

lint:
	python3 -m ruff check pixelvar/ scripts/ tests/

smoke:
	python3 -c "from pixelvar.data.palette import PaletteExtractor; from pixelvar.data.dataset import PixelArtDataset; print('OK: imports pass')"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache dist build *.egg-info
