import numpy as np

from pixelvar.data.palette import PaletteExtractor


def test_quantize_with_transparency_reserves_token_zero():
    image = np.zeros((4, 4, 4), dtype=np.uint8)
    image[:, :, :3] = [200, 20, 20]
    image[:, :, 3] = 255
    image[:2, :2, 3] = 0

    palette = PaletteExtractor(palette_size=2)
    palette.fit([image])
    index_map, alpha_mask, preview = palette.quantize_with_transparency(image)

    assert np.all(index_map[:2, :2] == 0)
    assert np.all(index_map[2:, 2:] >= 1)
    assert np.all(index_map <= 2)
    assert not alpha_mask[:2, :2].any()
    assert alpha_mask[2:, 2:].all()
    assert np.all(preview[:2, :2, 3] == 0)
    assert np.all(preview[2:, 2:, 3] == 255)
