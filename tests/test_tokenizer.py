import torch

from pixelvar.tokenizers import DeterministicPyramidTokenizer


def test_deterministic_pyramid_shapes_roundtrip_and_tie_break():
    tokenizer = DeterministicPyramidTokenizer()
    index_map = torch.arange(32 * 32, dtype=torch.long).reshape(32, 32) % 17
    index_map[:16, :16] = 3
    index_map[:16, 16:] = 4

    maps = tokenizer.encode(index_map)
    sequence = tokenizer.to_sequence(maps)
    restored = tokenizer.from_sequence(sequence)

    assert [tuple(m.shape) for m in maps] == [(1, 1), (2, 2), (4, 4), (8, 8), (16, 16), (32, 32)]
    assert sequence.shape == (1365,)
    assert torch.equal(restored[-1], index_map)
    assert torch.equal(tokenizer.decode(restored), index_map)

    tied = torch.zeros((32, 32), dtype=torch.long)
    tied[:16, :16] = torch.tensor([[2, 3], [3, 2]]).repeat(8, 8)
    assert tokenizer.encode(tied)[1][0, 0].item() == 2
