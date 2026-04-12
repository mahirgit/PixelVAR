"""
Unit tests for the deterministic pyramid tokenizer.

Tests encode/decode roundtrips, sequence flattening, vocab ranges,
and transparent token handling.
"""

import pytest
import torch

from pixelvar.tokenizers import DeterministicTokenizer
from pixelvar.tokenizers.base import BaseTokenizer


# ---------- fixtures ----------

@pytest.fixture
def tokenizer():
    return DeterministicTokenizer(palette_size=16)


@pytest.fixture
def sample_index_map():
    """A random 32x32 index map with values 0-16 (0=transparent, 1-16=palette)."""
    torch.manual_seed(42)
    return torch.randint(0, 17, (4, 32, 32))


@pytest.fixture
def uniform_index_map():
    """A 32x32 map where every pixel is the same index — mode-pooling should preserve it."""
    return torch.full((2, 32, 32), fill_value=5, dtype=torch.long)


# ---------- interface tests ----------

class TestTokenizerInterface:
    def test_is_base_tokenizer(self, tokenizer):
        assert isinstance(tokenizer, BaseTokenizer)

    def test_scale_resolutions(self, tokenizer):
        assert tokenizer.scale_resolutions == [1, 2, 4, 8, 16, 32]

    def test_total_tokens(self, tokenizer):
        assert tokenizer.total_tokens == 1 + 4 + 16 + 64 + 256 + 1024

    def test_total_tokens_is_1365(self, tokenizer):
        assert tokenizer.total_tokens == 1365

    def test_vocab_size(self, tokenizer):
        # 16 palette + 1 transparent = 17
        assert tokenizer.vocab_size == 17

    def test_scale_lengths(self, tokenizer):
        assert tokenizer.scale_lengths == [1, 4, 16, 64, 256, 1024]

    def test_scale_offsets(self, tokenizer):
        assert tokenizer.scale_offsets == [0, 1, 5, 21, 85, 341]

    def test_get_scale_info(self, tokenizer):
        info = tokenizer.get_scale_info()
        assert len(info) == 6
        assert info[0]["resolution"] == 1
        assert info[0]["num_tokens"] == 1
        assert info[-1]["resolution"] == 32
        assert info[-1]["num_tokens"] == 1024


# ---------- encode tests ----------

class TestEncode:
    def test_output_shapes(self, tokenizer, sample_index_map):
        maps = tokenizer.encode(sample_index_map)
        assert len(maps) == 6
        B = sample_index_map.shape[0]
        for m, res in zip(maps, [1, 2, 4, 8, 16, 32]):
            assert m.shape == (B, res, res), f"Scale {res}: expected ({B},{res},{res}), got {m.shape}"

    def test_finest_scale_is_identity(self, tokenizer, sample_index_map):
        maps = tokenizer.encode(sample_index_map)
        assert torch.equal(maps[-1], sample_index_map)

    def test_vocab_range(self, tokenizer, sample_index_map):
        maps = tokenizer.encode(sample_index_map)
        for m in maps:
            assert m.min() >= 0, f"Token below 0: {m.min()}"
            assert m.max() < tokenizer.vocab_size, f"Token >= vocab_size: {m.max()}"

    def test_uniform_map_preserved(self, tokenizer, uniform_index_map):
        maps = tokenizer.encode(uniform_index_map)
        for m in maps:
            assert (m == 5).all(), f"Uniform map not preserved at scale {m.shape}"

    def test_transparent_token_preserved(self, tokenizer):
        """If entire image is transparent (token 0), all scales should be 0."""
        transparent = torch.zeros(1, 32, 32, dtype=torch.long)
        maps = tokenizer.encode(transparent)
        for m in maps:
            assert (m == 0).all(), f"Transparent not preserved at scale {m.shape}"


# ---------- decode tests ----------

class TestDecode:
    def test_decode_returns_finest(self, tokenizer, sample_index_map):
        maps = tokenizer.encode(sample_index_map)
        decoded = tokenizer.decode(maps)
        assert torch.equal(decoded, sample_index_map)

    def test_decode_shape(self, tokenizer, sample_index_map):
        maps = tokenizer.encode(sample_index_map)
        decoded = tokenizer.decode(maps)
        assert decoded.shape == sample_index_map.shape


# ---------- sequence tests ----------

class TestSequence:
    def test_to_sequence_shape(self, tokenizer, sample_index_map):
        maps = tokenizer.encode(sample_index_map)
        seq = tokenizer.to_sequence(maps)
        B = sample_index_map.shape[0]
        assert seq.shape == (B, 1365)

    def test_to_sequence_vocab_range(self, tokenizer, sample_index_map):
        maps = tokenizer.encode(sample_index_map)
        seq = tokenizer.to_sequence(maps)
        assert seq.min() >= 0
        assert seq.max() < tokenizer.vocab_size

    def test_sequence_roundtrip(self, tokenizer, sample_index_map):
        """to_sequence -> from_sequence should be exact."""
        maps = tokenizer.encode(sample_index_map)
        seq = tokenizer.to_sequence(maps)
        recovered = tokenizer.from_sequence(seq)

        assert len(recovered) == len(maps)
        for orig, rec in zip(maps, recovered):
            assert torch.equal(orig, rec), f"Mismatch at scale {orig.shape}"

    def test_full_roundtrip(self, tokenizer, sample_index_map):
        """encode -> to_sequence -> from_sequence -> decode should recover input."""
        maps = tokenizer.encode(sample_index_map)
        seq = tokenizer.to_sequence(maps)
        recovered_maps = tokenizer.from_sequence(seq)
        decoded = tokenizer.decode(recovered_maps)
        assert torch.equal(decoded, sample_index_map)


# ---------- edge cases ----------

class TestEdgeCases:
    def test_single_sample(self, tokenizer):
        single = torch.randint(0, 17, (1, 32, 32))
        maps = tokenizer.encode(single)
        seq = tokenizer.to_sequence(maps)
        assert seq.shape == (1, 1365)

    def test_custom_palette_size(self):
        tok = DeterministicTokenizer(palette_size=32)
        assert tok.vocab_size == 33  # 32 palette + 1 transparent
        assert tok.MASK_TOKEN == 33

    def test_custom_scales(self):
        tok = DeterministicTokenizer(
            scale_resolutions=[1, 4, 16, 32], palette_size=16
        )
        assert tok.total_tokens == 1 + 16 + 256 + 1024
        assert len(tok.scale_resolutions) == 4
