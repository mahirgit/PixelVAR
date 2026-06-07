import torch

from pixelvar.models import VQVAE


def test_vqvae_forward_shapes_and_decode_code_indices():
    model = VQVAE(
        hidden_channels=32,
        embedding_dim=16,
        num_codes=32,
        latent_size=8,
        residual_blocks=1,
    )
    images = torch.rand(2, 4, 32, 32)

    output = model(images)
    decoded = model.decode_code_indices(output.code_indices)

    assert output.recon.shape == images.shape
    assert output.code_indices.shape == (2, 8, 8)
    assert int(output.code_indices.min()) >= 0
    assert int(output.code_indices.max()) < 32
    assert decoded.shape == images.shape
    assert torch.isfinite(output.loss_vq)
    assert torch.isfinite(output.loss_commit)
