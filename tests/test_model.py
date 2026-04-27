import torch

from pixelvar.models import VARTransformer


def test_var_transformer_shapes_and_sampling_range():
    model = VARTransformer(
        vocab_size=17,
        d_model=32,
        n_layers=1,
        n_heads=4,
        mlp_dim=64,
        dropout=0.0,
    )
    tokens = torch.randint(0, 17, (2, 1365))
    logits = model(tokens)

    assert logits.shape == (2, 1365, 17)
    scale_logits = model.forward_by_scale(tokens)
    assert [x.shape[1] for x in scale_logits] == [1, 4, 16, 64, 256, 1024]

    loss = logits.sum()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

    samples = model.sample(batch_size=2, top_k=8)
    assert samples.shape == (2, 1365)
    assert int(samples.min()) >= 0
    assert int(samples.max()) <= 16
