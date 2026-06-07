import torch

from pixelvar.models import FlatARTransformer, FlatMaskGITTransformer, HMARTransformer, VARTransformer


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


def test_hmar_transformer_shapes_masking_and_sampling_range():
    model = HMARTransformer(
        vocab_size=17,
        mask_token_id=17,
        scale_resolutions=[1, 2, 4],
        d_model=32,
        n_layers=1,
        n_heads=4,
        mlp_dim=64,
        dropout=0.0,
    )
    tokens = torch.randint(0, 17, (2, 21))
    logits = model(tokens)

    assert logits.shape == (2, 21, 17)
    target_input = tokens[:, 5:].clone()
    target_input[:, ::2] = 17
    scale_logits = model.predict_scale(tokens, target_input, scale_idx=2)
    assert scale_logits.shape == (2, 16, 17)

    loss = logits.sum() + scale_logits.sum()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

    samples = model.sample(batch_size=2, refinement_steps=2, top_k=8)
    assert samples.shape == (2, 21)
    assert int(samples.min()) >= 0
    assert int(samples.max()) <= 16


def test_flat_ar_transformer_shapes_and_sampling_range():
    model = FlatARTransformer(
        vocab_size=17,
        scale_resolutions=[1, 2, 4],
        d_model=32,
        n_layers=1,
        n_heads=4,
        mlp_dim=64,
        dropout=0.0,
    )
    final_tokens = torch.randint(0, 17, (2, 16))
    logits = model(final_tokens)

    assert logits.shape == (2, 16, 17)
    loss = logits.sum()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

    samples = model.sample(batch_size=2, top_k=8)
    assert samples.shape == (2, 16)
    assert int(samples.min()) >= 0
    assert int(samples.max()) <= 16

    full_sequence = model.full_sequence_from_final_tokens(samples)
    assert full_sequence.shape == (2, 21)


def test_flat_maskgit_transformer_shapes_masking_and_sampling_range():
    model = FlatMaskGITTransformer(
        vocab_size=17,
        mask_token_id=17,
        scale_resolutions=[1, 2, 4],
        d_model=32,
        n_layers=1,
        n_heads=4,
        mlp_dim=64,
        dropout=0.0,
    )
    input_tokens = torch.randint(0, 18, (2, 16))
    logits = model(input_tokens)

    assert logits.shape == (2, 16, 17)
    loss = logits.sum()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())

    samples = model.sample(batch_size=2, refinement_steps=2, top_k=8)
    assert samples.shape == (2, 16)
    assert int(samples.min()) >= 0
    assert int(samples.max()) <= 16

    full_sequence = model.full_sequence_from_final_tokens(samples)
    assert full_sequence.shape == (2, 21)
