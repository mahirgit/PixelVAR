# var_pokemon_overfit32

## Artifacts
- Best checkpoint: `/teamspace/studios/this_studio/PixelVAR/checkpoints/var_pokemon_overfit32/best.ckpt`
- Last checkpoint: `/teamspace/studios/this_studio/PixelVAR/checkpoints/var_pokemon_overfit32/last.ckpt`

## Config
```
{'run_name': 'var_pokemon_overfit32', 'seed': 42, 'output_dir': 'outputs/runs', 'checkpoint_dir': 'checkpoints', 'checkpoint_monitor': 'train/loss_epoch', 'log_lr': True, 'data': {'processed_dir': 'data/processed/pokemon', 'batch_size': 16, 'num_workers': 0, 'scale_resolutions': [1, 2, 4, 8, 16, 32], 'max_train_samples': 32, 'max_val_samples': 32, 'return_rgb': False}, 'model': {'vocab_size': 17, 'scale_resolutions': [1, 2, 4, 8, 16, 32], 'd_model': 256, 'n_layers': 6, 'n_heads': 8, 'mlp_dim': 1024, 'dropout': 0.1}, 'optimizer': {'lr': 0.0003, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'max_epochs': 200}, 'trainer': {'accelerator': 'auto', 'devices': 'auto', 'precision': '32-true', 'max_epochs': 200, 'log_every_n_steps': 1, 'gradient_clip_val': 1.0, 'limit_val_batches': 0, 'num_sanity_val_steps': 0}, 'logger': {'name': 'csv'}}
```

## Notes
- Option A deterministic tokenizer training run.
- Inspect per-scale loss and accuracy before promoting to the next ladder stage.