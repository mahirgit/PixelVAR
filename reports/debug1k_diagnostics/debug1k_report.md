# var_pokemon_debug1k

## Artifacts
- Best checkpoint: `/teamspace/studios/this_studio/PixelVAR/checkpoints/var_pokemon_debug1k/best.ckpt`
- Last checkpoint: `/teamspace/studios/this_studio/PixelVAR/checkpoints/var_pokemon_debug1k/last.ckpt`

## Config
```
{'run_name': 'var_pokemon_debug1k', 'seed': 42, 'output_dir': 'outputs/runs', 'checkpoint_dir': 'checkpoints', 'checkpoint_monitor': 'val/loss', 'log_lr': True, 'data': {'processed_dir': 'data/processed/pokemon', 'batch_size': 64, 'num_workers': 2, 'scale_resolutions': [1, 2, 4, 8, 16, 32], 'max_train_samples': 1000, 'max_val_samples': 256, 'return_rgb': False}, 'model': {'vocab_size': 17, 'scale_resolutions': [1, 2, 4, 8, 16, 32], 'd_model': 256, 'n_layers': 6, 'n_heads': 8, 'mlp_dim': 1024, 'dropout': 0.1}, 'optimizer': {'lr': 0.0003, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'scheduler': 'cosine', 'max_epochs': 100}, 'trainer': {'accelerator': 'auto', 'devices': 'auto', 'precision': '32-true', 'max_epochs': 100, 'log_every_n_steps': 10, 'gradient_clip_val': 1.0, 'limit_val_batches': 1.0, 'num_sanity_val_steps': 1}, 'logger': {'name': 'csv'}}
```

## Notes
- Option A deterministic tokenizer training run.
- Inspect per-scale loss and accuracy before promoting to the next ladder stage.