# hmar_sprites_v0_full

## Artifacts
- Best checkpoint: `/__modal/volumes/vo-AA7YnlPOI95ayXuOUZnbyz/hmar_sprites_v0_full/best.ckpt`
- Last checkpoint: `/__modal/volumes/vo-AA7YnlPOI95ayXuOUZnbyz/hmar_sprites_v0_full/last.ckpt`

## Config
```
{'run_name': 'hmar_sprites_v0_full', 'seed': 42, 'output_dir': 'outputs/runs', 'checkpoint_dir': 'checkpoints', 'checkpoint_monitor': 'val/loss', 'log_lr': True, 'matmul_precision': 'high', 'early_stopping': {'monitor': 'val/loss', 'mode': 'min', 'patience': 6, 'min_delta': 0.001}, 'data': {'processed_dir': 'data/processed/sprites', 'batch_size': 64, 'num_workers': 4, 'scale_resolutions': [1, 2, 4, 8, 16, 32], 'max_train_samples': None, 'max_val_samples': None, 'return_rgb': False}, 'model': {'vocab_size': 17, 'mask_token_id': 17, 'scale_resolutions': [1, 2, 4, 8, 16, 32], 'd_model': 256, 'n_layers': 6, 'n_heads': 8, 'mlp_dim': 1024, 'dropout': 0.1}, 'masking': {'mask_ratio_min': 0.1, 'mask_ratio_max': 1.0, 'full_mask_prob': 0.35, 'validation_mask_ratio': 1.0}, 'optimizer': {'lr': 0.0003, 'weight_decay': 0.01, 'betas': [0.9, 0.95], 'scheduler': 'cosine', 'max_epochs': 30}, 'trainer': {'accelerator': 'auto', 'devices': 'auto', 'precision': 'bf16-mixed', 'max_epochs': 30, 'log_every_n_steps': 10, 'gradient_clip_val': 1.0, 'limit_val_batches': 1.0, 'num_sanity_val_steps': 1}, 'logger': {'name': 'csv'}}
```

## Notes
- Option B HMAR masked-refinement training run.
- Validation loss masks whole target scales to test coarse-to-fine generation ability.
- Promote only if sampled/evaluated grids beat the real-only VAR baseline.