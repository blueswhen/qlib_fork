# Current Best Baseline

This file mirrors `current_best_baseline.md` for compatibility with older
notes.

Frozen baseline:

- formal run: `strict_zero_runs/strict_zero_h200_20260711_1245`
- stage1 seeds: `5678,2050,2044`
- stage2 signal: `cash_quality_z_tw001`
- strategy: `prac_m000_hold7_r095`
- account: `150000`
- topk: `5`
- final test ann: `0.2804402793535583`
- final test ir: `1.6660541147890104`
- final test mdd: `-0.10195254346954019`

One-key retrain from `tra_quant`:

```bash
./train_strict_fusion_from_zero.sh --run-id strict_zero_repro_YYYYMMDD_1
```

Detailed protocol:

```text
../STRICT_FROM_ZERO_RETRAIN.md
```
