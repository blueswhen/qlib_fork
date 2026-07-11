# Current Best Baseline

This file mirrors `current_best_baseline.md` for compatibility with older notes.

Frozen baseline:

- stage1: `rank_ensemble_3seed_tra72_old6`
- stage2: `cash_quality_z_tw001`
- strategy: `prac_m000_hold7_r085`
- account: `150000`
- topk: `5`
- final test ann: `0.2643467300531994`
- final test ir: `1.8243593079319251`
- final test mdd: `-0.09423283807818256`

Reproduce from this directory:

```bash
cd /home/blueswhen/DL/qlib_fork/tra_quant/stage2
PYTHONPATH=/home/blueswhen/DL/qlib_fork:$PYTHONPATH python evaluate_stage2_cv_selected_final.py
```

Data is expected under:

- `../../training_data/cn_data_latest`
- `../../training_data/processed/training`

The detailed reliability review, CV rule, artifacts, and cautions are in
`current_best_baseline.md`.
