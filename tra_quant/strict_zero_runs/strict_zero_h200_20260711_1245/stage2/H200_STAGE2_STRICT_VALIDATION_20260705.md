# H200 Stage2 Strict Validation-Only Final Test - 2026-07-05

## Protocol

- This rerun invalidates the earlier test-probe based H200 stage2 result for model selection.
- Validation/test windows were unchanged:
  - train: 2012-01-01 to 2019-12-31
  - valid: 2020-01-01 to 2022-12-31
  - test: 2023-01-01 to 2026-04-01
- Selection was performed with validation artifacts only.
- The test summaries were loaded only after a locked manifest was written.
- The final test was run once for the locked candidate. No follow-up test-based tuning is valid.

## Validation Search

Artifacts:

- strict selector grid: `tmp/h200_strict_validation_selector_old_explicit_20260705_2_grid.csv`
- strict selector summary: `tmp/h200_strict_validation_selector_old_explicit_20260705_2_summary.json`
- balanced lock grid: `tmp/h200_strict_validation_selector_old_explicit_20260705_2_balanced_grid.csv`
- balanced lock summary: `tmp/h200_strict_validation_selector_old_explicit_20260705_2_balanced_summary.json`
- final locked manifest: `tmp/h200_strict_validation_selector_old_explicit_20260705_2_balanced_locked_manifest.json`

Balanced validation rule:

- `full_ann >= 0.20`
- `year_ann_min >= 0.15`
- `year_ann_std <= 0.10`
- `full_mdd >= -0.25`
- sort by `balanced_pass`, `robust_ann_min=min(full_ann, year_ann_min)`, `year_ann_min`, `full_ann`, `stable_score`, `full_ir`

Locked candidate:

- combo: `7,11,1234`
- trial: `baseline_seq__prac_m000_hold7_r08`
- valid full ann: `0.2811535073099664`
- valid worst-year ann: `0.2503678278335165`
- valid year ann std: `0.0402277943934076`
- valid full IR: `1.5066528729828834`
- valid full MDD: `-0.1246077483606773`

The default strict selector's original top row was `7,11,1234 + cash_quality_z_tw001__prac_m000_hold7_r085`, but it had lower full-period validation ann and weaker MDD. The balanced lock was used because it is a validation-only rule that avoids selecting a candidate with strong yearly slices but weak full-period ann.

## New Seed Screening

Additional H200 valid-only seed screening jobs were started for exploratory validation coverage:

- `stage2_strict_valid_only_new8_20260705_1_b12000_valid`
- `stage2_strict_valid_only_new8b_20260705_1_b12000_valid`
- `stage2_strict_valid_only_new8c_20260705_1_b12000_valid`

They were terminated before producing validation summary files and were not used for selection or final evaluation. This avoided expanding the validation search indefinitely after a complete validation-only lock existed.

## Final Test Once

Artifacts:

- final summary: `tmp/stage2_strict_h200_old_explicit_balanced_20260705_1_final_once_summary.json`
- final text: `tmp/stage2_strict_h200_old_explicit_balanced_20260705_1_final_once.txt`
- final audit: `tmp/stage2_strict_h200_old_explicit_balanced_20260705_1_final_once_final_audit.json`

Final test result:

- test ann: `0.05063787807364036`
- test IR: `0.32474041463439857`
- test MDD: `-0.20847715269845518`
- current best baseline ann: `0.2643467300531994`
- improves current best: `false`

## Conclusion

The strict validation-only flow did not reproduce or exceed the current best baseline. The locked candidate looked strong on validation, but failed on the one allowed final test.

Because the final test has now been consumed for this locked candidate, it must not be used for more tuning, filtering, or candidate selection. Further work should be validation-only until a genuinely new holdout period is available.
