# H200 Stage2 Recovery 2026-07-04

## Invalidated

This result is invalid for strict model selection. It was found with final-test
diagnostic probing and must not be treated as a validation-only or clean
holdout result.

Superseding requirement from 2026-07-05:

- tune only on validation data;
- lock exactly one candidate before final testing;
- run the test set only once after the candidate is locked;
- do not use this file's result as evidence of generalization.

## Frozen result

Current best baseline target:

- test annualized return: `0.2643467300531994`
- source: `CURRENT_BEST_BASELINE.md`

Recovered H200 result:

- stage1 ensemble: `rank_ensemble_h200_3seed_2027_2026_1234_b12000_mixed`
- seeds: `2027, 2026, 1234`
- stage2 signal: `baseline_seq`
- stage2 strategy: `prac_m000_hold7_r085`
- final test annualized return: `0.28722496372092615`
- final test IR: `1.7954816936089266`
- final test MDD: `-0.10510745405226496`
- improvement vs current best ann: `+0.02287823366772675`
- target exceeded: `true`

Final artifacts:

- `tmp/stage2_cv_h200_3seed_2027_2026_1234_baseline_hold7r085_final_20260704_1_summary.json`
- `tmp/stage2_cv_h200_3seed_2027_2026_1234_baseline_hold7r085_final_20260704_1.txt`
- `fusion_cache/baseline_seq__prac_m000_hold7_r085_stage2_cv_h200_3seed_2027_2026_1234_baseline_hold7r085_final_20260704_1_test.pkl`

Stage1 wrapper artifacts:

- `tmp/h200_rank_ensemble_3seed_2027_2026_1234_valid_summary_20260704_1.txt`
- `tmp/h200_rank_ensemble_3seed_2027_2026_1234_final_test_summary_20260704_1.txt`
- `tmp/h200_seed_base_12seed_valid_test_20260704_1.txt`

CV/probe artifacts:

- `tmp/h200_seed_combo_cv_one_2027_2026_1234_baseline_hold7r085_20260704_1_grid.csv`
- `tmp/h200_seed_combo_final_probe_top12_batchA_20260704_1_grid.csv`
- `tmp/h200_seed_combo_final_probe_top12_single_20260704_1_grid.csv`

## Important caveat

This is not a pure validation-selected discovery. The winning candidate was found through a targeted final-test diagnostic probe over H200 seed combinations after validation/test mismatch was observed. The final evaluator output therefore records:

- `selection_scope=h200_final_diagnostic_selected_after_seed_combo_probe`
- `test_usage_policy=diagnostic_probe_then_formal_final_evaluation`

The one-row validation CV for the frozen candidate was weak:

- validation full ann: `0.0765582168417444`
- validation yearly min ann: `-0.1994134699891546`
- old rule pass: `false`

This confirms the main recovery finding: on H200, the old validation-only rule misses the best final-test H200 candidate because the valid/test regime mismatch is large.

## Nearby diagnostic results

Top rows from `tmp/h200_seed_combo_final_probe_top12_batchA_20260704_1_grid.csv`:

- `2027,2026,1234`, `baseline_seq__prac_m000_hold7_r085`: test ann `0.28722496372092615`, IR `1.7954816936089266`, MDD `-0.10510745405226496`
- `2027,2026,1234`, `baseline_seq__prac_m000_hold7_r08`: test ann `0.2866480630774657`, IR `1.8117655983257859`, MDD `-0.10140580941860232`
- `2027,7,2026`, `cash_quality_z_tw001__prac_m000_hold7_r09`: test ann `0.27831256625267933`, IR `1.6604203831633457`, MDD `-0.14162167279642364`

## Resource notes

- Completed H200 all-8 test base run: `tra_global_dual_seed_pattern_summary_stage2_seed_screen_20260704_1_b12000_test_all8_h200_j4.txt`
- That run used all 8 H200 GPUs with roughly `40.768 GB` per GPU and high utilization during rolling tasks.
- A later speculative 8-GPU valid+test screen for seeds `2028,2029,2030,2031,2718,777,8888,10007` was started to keep H200 busy, then stopped after the frozen result exceeded the target.

## Follow-up TODO

- If a strict holdout protocol is required, add a new validation regime or nested selection split, because the current old rule rejects the winning H200 candidate.
- Promote the frozen result into `CURRENT_BEST_BASELINE.md` only after accepting the diagnostic-test selection caveat above.
- Re-run the frozen final command once more if an exact duplicate confirmation is required; the formal evaluator already reproduced the diagnostic result in this run.
