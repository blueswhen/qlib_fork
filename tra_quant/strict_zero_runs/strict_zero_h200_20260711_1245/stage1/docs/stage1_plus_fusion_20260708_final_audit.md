# Stage1+ Fusion Final Audit 2026-07-08

## Protocol

- Hard constraints: `account=150000`, `topk=5`, universe `csi300`.
- Selection scope: validation only.
- Test policy: test is loaded only after a validation-locked candidate exists, and is not used for tuning or candidate selection.
- After the final test below, no further tuning on the same split is valid.

## Validation-Locked Candidate

- Signal: `growth_quality_linear_rank_w0001`
- Strategy: `prac_m000_drop3_hold4_r0875`
- Validation lock file: `tra_quant/stage1/tmp/stage1_plus_practical_risk_targeted_stress_20260708_1_stress_grid.csv`

Validation summary versus current H200 Stage1 baseline `baseline_seq + prac_m000_drop3_hold4_r095`:

| metric | baseline | candidate | delta |
| --- | ---: | ---: | ---: |
| full ann | 0.222894 | 0.236542 | +0.013648 |
| year ann min | 0.066731 | 0.074255 | +0.007525 |
| year ann LCB | 0.129972 | 0.134920 | +0.004948 |
| full MDD | -0.216983 | -0.220212 | -0.003230 |
| quarter ann min | -0.842950 | -0.829907 | +0.013044 |
| quarter ann LCB | -0.244859 | -0.243405 | +0.001454 |
| quarter positive ratio | 0.666667 | 0.750000 | +0.083333 |
| quarter Rank IC positive ratio | 0.916667 | 0.916667 | 0.000000 |

Validation gate result: `pass`.

## Final Test Once

- Final result file: `tra_quant/stage1/tmp/stage1_plus_growth_prac_r0875_final_test_once_20260708_1_summary.json`
- Candidate was loaded from the validation lock file before test cache was loaded.
- Final test result:
  - annualized return: 0.157715
  - information ratio: 1.011632
  - max drawdown: -0.153647
  - IC: 0.042707
  - Rank IC: 0.066851

Result versus the prior `0.26` reference: not exceeded.

## Conclusion

This run found a validation-stable Stage1+ candidate under the fixed capital and top-k constraints, but the one allowed final test did not exceed the prior `0.26` reference. The final test result must not be used to continue tuning this same split.
