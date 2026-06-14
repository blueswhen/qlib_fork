# tra_quant/stage2

This directory is reserved for stage2 management.

Stage2 means post-stage1 overlays or refinements that consume the fixed TRA
stage1 signal, such as:

- cash-quality fundamental overlay
- validation-only stage2 CV search
- final test-once evaluator
- stage2 fusion caches and summaries

Do not place TRA-only training, TRA cache provenance, or rank-ensemble stage1
artifacts here. Those belong to `../stage1`.

The current best stage2 baseline has not been migrated into this folder yet in
this restructuring pass; this placeholder keeps the top-level `tra_quant`
layout stable as:

```text
tra_quant/
  stage1/
  stage2/
```
