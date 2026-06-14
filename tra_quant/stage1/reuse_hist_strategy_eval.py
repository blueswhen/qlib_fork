from __future__ import annotations

import argparse
from ast import literal_eval
from pathlib import Path

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.workflow.record_temp import PortAnaRecord, SigAnaRecord

try:
    from examples.my_strategy.rolling_hist_alpha360_latest import PROVIDER_URI, _metric_value, build_base_task
    from examples.my_strategy.tune_hist_alpha360_global import _write_final_test
except ModuleNotFoundError:
    from rolling_hist_alpha360_latest import PROVIDER_URI, _metric_value, build_base_task
    from tune_hist_alpha360_global import _write_final_test


BASE_DIR = Path(__file__).resolve().parent


def _parse_summary(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value
    return data


def _normalize_best_summary(summary: dict[str, str]) -> dict[str, str]:
    normalized = dict(summary)
    if "score" not in normalized and "validation_score" in normalized:
        normalized["score"] = normalized["validation_score"]
    if "with_cost_ann_return" not in normalized and "validation_with_cost_ann_return" in normalized:
        normalized["with_cost_ann_return"] = normalized["validation_with_cost_ann_return"]
    if "with_cost_ir" not in normalized and "validation_with_cost_ir" in normalized:
        normalized["with_cost_ir"] = normalized["validation_with_cost_ir"]
    if "with_cost_mdd" not in normalized and "validation_with_cost_mdd" in normalized:
        normalized["with_cost_mdd"] = normalized["validation_with_cost_mdd"]
    if "IC" not in normalized and "validation_IC" in normalized:
        normalized["IC"] = normalized["validation_IC"]
    if "Rank IC" not in normalized and "validation_Rank_IC" in normalized:
        normalized["Rank IC"] = normalized["validation_Rank_IC"]
    return normalized


def _load_signal(cache_path: str | Path):
    cache = pd.read_pickle(cache_path)
    pred = cache["pred"].sort_index()
    label = cache["label"].sort_index()
    common_index = pred.index.intersection(label.index)
    return pred.loc[common_index].sort_index(), label.loc[common_index].sort_index()


def _evaluate_from_cache(best_summary: dict[str, str], cache_path: str, recorder_name: str) -> dict:
    qlib.init(provider_uri=PROVIDER_URI, region=REG_CN)

    task = build_base_task(
        topk=5,
        n_drop=int(best_summary["n_drop"]),
        window_key=best_summary["window_key"],
        model_key=best_summary["model_key"],
        handler_class="Alpha360",
        handler_module_path="qlib.contrib.data.handler",
        account=150000,
        risk_degree=float(best_summary["risk_degree"]),
        benchmark="SH000300",
        instruments="csi300",
        model_kwargs_override=literal_eval(best_summary["model_kwargs_override"]),
        handler_kwargs_extra=literal_eval(best_summary["handler_kwargs_extra"]),
        strategy_class=best_summary["strategy_class"],
        strategy_module_path=best_summary["strategy_module_path"],
        strategy_kwargs_extra=literal_eval(best_summary["kwargs_extra"]),
        eval_segment="test",
        backtest_segment="test",
    )
    pred, label = _load_signal(cache_path)
    port_record = task["record"][2]

    with R.start(experiment_name="hist_global_reuse_strategy_eval", recorder_name=recorder_name):
        rec = R.get_recorder()
        rec.save_objects(**{"pred.pkl": pred, "label.pkl": label})
        SigAnaRecord(recorder=rec, ana_long_short=False, ann_scaler=252).generate()
        PortAnaRecord(recorder=rec, **port_record["kwargs"]).generate()
        metrics = rec.list_metrics()

    return {
        "recorder_id": rec.id,
        "IC": metrics.get("IC"),
        "Rank IC": metrics.get("Rank IC"),
        "with_cost_ann_return": _metric_value(metrics, rec.id, "1day.excess_return_with_cost.annualized_return"),
        "with_cost_ir": _metric_value(metrics, rec.id, "1day.excess_return_with_cost.information_ratio"),
        "with_cost_mdd": _metric_value(metrics, rec.id, "1day.excess_return_with_cost.max_drawdown"),
    }


def main():
    parser = argparse.ArgumentParser(description="Reuse HIST base caches to rerun final strategy evaluation.")
    parser.add_argument("--validation-best", required=True, help="Path to the validation best summary file.")
    parser.add_argument("--source-final", required=True, help="Path to the old final summary file that contains test_cache_path.")
    parser.add_argument("--output", required=True, help="Path to write the refreshed final summary file.")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--recorder-name", default="")
    args = parser.parse_args()

    best_path = Path(args.validation_best).resolve()
    source_final_path = Path(args.source_final).resolve()
    output_path = Path(args.output).resolve()

    best_summary = _normalize_best_summary(_parse_summary(best_path))
    source_final = _parse_summary(source_final_path)
    recorder_name = args.recorder_name or f"reuse_hist_strategy_final_seed{args.seed}"
    test_result = _evaluate_from_cache(best_summary, source_final["test_cache_path"], recorder_name)
    test_result["result_path"] = str(output_path)
    test_result["cache_path"] = source_final["test_cache_path"]

    _write_final_test(
        best_summary,
        test_result,
        output_path=output_path,
        window_key=best_summary["window_key"],
        seed=args.seed,
    )

    print(f"validation_best={best_path}")
    print(f"source_test_cache={source_final['test_cache_path']}")
    print(f"output={output_path}")
    print(f"strategy_trial={best_summary['strategy_trial']}")
    print(f"risk_degree={best_summary['risk_degree']}")
    print(f"kwargs_extra={best_summary['kwargs_extra']}")
    print(f"test_recorder_id={test_result['recorder_id']}")
    print(f"test_IC={test_result['IC']}")
    print(f"test_Rank_IC={test_result['Rank IC']}")
    print(f"test_with_cost_ann_return={test_result['with_cost_ann_return']}")
    print(f"test_with_cost_ir={test_result['with_cost_ir']}")
    print(f"test_with_cost_mdd={test_result['with_cost_mdd']}")


if __name__ == "__main__":
    main()