from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
TRA_CACHE_DIR = BASE_DIR / "tra_cache"
DEFAULT_SUFFIX = "seed42_refcheck_retrain_20260620_1"

REFERENCE_VALID = (
    TRA_CACHE_DIR
    / "rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r07_"
    "strict_tra_lstm_s240_h5_tra18_residual_step1_20260506_1.pkl"
)
REFERENCE_TEST = (
    TRA_CACHE_DIR
    / "rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_r085_"
    "strict_tra_lstm_s240_h5_final_test_tra18_residual_step1_20260506_1.pkl"
)


def cache_path(suffix: str, segment: str) -> Path:
    risk_tag = "r07" if segment == "valid" else "r085"
    return (
        TRA_CACHE_DIR
        / f"rolling_cache_tra_w1_tra_lstm_base_Alpha360_step240_k5_d1_a150000_{risk_tag}_{suffix}_{segment}.pkl"
    )


def _frame(obj) -> pd.DataFrame:
    if isinstance(obj, pd.Series):
        return obj.to_frame()
    if isinstance(obj, pd.DataFrame):
        return obj
    raise TypeError(f"expected Series/DataFrame, got {type(obj)!r}")


def _aligned_values(left: pd.DataFrame, right: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    left, right = left.align(right, join="inner", axis=0)
    left, right = left.align(right, join="inner", axis=1)
    if left.shape != right.shape:
        raise ValueError(f"aligned shapes differ: {left.shape} vs {right.shape}")
    return left.to_numpy(dtype="float64"), right.to_numpy(dtype="float64")


def compare_frame(name: str, left: pd.DataFrame, right: pd.DataFrame) -> dict[str, float | bool | tuple[int, int]]:
    left_values, right_values = _aligned_values(left, right)
    mask = ~(np.isnan(left_values) & np.isnan(right_values))
    diff = left_values[mask] - right_values[mask]
    finite_left = left_values[mask]
    finite_right = right_values[mask]
    corr = float(np.corrcoef(finite_left, finite_right)[0, 1]) if finite_left.size > 1 else float("nan")
    return {
        "name": name,
        "shape": left.shape,
        "allclose": bool(np.allclose(left_values, right_values, rtol=0.0, atol=0.0, equal_nan=True)),
        "max_abs": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "mean_abs": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "corr": corr,
    }


def compare_cache(segment: str, reference_path: Path, generated_path: Path) -> list[dict]:
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)
    if not generated_path.exists():
        raise FileNotFoundError(generated_path)
    reference = pd.read_pickle(reference_path)
    generated = pd.read_pickle(generated_path)
    rows = []
    for key in ("pred", "label"):
        rows.append(
            compare_frame(
                f"{segment}.{key}",
                _frame(reference[key]),
                _frame(generated[key]),
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify a regenerated seed42 cache against the frozen stage1 baseline.")
    parser.add_argument("--suffix", default=DEFAULT_SUFFIX)
    parser.add_argument("--output-path", default=str(BASE_DIR / f"{DEFAULT_SUFFIX}_cache_comparison.txt"))
    args = parser.parse_args()

    rows = []
    rows.extend(compare_cache("valid", REFERENCE_VALID, cache_path(args.suffix, "valid")))
    rows.extend(compare_cache("test", REFERENCE_TEST, cache_path(args.suffix, "test")))

    lines = []
    ok = True
    for row in rows:
        ok = ok and bool(row["allclose"])
        lines.append(
            "{name}: shape={shape} allclose={allclose} max_abs={max_abs:.12g} "
            "mean_abs={mean_abs:.12g} corr={corr:.12g}".format(**row)
        )
    lines.append(f"verification_passed={ok}")

    output_path = Path(args.output_path).expanduser().resolve()
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"saved comparison: {output_path}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
