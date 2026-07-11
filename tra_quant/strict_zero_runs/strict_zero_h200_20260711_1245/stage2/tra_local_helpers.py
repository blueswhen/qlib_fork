from __future__ import annotations

from ast import literal_eval
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FUSION_CACHE_DIR = BASE_DIR / "fusion_cache"
LOCAL_DATA_DIR = (BASE_DIR.parent.parent / "training_data" / "cn_data_latest").resolve()

_SOURCE_WORKSPACE = "/home/blueswhen/DL/qlib/examples/my_strategy"
_SOURCE_NIUSHENGXIAO_WORKSPACE = "/home/niushengxiao/qlib_fork"
_SOURCE_DATA_DIR = "/home/blueswhen/.qlib/qlib_data/cn_data_latest"
_SOURCE_DATA_ALIAS = "~/.qlib/qlib_data/cn_data_latest"
_LOCAL_WORKSPACE = BASE_DIR.parent.parent.resolve()


def _replace_prefix(raw: str, old_prefix: str, new_prefix: str) -> str:
    if raw == old_prefix:
        return new_prefix
    if raw.startswith(old_prefix + "/"):
        return new_prefix + raw[len(old_prefix) :]
    return raw


def localize_path_string(raw: str) -> str:
    scheme = ""
    value = raw
    if raw.startswith("file://"):
        scheme = "file://"
        value = raw[len(scheme) :]

    value = _replace_prefix(value, _SOURCE_DATA_ALIAS, str(LOCAL_DATA_DIR))
    value = _replace_prefix(value, _SOURCE_DATA_DIR, str(LOCAL_DATA_DIR))
    value = _replace_prefix(value, _SOURCE_WORKSPACE, str(BASE_DIR))
    value = _replace_prefix(value, _SOURCE_NIUSHENGXIAO_WORKSPACE, str(_LOCAL_WORKSPACE))
    return scheme + value


def localize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: localize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [localize_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(localize_value(item) for item in value)
    if isinstance(value, str):
        return localize_path_string(value)
    return value


def _parse_scalar(value: str):
    try:
        return literal_eval(value)
    except Exception:
        return value


def _parse_summary_file(path: Path) -> dict:
    data = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = localize_value(_parse_scalar(value))
    return data


def _ensure_score_df(df: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(df, pd.Series):
        return df.to_frame("score")
    if "score" in df.columns:
        return df[["score"]].copy()
    return df.iloc[:, [0]].rename(columns={df.columns[0]: "score"})


def _cross_sectional_zscore(df: pd.DataFrame) -> pd.DataFrame:
    def _normalize(group: pd.DataFrame) -> pd.DataFrame:
        score = group["score"]
        std = score.std(ddof=0)
        if std is None or pd.isna(std) or std == 0:
            return pd.DataFrame({"score": pd.Series(0.0, index=group.index)})
        return pd.DataFrame({"score": (score - score.mean()) / std})

    return df.groupby(level="datetime", group_keys=False).apply(_normalize)


def _cross_sectional_rank_pct(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(level="datetime", group_keys=False).apply(
        lambda group: pd.DataFrame({"score": group["score"].rank(pct=True)})
    )


def _normalize_scores(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "zscore":
        return _cross_sectional_zscore(df)
    if mode == "rank_pct":
        return _cross_sectional_rank_pct(df)
    raise ValueError(f"unsupported fusion mode: {mode}")


def _artifact_path_from_recorder(recorder_id: str, artifact_name: str) -> Path:
    matches = list((BASE_DIR / "mlruns").glob(f"*/{recorder_id}/artifacts/{artifact_name}"))
    if not matches:
        raise FileNotFoundError(f"artifact `{artifact_name}` not found for recorder `{recorder_id}`")
    return matches[0]


def _load_signal_from_cache(cache_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = pd.read_pickle(localize_path_string(str(cache_path)))
    pred = _ensure_score_df(cache["pred"]).sort_index()
    label = cache["label"].sort_index()
    common_index = pred.index.intersection(label.index)
    return pred.loc[common_index].sort_index(), label.loc[common_index].sort_index()


def _load_signal_from_recorder(recorder_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred_path = _artifact_path_from_recorder(recorder_id, "pred.pkl")
    label_path = _artifact_path_from_recorder(recorder_id, "label.pkl")
    pred = _ensure_score_df(pd.read_pickle(pred_path)).sort_index()
    label = pd.read_pickle(label_path).sort_index()
    common_index = pred.index.intersection(label.index)
    return pred.loc[common_index].sort_index(), label.loc[common_index].sort_index()


def _align_signals(
    seq_pred: pd.DataFrame,
    seq_label: pd.DataFrame,
    tab_pred: pd.DataFrame,
    tab_label: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common_index = seq_pred.index.intersection(tab_pred.index).intersection(seq_label.index).intersection(tab_label.index)
    common_index = common_index.sort_values()
    return (
        seq_pred.loc[common_index].sort_index(),
        tab_pred.loc[common_index].sort_index(),
        seq_label.loc[common_index].sort_index(),
    )


def _write_fused_cache(cache_name: str, pred: pd.DataFrame, label: pd.DataFrame) -> Path:
    FUSION_CACHE_DIR.mkdir(exist_ok=True)
    cache_path = FUSION_CACHE_DIR / f"{cache_name}.pkl"
    pd.to_pickle({"pred": pred, "label": label}, cache_path)
    return cache_path
