from __future__ import annotations

import copy
from pathlib import Path


DEFAULT_MASK_START = "2015-06-15"
DEFAULT_MASK_END = "2016-01-31"
PROCESSOR_CLASS = "DropLabelByDateRange"
PROCESSOR_MODULE_PATH = "qlib.contrib.data.processor"


def build_drop_label_processor(start_date: str, end_date: str) -> dict:
    return {
        "class": PROCESSOR_CLASS,
        "module_path": PROCESSOR_MODULE_PATH,
        "kwargs": {
            "start_date": start_date,
            "end_date": end_date,
            "fields_group": "label",
        },
    }


def apply_train_label_mask(
    handler_kwargs_extra: dict | None,
    start_date: str | None,
    end_date: str | None,
    base_learn_processors: list | None = None,
) -> dict | None:
    if not start_date and not end_date:
        return copy.deepcopy(handler_kwargs_extra)
    if not start_date or not end_date:
        raise ValueError("train label mask start/end must be provided together")

    merged = copy.deepcopy(handler_kwargs_extra) if handler_kwargs_extra is not None else {}
    if "learn_processors" in merged:
        existing = list(merged["learn_processors"])
    elif base_learn_processors is not None:
        existing = copy.deepcopy(list(base_learn_processors))
    else:
        existing = []
    existing.insert(0, build_drop_label_processor(start_date, end_date))
    merged["learn_processors"] = existing
    return merged


def suffix_path(path: Path, result_suffix: str = "") -> Path:
    if not result_suffix:
        return path
    return path.with_name(f"{path.stem}_{result_suffix}{path.suffix}")


def compose_exp_suffix(base_suffix: str, result_suffix: str = "") -> str:
    if not result_suffix:
        return base_suffix
    if not base_suffix:
        return result_suffix
    return f"{base_suffix}_{result_suffix}"