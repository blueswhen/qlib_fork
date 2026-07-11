#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


CNINFO_COLUMNS = ["代码", "简称", "公告标题", "公告时间", "公告链接"]
MARKET_CLOSE_TIME = "15:00:00"

POSITIVE_KEYWORDS = [
    "中标",
    "订单",
    "合同",
    "合作",
    "回购",
    "增持",
    "股权激励",
    "预增",
    "扭亏",
    "量产",
    "认证",
    "获批",
    "扩产",
    "投产",
]

NEGATIVE_KEYWORDS = [
    "减持",
    "终止",
    "诉讼",
    "仲裁",
    "问询函",
    "监管函",
    "立案",
    "处罚",
    "风险提示",
    "亏损",
    "预亏",
    "下修",
    "延期",
    "质押",
]

CATEGORY_RULES = {
    "ann_is_earnings": ["年报", "半年报", "季报", "业绩预告", "业绩快报", "预增", "预亏", "扭亏"],
    "ann_is_contract_order": ["中标", "订单", "合同", "框架协议"],
    "ann_is_buyback": ["回购"],
    "ann_is_insider_change": ["增持", "减持", "持股变动"],
    "ann_is_regulatory": ["问询函", "监管函", "立案", "处罚", "关注函", "警示函"],
    "ann_is_capital_action": ["定增", "募投", "重组", "收购", "并购", "股权激励"],
    "ann_is_risk": ["风险提示", "诉讼", "仲裁", "质押", "延期", "预亏", "亏损", "问询函", "监管函", "立案", "处罚"],
}

QLIB_FEATURES = [
    "ann_cnt_1d",
    "ann_positive_cnt_1d",
    "ann_negative_cnt_1d",
    "ann_earnings_cnt_1d",
    "ann_contract_order_cnt_1d",
    "ann_buyback_cnt_1d",
    "ann_insider_change_cnt_1d",
    "ann_regulatory_cnt_1d",
    "ann_capital_action_cnt_1d",
    "ann_risk_cnt_1d",
    "ann_sentiment_score_1d",
    "ann_cnt_5d",
    "ann_positive_cnt_5d",
    "ann_negative_cnt_5d",
    "ann_earnings_cnt_5d",
    "ann_contract_order_cnt_5d",
    "ann_buyback_cnt_5d",
    "ann_insider_change_cnt_5d",
    "ann_regulatory_cnt_5d",
    "ann_capital_action_cnt_5d",
    "ann_risk_cnt_5d",
    "ann_sentiment_score_5d",
    "ann_cnt_20d",
    "ann_positive_cnt_20d",
    "ann_negative_cnt_20d",
    "ann_earnings_cnt_20d",
    "ann_contract_order_cnt_20d",
    "ann_buyback_cnt_20d",
    "ann_insider_change_cnt_20d",
    "ann_regulatory_cnt_20d",
    "ann_capital_action_cnt_20d",
    "ann_risk_cnt_20d",
    "ann_sentiment_score_20d",
    "ann_cnt_60d",
    "ann_positive_cnt_60d",
    "ann_negative_cnt_60d",
    "ann_earnings_cnt_60d",
    "ann_contract_order_cnt_60d",
    "ann_buyback_cnt_60d",
    "ann_insider_change_cnt_60d",
    "ann_regulatory_cnt_60d",
    "ann_capital_action_cnt_60d",
    "ann_risk_cnt_60d",
    "ann_sentiment_score_60d",
    "days_since_last_announcement",
    "days_since_last_earnings_announcement",
]

DECAY_BASE_FEATURES = [
    "ann_cnt_1d",
    "ann_positive_cnt_1d",
    "ann_negative_cnt_1d",
    "ann_earnings_cnt_1d",
    "ann_contract_order_cnt_1d",
    "ann_buyback_cnt_1d",
    "ann_insider_change_cnt_1d",
    "ann_regulatory_cnt_1d",
    "ann_capital_action_cnt_1d",
    "ann_risk_cnt_1d",
    "ann_sentiment_score_1d",
]


@dataclass
class Context:
    data_root: Path
    universe_file: Path
    qlib_calendar_file: Path
    start_date: str
    end_date: str
    sleep_sec: float
    retries: int
    force: bool

    @property
    def raw_root(self) -> Path:
        return self.data_root / "raw"

    @property
    def processed_root(self) -> Path:
        return self.data_root / "processed"

    @property
    def state_root(self) -> Path:
        return self.data_root / "state"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root()
    parser = argparse.ArgumentParser(
        description=(
            "Download CNInfo announcement data and build the local announcement "
            "training parquet files under training_data/tushare."
        )
    )
    parser.add_argument("--data-root", default=str(root / "training_data" / "tushare"))
    parser.add_argument("--universe-file", default=str(root / "training_data" / "cn_data_latest" / "instruments" / "csi300.txt"))
    parser.add_argument("--qlib-calendar-file", default=str(root / "training_data" / "cn_data_latest" / "calendars" / "day.txt"))
    parser.add_argument("--start-date", default="20120101")
    parser.add_argument("--end-date", default="20260426")
    parser.add_argument("--sleep-sec", type=float, default=0.12)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="Re-download raw CNInfo partitions that already exist.")
    parser.add_argument("--skip-download", action="store_true", help="Build processed files from existing raw/anns_d partitions.")
    return parser.parse_args()


def ensure_dirs(ctx: Context) -> None:
    for sub in ["raw", "processed/marts", "processed/training", "state", "logs"]:
        (ctx.data_root / sub).mkdir(parents=True, exist_ok=True)


def write_parquet_atomic(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path, index=index)
    tmp_path.replace(path)


def call_with_retry(
    func: Callable,
    retries: int,
    sleep_sec: float,
    non_retryable: Callable[[Exception], bool] | None = None,
    **kwargs,
) -> pd.DataFrame:
    last_err = None
    func_name = getattr(func, "__name__", "api_call")
    for attempt in range(1, retries + 1):
        try:
            return func(**kwargs)
        except Exception as err:  # pragma: no cover - network/API failure path
            if non_retryable is not None and non_retryable(err):
                raise
            last_err = err
            backoff = sleep_sec * max(2, attempt)
            print(f"retry {attempt}/{retries} for {func_name} due to: {err}")
            time.sleep(backoff)
    raise RuntimeError(f"failed after {retries} retries: {func_name}") from last_err


def is_cninfo_empty_result_error(err: Exception) -> bool:
    if not isinstance(err, KeyError):
        return False
    text = str(err)
    return all(token in text for token in ["代码", "简称", "公告标题", "公告时间"])


def fetch_cninfo_announcements(symbol: str, start_date: str, end_date: str, retries: int, sleep_sec: float) -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is required: pip install akshare") from exc

    try:
        return call_with_retry(
            ak.stock_zh_a_disclosure_report_cninfo,
            retries,
            sleep_sec,
            non_retryable=is_cninfo_empty_result_error,
            symbol=symbol,
            market="沪深京",
            keyword="",
            category="",
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as err:
        root = err.__cause__ if isinstance(err, RuntimeError) else err
        if root is not None and is_cninfo_empty_result_error(root):
            print(f"cninfo empty result for symbol={symbol} window={start_date}-{end_date}; saving empty partition")
            return pd.DataFrame(columns=CNINFO_COLUMNS)
        raise


def qlib_to_tushare(code: str) -> str:
    if code.startswith("SH"):
        return f"{code[2:]}.SH"
    if code.startswith("SZ"):
        return f"{code[2:]}.SZ"
    if code.startswith("BJ"):
        return f"{code[2:]}.BJ"
    raise ValueError(f"unsupported qlib code: {code}")


def tushare_to_qlib(ts_code: str) -> str:
    symbol, market = ts_code.split(".")
    return f"{market}{symbol}"


def load_universe_bounds(universe_file: Path, start_date: str, end_date: str) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    df = pd.read_csv(universe_file, sep="\t", header=None, names=["instrument", "start_date", "end_date"])
    df["ts_code"] = df["instrument"].astype(str).map(qlib_to_tushare)
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    request_start = pd.Timestamp(start_date)
    request_end = pd.Timestamp(end_date)
    out: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for ts_code, group in df.groupby("ts_code", sort=True):
        clipped_start = max(group["start_date"].min(), request_start)
        clipped_end = min(group["end_date"].max(), request_end)
        if clipped_start <= clipped_end:
            out[ts_code] = (clipped_start.normalize(), clipped_end.normalize())
    return out


def year_windows(start_date: str, end_date: str) -> list[tuple[str, str, int]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    windows = []
    for period in pd.period_range(start=start, end=end, freq="Y"):
        year_start = max(start, period.start_time.normalize())
        year_end = min(end, period.end_time.normalize())
        windows.append((year_start.strftime("%Y%m%d"), year_end.strftime("%Y%m%d"), period.year))
    return windows


def normalize_cninfo_announcements(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["ann_date", "ts_code", "name", "title", "url", "rec_time", "source", "ingest_time"])

    out = df.copy().rename(
        columns={
            "代码": "ts_code",
            "简称": "name",
            "公告标题": "title",
            "公告时间": "ann_date",
            "公告链接": "url",
        }
    )
    for col in ["ts_code", "name", "title", "ann_date", "url"]:
        if col not in out.columns:
            out[col] = pd.NA
    out["ann_date"] = pd.to_datetime(out["ann_date"], errors="coerce").dt.strftime("%Y%m%d")
    out["rec_time"] = pd.NA
    out["source"] = "cninfo_akshare"
    out["ingest_time"] = pd.Timestamp.utcnow().isoformat()
    return out[["ann_date", "ts_code", "name", "title", "url", "rec_time", "source", "ingest_time"]].copy()


def download_announcements(ctx: Context, ts_code_bounds: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> None:
    root = ctx.raw_root / "anns_d"
    saved = 0
    for ts_code, (active_start, active_end) in ts_code_bounds.items():
        symbol = ts_code.split(".")[0]
        for window_start, window_end, year in year_windows(active_start.strftime("%Y%m%d"), active_end.strftime("%Y%m%d")):
            path = root / f"ts_code={ts_code}" / f"year={year}" / "data.parquet"
            if path.exists() and not ctx.force:
                continue
            df = fetch_cninfo_announcements(symbol, window_start, window_end, ctx.retries, ctx.sleep_sec)
            normalized = normalize_cninfo_announcements(df)
            if not normalized.empty:
                normalized["ts_code"] = ts_code
            write_parquet_atomic(normalized, path, index=False)
            saved += 1
            if saved % 100 == 0:
                print(f"cninfo announcements saved partitions={saved} latest={ts_code} year={year}")
            time.sleep(ctx.sleep_sec)
    print(f"cninfo announcements completed ts_codes={len(ts_code_bounds)} new_saved={saved}")


def load_partitions_under(root: Path) -> pd.DataFrame:
    files = sorted(root.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(file) for file in files]
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def make_ann_id(row: pd.Series) -> str:
    raw = "|".join(
        [
            str(row.get("ts_code") or ""),
            str(row.get("ann_date") or ""),
            str(row.get("title") or ""),
            str(row.get("url") or ""),
            str(row.get("rec_time") or ""),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class TradingCalendar:
    def __init__(self, trade_dates: Iterable[pd.Timestamp]):
        self.trade_dates = sorted(dict.fromkeys(pd.Timestamp(d).normalize() for d in trade_dates))
        self.trade_set = set(self.trade_dates)

    def next_trade_day(self, dt: pd.Timestamp, include_today: bool) -> pd.Timestamp | pd.NaT:
        key = pd.Timestamp(dt).normalize()
        idx = bisect_left(self.trade_dates, key) if include_today else bisect_right(self.trade_dates, key)
        if idx >= len(self.trade_dates):
            return pd.NaT
        return self.trade_dates[idx]

    def effective_date(self, ann_date: pd.Timestamp, rec_time: pd.Timestamp | pd.NaT) -> pd.Timestamp | pd.NaT:
        close_time = pd.Timestamp(MARKET_CLOSE_TIME).time()
        ann_date = pd.Timestamp(ann_date).normalize()
        if pd.notna(rec_time):
            rec_time = pd.Timestamp(rec_time)
            rec_day = rec_time.normalize()
            if rec_day in self.trade_set and rec_time.time() < close_time:
                return rec_day
            return self.next_trade_day(rec_day, include_today=rec_day not in self.trade_set)
        return self.next_trade_day(ann_date, include_today=ann_date not in self.trade_set)


def load_trade_dates(ctx: Context) -> pd.DatetimeIndex:
    if ctx.qlib_calendar_file.exists():
        dates = pd.read_csv(ctx.qlib_calendar_file, header=None, names=["cal_date"])["cal_date"]
        trade_dates = pd.to_datetime(dates)
        return pd.DatetimeIndex(trade_dates[(trade_dates >= pd.Timestamp(ctx.start_date)) & (trade_dates <= pd.Timestamp(ctx.end_date))])

    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "missing qlib calendar file and TUSHARE_TOKEN. "
            "Set TUSHARE_TOKEN only on the server running this script; do not commit it."
        )
    try:
        import tushare as ts
    except ImportError as exc:
        raise RuntimeError("tushare is required for calendar fallback: pip install tushare") from exc

    pro = ts.pro_api(token)
    cal = pro.trade_cal(exchange="SSE", start_date=ctx.start_date, end_date=ctx.end_date)
    cal["cal_date"] = pd.to_datetime(cal["cal_date"])
    raw_path = ctx.raw_root / "trade_cal" / "SSE.parquet"
    write_parquet_atomic(cal, raw_path, index=False)
    return pd.DatetimeIndex(cal.loc[cal["is_open"] == 1, "cal_date"].sort_values())


def standardize_announcements(df: pd.DataFrame, calendar: TradingCalendar) -> pd.DataFrame:
    if df.empty:
        raise RuntimeError("raw announcements are missing")

    out = df.copy()
    out["ann_date"] = pd.to_datetime(out["ann_date"], format="%Y%m%d", errors="coerce")
    out["rec_time"] = pd.to_datetime(out["rec_time"], errors="coerce")
    for col in ["title", "name", "url", "source"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("").astype(str)
    out = out[out["ts_code"].notna() & out["ann_date"].notna()].copy()
    out["ann_id"] = out.apply(make_ann_id, axis=1)
    out = out.drop_duplicates("ann_id").reset_index(drop=True)
    out["effective_date"] = [calendar.effective_date(ann_date, rec_time) for ann_date, rec_time in zip(out["ann_date"], out["rec_time"])]
    out = out[out["effective_date"].notna()].copy()

    title_text = out["title"].astype(str)
    out["ann_positive_hit"] = title_text.apply(lambda x: int(any(word in x for word in POSITIVE_KEYWORDS)))
    out["ann_negative_hit"] = title_text.apply(lambda x: int(any(word in x for word in NEGATIVE_KEYWORDS)))
    out["ann_sentiment_score"] = out["ann_positive_hit"] - out["ann_negative_hit"]
    for feature_name, keywords in CATEGORY_RULES.items():
        out[feature_name] = title_text.apply(lambda x: int(any(word in x for word in keywords)))

    cols = [
        "ann_id",
        "ann_date",
        "effective_date",
        "ts_code",
        "name",
        "title",
        "url",
        "rec_time",
        "source",
        "ann_positive_hit",
        "ann_negative_hit",
        "ann_sentiment_score",
        *CATEGORY_RULES.keys(),
    ]
    return out[cols].sort_values(["ts_code", "effective_date", "ann_date", "title"]).reset_index(drop=True)


def build_universe_rows(universe_file: Path, trade_dates: pd.DatetimeIndex) -> pd.DataFrame:
    rows = []
    df = pd.read_csv(universe_file, sep="\t", header=None, names=["instrument", "start_date", "end_date"])
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"])
    for row in df.itertuples(index=False):
        mask = (trade_dates >= row.start_date) & (trade_dates <= row.end_date)
        active_dates = trade_dates[mask]
        if len(active_dates) == 0:
            continue
        ts_code = qlib_to_tushare(row.instrument)
        rows.extend((ts_code, d) for d in active_dates)
    return pd.DataFrame(rows, columns=["ts_code", "trade_date"]).drop_duplicates().sort_values(["ts_code", "trade_date"]).reset_index(drop=True)


def build_daily_event_aggregates(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.groupby(["ts_code", "effective_date"], as_index=False)
        .agg(
            ann_cnt_1d=("ann_id", "count"),
            ann_positive_cnt_1d=("ann_positive_hit", "sum"),
            ann_negative_cnt_1d=("ann_negative_hit", "sum"),
            ann_earnings_cnt_1d=("ann_is_earnings", "sum"),
            ann_contract_order_cnt_1d=("ann_is_contract_order", "sum"),
            ann_buyback_cnt_1d=("ann_is_buyback", "sum"),
            ann_insider_change_cnt_1d=("ann_is_insider_change", "sum"),
            ann_regulatory_cnt_1d=("ann_is_regulatory", "sum"),
            ann_capital_action_cnt_1d=("ann_is_capital_action", "sum"),
            ann_risk_cnt_1d=("ann_is_risk", "sum"),
            ann_sentiment_score_1d=("ann_sentiment_score", "sum"),
        )
        .rename(columns={"effective_date": "trade_date"})
    )


def days_since_last_flag(flag: np.ndarray) -> np.ndarray:
    out = np.full(len(flag), np.nan, dtype=float)
    last_idx = None
    for idx, value in enumerate(flag):
        if value > 0:
            last_idx = idx
            out[idx] = 0.0
        elif last_idx is not None:
            out[idx] = float(idx - last_idx)
    return out


def add_rolling_features(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    base_cols = DECAY_BASE_FEATURES
    for col in base_cols:
        panel[col] = panel[col].fillna(0.0)

    result = []
    for _, group in panel.groupby("ts_code", sort=False):
        group = group.copy()
        for window in [5, 20, 60]:
            for col in base_cols:
                group[col.replace("_1d", f"_{window}d")] = group[col].rolling(window, min_periods=1).sum()
        group["days_since_last_announcement"] = days_since_last_flag(group["ann_cnt_1d"].to_numpy())
        group["days_since_last_earnings_announcement"] = days_since_last_flag(group["ann_earnings_cnt_1d"].to_numpy())
        result.append(group)
    return pd.concat(result, ignore_index=True)


def build_panel(ctx: Context, trade_dates: pd.DatetimeIndex) -> tuple[Path, Path]:
    raw = load_partitions_under(ctx.raw_root / "anns_d")
    events = standardize_announcements(raw, TradingCalendar(trade_dates))
    universe_rows = build_universe_rows(ctx.universe_file, trade_dates)
    events = events[events["ts_code"].isin(set(universe_rows["ts_code"].unique()))].copy()

    marts_path = ctx.processed_root / "marts" / "announcements.parquet"
    write_parquet_atomic(events, marts_path, index=False)

    daily_events = build_daily_event_aggregates(events)
    panel = universe_rows.merge(daily_events, on=["ts_code", "trade_date"], how="left")
    panel = add_rolling_features(panel)
    panel_path = ctx.processed_root / "training" / "csi300_daily_announcement_panel.parquet"
    write_parquet_atomic(panel, panel_path, index=False)
    return marts_path, panel_path


def export_qlib_features(panel_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(panel_path)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    feature_cols = [col for col in QLIB_FEATURES if col in df.columns]
    out = df[["ts_code", "trade_date", *feature_cols]].copy()
    out["instrument"] = out["ts_code"].map(tushare_to_qlib)
    out = out.drop(columns=["ts_code"]).rename(columns={"trade_date": "datetime"})
    out = out.set_index(["datetime", "instrument"]).sort_index()
    out.columns = pd.MultiIndex.from_product([["feature"], out.columns])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path)
    return out


def write_rankpct(src: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    ranked = src.groupby(level="datetime", group_keys=False).apply(lambda group: group.rank(pct=True, method="average"))
    ranked = (ranked - 0.5).fillna(0.0).astype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_parquet(output_path)
    return ranked


def write_decay_features(features: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    simple = features.copy()
    simple.columns = simple.columns.get_level_values(-1)
    simple = simple[DECAY_BASE_FEATURES].fillna(0.0)

    parts = []
    for _, group in simple.groupby(level="instrument", sort=False):
        group = group.sort_index(level="datetime")
        cols = {}
        for col in DECAY_BASE_FEATURES:
            stem = col.removesuffix("_1d")
            series = group[col]
            for halflife in [3, 10, 30]:
                cols[f"{stem}_decay_h{halflife}"] = series.ewm(halflife=halflife, adjust=False).mean()
        cols["days_since_last_announcement"] = features.loc[group.index, ("feature", "days_since_last_announcement")]
        cols["days_since_last_earnings_announcement"] = features.loc[group.index, ("feature", "days_since_last_earnings_announcement")]
        parts.append(pd.DataFrame(cols, index=group.index))

    out = pd.concat(parts).sort_index().astype(np.float32)
    out.columns = pd.MultiIndex.from_product([["feature"], out.columns])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path)
    return out


def write_manifest(ctx: Context, ts_code_count: int, paths: list[Path]) -> None:
    manifest = {
        "provider": "cninfo_akshare",
        "token": "not_required_for_cninfo_akshare; optional TUSHARE_TOKEN environment variable is used only for calendar fallback",
        "start_date": ctx.start_date,
        "end_date": ctx.end_date,
        "universe_file": str(ctx.universe_file),
        "qlib_calendar_file": str(ctx.qlib_calendar_file),
        "ts_code_count": ts_code_count,
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "outputs": [str(path) for path in paths],
    }
    ctx.state_root.mkdir(parents=True, exist_ok=True)
    (ctx.state_root / "announcements_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ctx = Context(
        data_root=Path(args.data_root).expanduser().resolve(),
        universe_file=Path(args.universe_file).expanduser().resolve(),
        qlib_calendar_file=Path(args.qlib_calendar_file).expanduser().resolve(),
        start_date=args.start_date,
        end_date=args.end_date,
        sleep_sec=args.sleep_sec,
        retries=args.retries,
        force=args.force,
    )
    ensure_dirs(ctx)

    ts_code_bounds = load_universe_bounds(ctx.universe_file, ctx.start_date, ctx.end_date)
    if not args.skip_download:
        download_announcements(ctx, ts_code_bounds)

    trade_dates = load_trade_dates(ctx)
    marts_path, panel_path = build_panel(ctx, trade_dates)

    training_root = ctx.processed_root / "training"
    features_path = training_root / "csi300_daily_announcement_features_qlib.parquet"
    rankpct_path = training_root / "csi300_daily_announcement_features_rankpct_qlib.parquet"
    decay_path = training_root / "csi300_daily_announcement_features_decay_qlib.parquet"
    decay_rankpct_path = training_root / "csi300_daily_announcement_features_decay_rankpct_qlib.parquet"

    features = export_qlib_features(panel_path, features_path)
    write_rankpct(features, rankpct_path)
    decay = write_decay_features(features, decay_path)
    write_rankpct(decay, decay_rankpct_path)

    outputs = [marts_path, panel_path, features_path, rankpct_path, decay_path, decay_rankpct_path, ctx.state_root / "announcements_manifest.json"]
    write_manifest(ctx, len(ts_code_bounds), outputs)
    for path in outputs:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
