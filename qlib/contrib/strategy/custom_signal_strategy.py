from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import pandas as pd

from qlib.data import D
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.backtest.position import Position


class TrendAwareTopkDropoutStrategy(TopkDropoutStrategy):
    """
    A Top-k dropout strategy with a simple market regime filter built only from daily benchmark prices.

    Regime rules:
    - normal: use full configured risk_degree
    - caution: use reduced risk_degree
    - defensive: no new buying capital
    """

    def __init__(
        self,
        *,
        benchmark: str = "SH000300",
        benchmark_freq: str = "day",
        trend_short_window: int = 10,
        trend_long_window: int = 30,
        vol_window: int = 20,
        vol_thresh: float = 0.022,
        caution_risk_degree: float = 0.35,
        defensive_risk_degree: float = 0.0,
        defensive_drop_all: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.benchmark = benchmark
        self.benchmark_freq = benchmark_freq
        self.trend_short_window = trend_short_window
        self.trend_long_window = trend_long_window
        self.vol_window = vol_window
        self.vol_thresh = vol_thresh
        self.caution_risk_degree = caution_risk_degree
        self.defensive_risk_degree = defensive_risk_degree
        self.defensive_drop_all = defensive_drop_all
        self._regime_signal: Optional[pd.DataFrame] = None

    def reset_level_infra(self, level_infra):
        super().reset_level_infra(level_infra)
        trade_len = self.trade_calendar.get_trade_len()
        signal_start_time, _ = self.trade_calendar.get_step_time(trade_step=0, shift=1)
        _, signal_end_time = self.trade_calendar.get_step_time(trade_step=trade_len - 1, shift=1)
        fields = [
            f"EMA($close, {self.trend_short_window})/EMA($close, {self.trend_long_window})-1",
            f"Std(Log($close/Ref($close, 1)), {self.vol_window})",
        ]
        signal_df = D.features(
            [self.benchmark],
            fields=fields,
            start_time=signal_start_time,
            end_time=signal_end_time,
            freq=self.benchmark_freq,
        )
        signal_df.columns = ["trend", "vol"]
        self._regime_signal = signal_df.droplevel(level="instrument")

    def _get_regime(self, trade_step: int) -> str:
        if self._regime_signal is None or self._regime_signal.empty:
            return "normal"
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        signal_slice = self._regime_signal.loc[pred_start_time:pred_end_time]
        if signal_slice.empty:
            return "normal"
        latest = signal_slice.iloc[-1]
        trend = latest["trend"]
        vol = latest["vol"]
        if pd.isna(trend) or pd.isna(vol):
            return "normal"
        if trend <= 0 and vol >= self.vol_thresh:
            return "defensive"
        if trend <= 0 or vol >= self.vol_thresh:
            return "caution"
        return "normal"

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        regime = self._get_regime(trade_step)
        original_risk = self.risk_degree
        original_n_drop = self.n_drop
        try:
            if regime == "caution":
                self.risk_degree = min(self.risk_degree, self.caution_risk_degree)
            elif regime == "defensive":
                self.risk_degree = self.defensive_risk_degree
                if self.defensive_drop_all:
                    self.n_drop = self.topk
            return super().generate_trade_decision(execute_result=execute_result)
        finally:
            self.risk_degree = original_risk
            self.n_drop = original_n_drop


class PracticalTopkDropoutStrategy(TopkDropoutStrategy):
    """
    A more execution-aware Top-k dropout strategy for small A-share accounts.

    Enhancements over the default strategy:
    - Skip candidates that cannot fit at least one trade unit within the target per-slot budget.
    - Only replace an existing position when the new score is better by a configurable margin.
    """

    def __init__(
        self,
        *,
        score_margin: float = 0.0,
        require_slot_affordable: bool = True,
        slot_budget_ratio: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.score_margin = score_margin
        self.require_slot_affordable = require_slot_affordable
        self.slot_budget_ratio = slot_budget_ratio

    def _slot_budget(self, current: Position) -> float:
        total_value = current.calculate_value()
        if self.topk <= 0:
            return 0.0
        return total_value * self.risk_degree / self.topk

    def _can_afford_target_slot(self, stock_id: str, slot_budget: float, trade_start_time, trade_end_time) -> bool:
        if not self.require_slot_affordable:
            return True
        if slot_budget <= 0:
            return False
        buy_price = self.trade_exchange.get_deal_price(
            stock_id=stock_id,
            start_time=trade_start_time,
            end_time=trade_end_time,
            direction=OrderDir.BUY,
        )
        factor = self.trade_exchange.get_factor(stock_id=stock_id, start_time=trade_start_time, end_time=trade_end_time)
        theoretical_amount = slot_budget * self.slot_budget_ratio / buy_price
        rounded_amount = self.trade_exchange.round_amount_by_trade_unit(theoretical_amount, factor)
        return rounded_amount > 0

    def _get_buy_allocations(self, buy, cash: float, trade_start_time, trade_end_time) -> dict[str, float]:
        if len(buy) == 0:
            return {}
        per_stock_value = cash * self.risk_degree / len(buy)
        return {code: per_stock_value for code in buy}

    def _load_signal_score(self, start_time, end_time) -> Optional[pd.Series]:
        pred_score = self.signal.get_signal(start_time=start_time, end_time=end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return None
        return pred_score.astype(float)

    def _get_pred_score(self, trade_step: int, pred_start_time, pred_end_time) -> Optional[pd.Series]:
        return self._load_signal_score(pred_start_time, pred_end_time)

    def _adjust_pred_score(
        self,
        pred_score: pd.Series,
        current_stock_list: list[str],
        trade_step: int,
        trade_start_time,
        trade_end_time,
    ) -> pd.Series:
        return pred_score

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self._get_pred_score(trade_step, pred_start_time, pred_end_time)
        if pred_score is None:
            return TradeDecisionWO([], self)

        if self.only_tradable:

            def get_first_n(li, n, reverse=False):
                cur_n = 0
                res = []
                for si in reversed(li) if reverse else li:
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    ):
                        res.append(si)
                        cur_n += 1
                        if cur_n >= n:
                            break
                return res[::-1] if reverse else res

            def get_last_n(li, n):
                return get_first_n(li, n, reverse=True)

        else:

            def get_first_n(li, n):
                return list(li)[:n]

            def get_last_n(li, n):
                return list(li)[-n:]

        current_temp: Position = copy.deepcopy(self.trade_position)
        sell_order_list = []
        buy_order_list = []
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()
        pred_score = self._adjust_pred_score(
            pred_score=pred_score,
            current_stock_list=current_stock_list,
            trade_step=trade_step,
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
        )
        current_scores = pred_score.reindex(current_stock_list)
        last = current_scores.sort_values(ascending=False).index

        if self.method_buy == "top":
            raw_today = get_first_n(
                pred_score[~pred_score.index.isin(last)].sort_values(ascending=False).index,
                self.n_drop + self.topk - len(last),
            )
        elif self.method_buy == "random":
            topk_candi = get_first_n(pred_score.sort_values(ascending=False).index, self.topk)
            candi = [x for x in topk_candi if x not in last]
            n = self.n_drop + self.topk - len(last)
            try:
                raw_today = np.random.choice(candi, n, replace=False)
            except ValueError:
                raw_today = candi
        else:
            raise NotImplementedError(f"This type of input is not supported")

        slot_budget = self._slot_budget(current_temp)
        today = [
            code
            for code in raw_today
            if self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
            )
            and self._can_afford_target_slot(code, slot_budget, trade_start_time, trade_end_time)
        ]

        comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index

        if self.method_sell == "bottom":
            sell_candidates = list(last[last.isin(get_last_n(comb, self.n_drop))])
        elif self.method_sell == "random":
            candi = [
                code
                for code in last
                if self.trade_exchange.is_stock_tradable(
                    stock_id=code,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
                )
            ]
            try:
                sell_candidates = list(np.random.choice(candi, self.n_drop, replace=False) if len(last) else [])
            except ValueError:
                sell_candidates = list(candi)
        else:
            raise NotImplementedError(f"This type of input is not supported")

        actual_sellable = []
        time_per_step = self.trade_calendar.get_freq()
        for code in sell_candidates:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                continue
            if current_temp.get_stock_count(code, bar=time_per_step) < self.hold_thresh:
                continue
            actual_sellable.append(code)

        today = list(today[: len(actual_sellable) + self.topk - len(last)])
        underfilled_slots = max(0, self.topk - len(last))
        approved_buys = []
        approved_sells = []
        sell_queue = sorted(actual_sellable, key=lambda code: pred_score.get(code, -np.inf))
        for code in today:
            if underfilled_slots > 0:
                approved_buys.append(code)
                underfilled_slots -= 1
                continue
            if not sell_queue:
                break
            weakest = sell_queue[0]
            if pred_score.get(code, -np.inf) > pred_score.get(weakest, -np.inf) + self.score_margin:
                approved_buys.append(code)
                approved_sells.append(weakest)
                sell_queue.pop(0)

        sell = approved_sells
        buy = approved_buys

        for code in current_stock_list:
            if code in sell:
                sell_amount = current_temp.get_stock_amount(code=code)
                sell_order = Order(
                    stock_id=code,
                    amount=sell_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.SELL,
                )
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                        sell_order, position=current_temp
                    )
                    cash += trade_val - trade_cost

        allocations = self._get_buy_allocations(
            buy=buy,
            cash=cash,
            trade_start_time=trade_start_time,
            trade_end_time=trade_end_time,
        )
        for code in buy:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
            ):
                continue
            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
            )
            buy_value = allocations.get(code, 0.0)
            if buy_value <= 0:
                continue
            buy_amount = buy_value / buy_price
            factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            if buy_amount <= 0:
                continue
            buy_order = Order(
                stock_id=code,
                amount=buy_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY,
            )
            buy_order_list.append(buy_order)
        return TradeDecisionWO(sell_order_list + buy_order_list, self)


class InverseVolPracticalTopkDropoutStrategy(PracticalTopkDropoutStrategy):
    """
    Size new buys by inverse realized volatility instead of equal slot value.
    """

    def __init__(
        self,
        *,
        universe: str = "csi300",
        price_field: str = "$close",
        vol_window: int = 20,
        vol_power: float = 1.0,
        min_vol: float = 1e-4,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.universe = universe
        self.price_field = price_field
        self.vol_window = vol_window
        self.vol_power = vol_power
        self.min_vol = min_vol
        self._volatility: Optional[pd.DataFrame] = None

    def reset_level_infra(self, level_infra):
        super().reset_level_infra(level_infra)
        trade_len = self.trade_calendar.get_trade_len()
        signal_start_time, _ = self.trade_calendar.get_step_time(trade_step=0, shift=1)
        _, signal_end_time = self.trade_calendar.get_step_time(trade_step=trade_len - 1, shift=1)
        field = f"Std(Log({self.price_field}/Ref({self.price_field}, 1)), {self.vol_window})"
        instruments = D.instruments(self.universe) if isinstance(self.universe, str) else self.universe
        vol_df = D.features(
            instruments,
            fields=[field],
            start_time=signal_start_time,
            end_time=signal_end_time,
            freq=self.trade_calendar.get_freq(),
        )
        vol_df.columns = ["volatility"]
        self._volatility = vol_df["volatility"].unstack(level="instrument").sort_index()

    def _get_latest_volatility(self, trade_start_time, trade_end_time) -> Optional[pd.Series]:
        if self._volatility is None or self._volatility.empty:
            return None
        vol_slice = self._volatility.loc[trade_start_time:trade_end_time]
        if vol_slice.empty:
            return None
        latest = vol_slice.iloc[-1].replace([np.inf, -np.inf], np.nan)
        if latest.isna().all():
            return None
        return latest

    def _get_buy_allocations(self, buy, cash: float, trade_start_time, trade_end_time) -> dict[str, float]:
        if len(buy) == 0:
            return {}

        latest_vol = self._get_latest_volatility(trade_start_time, trade_end_time)
        if latest_vol is None:
            return super()._get_buy_allocations(buy, cash, trade_start_time, trade_end_time)

        buy_vol = latest_vol.reindex(buy)
        fill_value = buy_vol.median(skipna=True)
        if pd.isna(fill_value):
            return super()._get_buy_allocations(buy, cash, trade_start_time, trade_end_time)

        buy_vol = buy_vol.fillna(fill_value).clip(lower=self.min_vol)
        inv_vol = np.power(buy_vol.astype(float), -self.vol_power)
        total = float(inv_vol.sum())
        if not np.isfinite(total) or total <= 0:
            return super()._get_buy_allocations(buy, cash, trade_start_time, trade_end_time)

        budget = cash * self.risk_degree
        weights = inv_vol / total
        return {code: float(budget * weights.loc[code]) for code in buy}


class TurnoverAwarePracticalTopkDropoutStrategy(PracticalTopkDropoutStrategy):
    """
    Reduce unnecessary turnover by smoothing scores over time and rewarding incumbent holdings.
    """

    def __init__(
        self,
        *,
        smooth_alpha: float = 0.7,
        hold_bonus: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.smooth_alpha = smooth_alpha
        self.hold_bonus = hold_bonus

    def _get_pred_score(self, trade_step: int, pred_start_time, pred_end_time) -> Optional[pd.Series]:
        current_score = self._load_signal_score(pred_start_time, pred_end_time)
        if current_score is None:
            return None
        if trade_step <= 0 or self.smooth_alpha >= 1.0:
            return current_score

        prev_start_time, prev_end_time = self.trade_calendar.get_step_time(trade_step - 1, shift=1)
        prev_score = self._load_signal_score(prev_start_time, prev_end_time)
        if prev_score is None:
            return current_score

        prev_score = prev_score.reindex(current_score.index)
        blended = current_score * self.smooth_alpha + prev_score.fillna(current_score) * (1.0 - self.smooth_alpha)
        return blended.astype(float)

    def _adjust_pred_score(
        self,
        pred_score: pd.Series,
        current_stock_list: list[str],
        trade_step: int,
        trade_start_time,
        trade_end_time,
    ) -> pd.Series:
        if self.hold_bonus <= 0 or len(current_stock_list) == 0:
            return pred_score
        adjusted = pred_score.copy()
        incumbents = adjusted.index.intersection(pd.Index(current_stock_list))
        if len(incumbents) > 0:
            adjusted.loc[incumbents] = adjusted.loc[incumbents] + self.hold_bonus
        return adjusted


class AdaptiveTopkPracticalTopkDropoutStrategy(PracticalTopkDropoutStrategy):
    """
    Adapt concentration by daily cross-sectional score dispersion.
    """

    def __init__(
        self,
        *,
        low_topk: int = 10,
        mid_topk: int = 5,
        high_topk: int = 3,
        low_spread_thresh: float = 0.95,
        high_spread_thresh: float = 1.20,
        spread_upper_quantile: float = 0.9,
        spread_lower_quantile: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.low_topk = low_topk
        self.mid_topk = mid_topk
        self.high_topk = high_topk
        self.low_spread_thresh = low_spread_thresh
        self.high_spread_thresh = high_spread_thresh
        self.spread_upper_quantile = spread_upper_quantile
        self.spread_lower_quantile = spread_lower_quantile

    def _get_daily_spread(self, pred_score: pd.Series) -> float:
        upper = float(pred_score.quantile(self.spread_upper_quantile))
        lower = float(pred_score.quantile(self.spread_lower_quantile))
        return upper - lower

    def _select_topk(self, pred_score: pd.Series) -> int:
        spread = self._get_daily_spread(pred_score)
        if spread >= self.high_spread_thresh:
            return self.high_topk
        if spread >= self.low_spread_thresh:
            return self.mid_topk
        return self.low_topk

    def generate_trade_decision(self, execute_result=None):
        trade_step = self.trade_calendar.get_trade_step()
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self._get_pred_score(trade_step, pred_start_time, pred_end_time)
        if pred_score is None or pred_score.empty:
            return TradeDecisionWO([], self)

        original_topk = self.topk
        try:
            self.topk = self._select_topk(pred_score)
            return super().generate_trade_decision(execute_result=execute_result)
        finally:
            self.topk = original_topk


class MultiTopkConsensusPracticalTopkDropoutStrategy(PracticalTopkDropoutStrategy):
    """
    Blend multiple top-k sleeves into a consensus bonus and allocation tilt.
    """

    def __init__(
        self,
        *,
        topk_list=(3, 5, 10, 20),
        sleeve_weights=None,
        consensus_strength: float = 0.6,
        allocation_blend: float = 0.5,
        use_rank_decay: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.topk_list = tuple(self._parse_topk_list(topk_list))
        self.sleeve_weights = self._parse_sleeve_weights(sleeve_weights, len(self.topk_list))
        self.consensus_strength = consensus_strength
        self.allocation_blend = allocation_blend
        self.use_rank_decay = use_rank_decay
        self._latest_consensus_score: Optional[pd.Series] = None

    @staticmethod
    def _parse_topk_list(topk_list) -> list[int]:
        if isinstance(topk_list, str):
            values = [int(value.strip()) for value in topk_list.split(",") if value.strip()]
        else:
            values = [int(value) for value in topk_list]
        values = [value for value in values if value > 0]
        if not values:
            raise ValueError("topk_list must contain at least one positive integer")
        return values

    @staticmethod
    def _parse_sleeve_weights(sleeve_weights, expected_len: int) -> np.ndarray:
        if sleeve_weights is None:
            weights = np.ones(expected_len, dtype=float)
        elif isinstance(sleeve_weights, str):
            weights = np.array([float(value.strip()) for value in sleeve_weights.split(",") if value.strip()], dtype=float)
        else:
            weights = np.array([float(value) for value in sleeve_weights], dtype=float)
        if len(weights) != expected_len:
            raise ValueError("sleeve_weights length must match topk_list length")
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("sleeve_weights must sum to a positive value")
        return weights / total

    def _build_consensus_score(self, pred_score: pd.Series) -> pd.Series:
        ordered = pred_score.astype(float).sort_values(ascending=False)
        if ordered.empty:
            return ordered

        ranks = pd.Series(np.arange(1, len(ordered) + 1), index=ordered.index, dtype=float)
        consensus = pd.Series(0.0, index=ordered.index, dtype=float)
        for topk, weight in zip(self.topk_list, self.sleeve_weights):
            effective_topk = min(int(topk), len(ordered))
            if effective_topk <= 0:
                continue
            members = ranks <= effective_topk
            contribution = pd.Series(0.0, index=ordered.index, dtype=float)
            if self.use_rank_decay:
                contribution.loc[members] = (effective_topk - ranks.loc[members] + 1.0) / effective_topk
                total = float(contribution.sum())
                if total > 0:
                    contribution = contribution / total
            else:
                contribution.loc[members] = 1.0 / effective_topk
            consensus = consensus.add(weight * contribution, fill_value=0.0)
        return consensus.reindex(pred_score.index).fillna(0.0).astype(float)

    def _adjust_pred_score(
        self,
        pred_score: pd.Series,
        current_stock_list: list[str],
        trade_step: int,
        trade_start_time,
        trade_end_time,
    ) -> pd.Series:
        consensus = self._build_consensus_score(pred_score)
        self._latest_consensus_score = consensus
        centered_bonus = consensus - float(consensus.mean())
        adjusted = pred_score.astype(float) + centered_bonus * self.consensus_strength
        return adjusted.astype(float)

    def _get_buy_allocations(self, buy, cash: float, trade_start_time, trade_end_time) -> dict[str, float]:
        base_allocations = super()._get_buy_allocations(buy, cash, trade_start_time, trade_end_time)
        if len(buy) == 0 or self._latest_consensus_score is None:
            return base_allocations

        consensus = self._latest_consensus_score.reindex(buy).fillna(0.0).clip(lower=0.0)
        total = float(consensus.sum())
        if not np.isfinite(total) or total <= 0:
            return base_allocations

        consensus_weights = consensus / total
        equal_weight = 1.0 / len(buy)
        budget = cash * self.risk_degree
        allocations = {}
        for code in buy:
            weight = (1.0 - self.allocation_blend) * equal_weight + self.allocation_blend * float(consensus_weights.loc[code])
            allocations[code] = budget * weight
        return allocations
