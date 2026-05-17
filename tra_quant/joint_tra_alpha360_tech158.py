from __future__ import annotations

import copy
import re
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from qlib.contrib.data.custom_handler import TechAlpha158A3DFundamental
from qlib.contrib.data.dataset import (
    MTSDatasetH,
    _create_ts_slices,
    _get_date_parse_fn,
    _maybe_padding,
    _to_tensor,
)
from qlib.contrib.data.handler import _DEFAULT_LEARN_PROCESSORS, check_transform_proc
from qlib.contrib.data.loader import Alpha360DL
from qlib.contrib.model import pytorch_tra as tra_mod
from qlib.contrib.model.pytorch_tra import RNN, TRA, TRAModel, evaluate
from qlib.data.dataset.handler import DataHandlerLP

from rolling_tra_alpha360_latest import (
    DEFAULT_HANDLER_KWARGS_EXTRA,
    WINDOWS,
    _deep_merge,
    _resolve_segment,
)


device = tra_mod.device


class Alpha360TechAlpha158Handler(DataHandlerLP):
    """Join Alpha360 sequence features with TechAlpha158A3D point-in-time features."""

    def __init__(
        self,
        instruments="csi300",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=None,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        if infer_processors is None:
            infer_processors = []

        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        label = kwargs.pop("label", self.get_label_config())
        alpha360_fields, alpha360_names = Alpha360DL.get_feature_config()
        tech_fields, tech_names = TechAlpha158A3DFundamental._technical_feature_config()

        alpha360_loader = {
            "class": "QlibDataLoader",
            "module_path": "qlib.data.dataset.loader",
            "kwargs": {
                "config": {
                    "feature": (alpha360_fields, [f"A360_{name}" for name in alpha360_names]),
                    "label": label,
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        tech_loader = {
            "class": "QlibDataLoader",
            "module_path": "qlib.data.dataset.loader",
            "kwargs": {
                "config": {
                    "feature": (tech_fields, [f"T158_{name}" for name in tech_names]),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        data_loader = {
            "class": "NestedDataLoader",
            "module_path": "qlib.data.dataset.loader",
            "kwargs": {
                "dataloader_l": [alpha360_loader, tech_loader],
                "join": "left",
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    @staticmethod
    def get_label_config():
        return ["Ref($close, -4)/Ref($close, -1) - 1"], ["LABEL0"]


class JointMTSDatasetH(MTSDatasetH):
    """MTS dataset with an extra same-day technical feature branch."""

    def __init__(
        self,
        handler,
        segments,
        seq_len=60,
        horizon=0,
        num_states=0,
        memory_mode="sample",
        batch_size=-1,
        n_samples=None,
        shuffle=True,
        drop_last=False,
        input_size=None,
        tech_prefix="T158_",
        seq_prefix="A360_",
        **kwargs,
    ):
        self.tech_prefix = tech_prefix
        self.seq_prefix = seq_prefix
        if horizon == 0:
            horizon = self._infer_horizon(handler)
        super().__init__(
            handler=handler,
            segments=segments,
            seq_len=seq_len,
            horizon=horizon,
            num_states=num_states,
            memory_mode=memory_mode,
            batch_size=batch_size,
            n_samples=n_samples,
            shuffle=shuffle,
            drop_last=drop_last,
            input_size=input_size,
            **kwargs,
        )

    @staticmethod
    def _infer_horizon(handler) -> int:
        label_expr = None

        if isinstance(handler, dict):
            handler_kwargs = handler.get("kwargs", {})
            label_expr = handler_kwargs.get("label")
        elif isinstance(handler, str):
            label_expr = None
        else:
            if hasattr(handler, "kwargs") and isinstance(handler.kwargs, dict):
                label_expr = handler.kwargs.get("label")
            if label_expr is None and hasattr(handler, "get_label_config"):
                label_expr = handler.get_label_config()[0]

        if label_expr is None:
            label_expr = Alpha360TechAlpha158Handler.get_label_config()[0]

        if isinstance(label_expr, tuple):
            label_expr = list(label_expr)
        if isinstance(label_expr, str):
            label_expr = [label_expr]

        if not isinstance(label_expr, list):
            raise ValueError(f"unsupported label config for horizon inference: {label_expr}")

        inferred = 0
        for expr in label_expr:
            for offset_text in re.findall(r"Ref\([^,]+,\s*(-?\d+)\)", str(expr)):
                offset = int(offset_text)
                if offset < 0:
                    inferred = max(inferred, abs(offset))
        if inferred <= 0:
            raise ValueError(f"failed to infer a positive horizon from label config: {label_expr}")
        return inferred

    def setup_data(self, handler_kwargs: dict = None, **kwargs):
        super(MTSDatasetH, self).setup_data(**kwargs)

        if handler_kwargs is not None:
            self.handler.setup_data(**handler_kwargs)

        try:
            df = self.handler._learn.copy()
        except Exception:
            warnings.warn("cannot access `_learn`, will load raw data")
            df = self.handler._data.copy()
        df.index = df.index.swaplevel()
        df.sort_index(inplace=True)

        feature_df = df["feature"]
        seq_cols = [col for col in feature_df.columns if str(col).startswith(self.seq_prefix)]
        tech_cols = [col for col in feature_df.columns if str(col).startswith(self.tech_prefix)]
        if not seq_cols:
            raise ValueError(f"no sequence feature columns found with prefix {self.seq_prefix}")
        if not tech_cols:
            raise ValueError(f"no tech feature columns found with prefix {self.tech_prefix}")

        self._data = feature_df[seq_cols].values.astype("float32")
        self._tech_data = feature_df[tech_cols].values.astype("float32")
        np.nan_to_num(self._data, copy=False)
        np.nan_to_num(self._tech_data, copy=False)

        self._label = df["label"].squeeze().values.astype("float32")
        self._index = df.index

        if self.input_size is not None and self.input_size != self._data.shape[1]:
            warnings.warn("the data has different shape from input_size and the data will be reshaped")
            assert self._data.shape[1] % self.input_size == 0, "data mismatch, please check `input_size`"

        self._batch_slices = _create_ts_slices(self._index, self.seq_len)

        daily_slices = {date: [] for date in sorted(self._index.unique(level=1))}
        for i, (code, date) in enumerate(self._index):
            daily_slices[date].append(self._batch_slices[i])
        self._daily_slices = np.array(list(daily_slices.values()), dtype="object")
        self._daily_index = pd.Series(list(daily_slices.keys()))

        if self.memory_mode == "sample":
            self._memory = np.zeros((len(self._data), self.num_states), dtype=np.float32)
        elif self.memory_mode == "daily":
            self._memory = np.zeros((len(self._daily_index), self.num_states), dtype=np.float32)
        else:
            raise ValueError(f"invalid memory_mode `{self.memory_mode}`")

        self._zeros = np.zeros((self.seq_len, max(self.num_states, self._data.shape[1])), dtype=np.float32)

    def _prepare_seg(self, slc, **kwargs):
        fn = _get_date_parse_fn(self._index[0][1])
        if isinstance(slc, slice):
            start, stop = slc.start, slc.stop
        elif isinstance(slc, (list, tuple)):
            start, stop = slc
        else:
            raise NotImplementedError("This type of input is not supported")
        start_date = pd.Timestamp(fn(start))
        end_date = pd.Timestamp(fn(stop))
        obj = copy.copy(self)
        obj._data = self._data
        obj._tech_data = self._tech_data
        obj._label = self._label
        obj._index = self._index
        obj._memory = self._memory
        obj._zeros = self._zeros
        date_index = self._index.get_level_values(1)
        obj._batch_slices = self._batch_slices[(date_index >= start_date) & (date_index <= end_date)]
        mask = (self._daily_index.values >= start_date) & (self._daily_index.values <= end_date)
        obj._daily_slices = self._daily_slices[mask]
        obj._daily_index = self._daily_index[mask]
        return obj

    def __iter__(self):
        slices, batch_size = self._get_slices()
        indices = np.arange(len(slices))
        if self.shuffle:
            np.random.shuffle(indices)

        for i in range(len(indices))[::batch_size]:
            if self.drop_last and i + batch_size > len(indices):
                break

            data = []
            tech = []
            label = []
            index = []
            state = []
            daily_index = []
            daily_count = []

            for j in indices[i : i + batch_size]:
                slices_subset = slices[j]
                if self.batch_size < 0:
                    idx = self._daily_index.index[j]
                    daily_index.append(idx)
                    if self.memory_mode == "daily":
                        slc = slice(max(idx - self.seq_len - self.horizon, 0), max(idx - self.horizon, 0))
                        state.append(_maybe_padding(self._memory[slc], self.seq_len, self._zeros))
                    if self.n_samples and 0 < self.n_samples < len(slices_subset):
                        slices_subset = np.random.choice(slices_subset, self.n_samples, replace=False)
                    daily_count.append(len(slices_subset))
                else:
                    slices_subset = [slices_subset]

                for slc in slices_subset:
                    if self.input_size:
                        data.append(self._data[slc.stop - 1].reshape(self.input_size, -1).T)
                    else:
                        data.append(_maybe_padding(self._data[slc], self.seq_len, self._zeros))
                    tech.append(self._tech_data[slc.stop - 1])

                    if self.memory_mode == "sample":
                        state.append(_maybe_padding(self._memory[slc], self.seq_len, self._zeros)[: -self.horizon])

                    label.append(self._label[slc.stop - 1])
                    index.append(slc.stop - 1)

            label_array = np.stack(label)
            if label_array.ndim == 2:
                label_array = label_array[:, 0]

            yield {
                "data": _to_tensor(np.stack(data)),
                "tech": _to_tensor(np.stack(tech)),
                "state": _to_tensor(np.stack(state)),
                "label": _to_tensor(label_array),
                "index": np.array(index),
                "daily_index": np.array(daily_index),
                "daily_count": np.array(daily_count),
            }


class DualBranchFusion(nn.Module):
    """Stable v1: Alpha360 sequence encoder + TechAlpha158 point encoder + concat fusion."""

    def __init__(
        self,
        input_size=6,
        hidden_size=64,
        num_layers=2,
        rnn_arch="LSTM",
        use_attn=True,
        dropout=0.0,
        tech_input_size=158,
        tech_hidden_size=64,
        fusion_hidden_size=128,
        **kwargs,
    ):
        super().__init__()
        self.seq_encoder = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_arch=rnn_arch,
            use_attn=use_attn,
            dropout=dropout,
        )
        self.tech_encoder = nn.Sequential(
            nn.LayerNorm(tech_input_size),
            nn.Linear(tech_input_size, tech_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(tech_hidden_size, tech_hidden_size),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.seq_encoder.output_size + tech_hidden_size, fusion_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(fusion_hidden_size),
        )
        self.output_size = fusion_hidden_size

    def forward(self, seq_x, tech_x):
        seq_hidden = self.seq_encoder(seq_x)
        tech_hidden = self.tech_encoder(tech_x)
        return self.fusion(torch.cat([seq_hidden, tech_hidden], dim=1))


class JointTRAModel(TRAModel):
    """TRA variant whose backbone jointly encodes Alpha360 and TechAlpha158."""

    def _init_model(self):
        self.logger.info("init JointTRAModel...")
        self.model = DualBranchFusion(**self.model_config).to(device)
        self.tra = TRA(self.model.output_size, **self.tra_config).to(device)

        if self.init_state:
            state_dict = torch.load(self.init_state, map_location="cpu")
            self.model.load_state_dict(state_dict["model"])
            self.tra.load_state_dict(state_dict["tra"], strict=False)

        if self.reset_router and hasattr(self.tra, "fc"):
            self.tra.fc.reset_parameters()
            if hasattr(self.tra, "router"):
                self.tra.router.reset_parameters()

        if self.freeze_model:
            for param in self.model.parameters():
                param.requires_grad_(False)

        if self.freeze_predictors:
            for param in self.tra.predictors.parameters():
                param.requires_grad_(False)

        self.optimizer = optim.Adam(list(self.model.parameters()) + list(self.tra.parameters()), lr=self.lr)
        self.fitted = False
        self.global_step = -1

    def train_epoch(self, epoch, data_set, is_pretrain=False):
        self.model.train()
        self.tra.train()
        data_set.train()
        self.optimizer.zero_grad()

        max_steps = len(data_set)
        if self.max_steps_per_epoch is not None:
            max_steps = min(self.max_steps_per_epoch, max_steps)

        cur_step = 0
        total_loss = 0.0
        total_count = 0
        for batch in data_set:
            cur_step += 1
            if cur_step > max_steps:
                break
            if not is_pretrain:
                self.global_step += 1

            data = batch["data"]
            tech = batch["tech"]
            state = batch["state"]
            label = batch["label"]
            count = batch["daily_count"]
            index = batch["daily_index"] if self.use_daily_transport else batch["index"]

            with torch.set_grad_enabled(not self.freeze_model):
                hidden = self.model(data, tech)
            all_preds, choice, prob = self.tra(hidden, state)

            if is_pretrain or self.transport_method != "none":
                loss, pred, L, P = self.transport_fn(
                    all_preds,
                    label,
                    choice,
                    prob,
                    state.mean(dim=1),
                    count,
                    self.transport_method if not is_pretrain else "oracle",
                    self.alpha,
                    training=True,
                )
                data_set.assign_data(index, L)
                decay = self.rho ** (self.global_step // 100)
                lamb = 0 if is_pretrain else self.lamb * decay
                reg = prob.log().mul(P).sum(dim=1).mean()
                loss = loss - lamb * reg
            else:
                pred = all_preds.mean(dim=1)
                loss = tra_mod.loss_fn(pred, label)

            (loss / self.update_freq).backward()
            if cur_step % self.update_freq == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
            total_loss += float(loss.item())
            total_count += 1

        return total_loss / max(total_count, 1)

    def test_epoch(self, epoch, data_set, return_pred=False, prefix="test", is_pretrain=False):
        self.model.eval()
        self.tra.eval()
        data_set.eval()

        preds = []
        metrics = []
        probs = []
        p_all = []
        for batch in data_set:
            data = batch["data"]
            tech = batch["tech"]
            state = batch["state"]
            label = batch["label"]
            count = batch["daily_count"]
            index = batch["daily_index"] if self.use_daily_transport else batch["index"]

            with torch.no_grad():
                hidden = self.model(data, tech)
                all_preds, choice, prob = self.tra(hidden, state)

            if is_pretrain or self.transport_method != "none":
                loss, pred, L, P = self.transport_fn(
                    all_preds,
                    label,
                    choice,
                    prob,
                    state.mean(dim=1),
                    count,
                    self.transport_method if not is_pretrain else "oracle",
                    self.alpha,
                    training=False,
                )
                data_set.assign_data(index, L)
                if P is not None and return_pred:
                    p_all.append(pd.DataFrame(P.cpu().numpy(), index=index))
            else:
                pred = all_preds.mean(dim=1)

            pred_frame = pd.DataFrame(
                np.c_[pred.cpu().numpy(), label.cpu().numpy(), all_preds.cpu().numpy()],
                index=batch["index"],
                columns=["score", "label"] + [f"score_{d}" for d in range(all_preds.shape[1])],
            )
            metrics.append(evaluate(pred_frame))
            if return_pred:
                preds.append(pred_frame)
                if prob is not None:
                    probs.append(
                        pd.DataFrame(
                            prob.cpu().numpy(),
                            index=index,
                            columns=[f"prob_{d}" for d in range(all_preds.shape[1])],
                        )
                    )

        metrics_df = pd.DataFrame(metrics)
        metrics_out = {
            "MSE": metrics_df.MSE.mean(),
            "MAE": metrics_df.MAE.mean(),
            "IC": metrics_df.IC.mean(),
            "ICIR": metrics_df.IC.mean() / metrics_df.IC.std(),
        }

        if not return_pred:
            return metrics_out, [], [], []

        preds = pd.concat(preds, axis=0)
        preds.index = data_set.restore_index(preds.index)
        preds.index = preds.index.swaplevel()
        preds.sort_index(inplace=True)

        if probs:
            probs = pd.concat(probs, axis=0)
            if self.use_daily_transport:
                probs.index = data_set.restore_daily_index(probs.index)
            else:
                probs.index = data_set.restore_index(probs.index)
                probs.index = probs.index.swaplevel()
                probs.sort_index(inplace=True)
        if p_all:
            p_all = pd.concat(p_all, axis=0)
            if self.use_daily_transport:
                p_all.index = data_set.restore_daily_index(p_all.index)
            else:
                p_all.index = data_set.restore_index(p_all.index)
                p_all.index = p_all.index.swaplevel()
                p_all.sort_index(inplace=True)
        return metrics_out, preds, probs, p_all

    def fit(self, dataset, evals_result=dict()):
        assert isinstance(dataset, JointMTSDatasetH), "JointTRAModel only supports `JointMTSDatasetH`"
        return super().fit(dataset, evals_result)

    def predict(self, dataset, segment="test"):
        assert isinstance(dataset, JointMTSDatasetH), "JointTRAModel only supports `JointMTSDatasetH`"
        if not self.fitted:
            raise ValueError("model is not fitted yet!")
        test_set = dataset.prepare(segment)
        metrics, preds, _, _ = self.test_epoch(-1, test_set, return_pred=True)
        self.logger.info("test metrics: %s", metrics)
        return preds


def joint_model_config() -> dict:
    tech_input_size = len(TechAlpha158A3DFundamental._technical_feature_config()[1])
    return {
        "model_config": {
            "input_size": 6,
            "hidden_size": 64,
            "num_layers": 2,
            "rnn_arch": "LSTM",
            "use_attn": True,
            "dropout": 0.0,
            "tech_input_size": tech_input_size,
            "tech_hidden_size": 64,
            "fusion_hidden_size": 128,
        },
        "tra_config": {
            "num_states": 3,
            "rnn_arch": "LSTM",
            "hidden_size": 32,
            "num_layers": 1,
            "dropout": 0.0,
            "tau": 1.0,
            "src_info": "LR_TPE",
        },
        "model_type": "RNN",
        "lr": 1e-3,
        "n_epochs": 100,
        "early_stop": 20,
        "eval_freq": 5,
        "lamb": 1.0,
        "rho": 0.99,
        "alpha": 0.5,
        "pretrain": True,
        "transport_method": "router",
        "memory_mode": "sample",
    }


def build_joint_task(
    topk: int = 5,
    n_drop: int = 1,
    window_key: str = "w1",
    account: int = 150000,
    risk_degree: float = 0.70,
    benchmark: str = "SH000300",
    instruments: str = "csi300",
    model_kwargs_override: dict | None = None,
    handler_kwargs_extra: dict | None = None,
    dataset_kwargs_extra: dict | None = None,
    strategy_class: str = "TopkDropoutStrategy",
    strategy_module_path: str = "qlib.contrib.strategy",
    strategy_kwargs_extra: dict | None = None,
    eval_segment: str = "test",
    backtest_segment: str | None = None,
) -> dict:
    window = WINDOWS[window_key]
    model_kwargs = _deep_merge(joint_model_config(), model_kwargs_override)
    handler_kwargs = _deep_merge(DEFAULT_HANDLER_KWARGS_EXTRA, handler_kwargs_extra)
    eval_range = _resolve_segment(window, eval_segment)
    backtest_range = _resolve_segment(window, backtest_segment or eval_segment)

    strategy_kwargs = {
        "signal": "<PRED>",
        "topk": topk,
        "n_drop": n_drop,
        "risk_degree": risk_degree,
    }
    if strategy_kwargs_extra:
        strategy_kwargs.update(strategy_kwargs_extra)

    data_handler_config = {
        "start_time": window["start_time"],
        "end_time": window["end_time"],
        "fit_start_time": window["fit_start_time"],
        "fit_end_time": window["fit_end_time"],
        "instruments": instruments,
    }
    data_handler_config.update(handler_kwargs)

    port_analysis_config = {
        "strategy": {
            "class": strategy_class,
            "module_path": strategy_module_path,
            "kwargs": strategy_kwargs,
        },
        "backtest": {
            "start_time": backtest_range[0],
            "end_time": backtest_range[1],
            "account": account,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "limit_threshold": 0.095,
                "deal_price": "close",
                "open_cost": 0.0005,
                "close_cost": 0.0015,
                "min_cost": 5,
            },
        },
    }

    return {
        "model": {
            "class": "JointTRAModel",
            "module_path": "joint_tra_alpha360_tech158",
            "kwargs": model_kwargs,
        },
        "dataset": {
            "class": "JointMTSDatasetH",
            "module_path": "joint_tra_alpha360_tech158",
            "kwargs": _deep_merge(
                {
                    "handler": {
                        "class": "Alpha360TechAlpha158Handler",
                        "module_path": "joint_tra_alpha360_tech158",
                        "kwargs": data_handler_config,
                    },
                    "segments": {
                        "train": window["train"],
                        "valid": window["valid"],
                        "test": eval_range,
                    },
                    "seq_len": 60,
                    "input_size": int(model_kwargs["model_config"]["input_size"]),
                    "num_states": int(model_kwargs["tra_config"]["num_states"]),
                    "batch_size": 16384,
                    "n_samples": None,
                    "memory_mode": model_kwargs.get("memory_mode", "sample"),
                    "drop_last": True,
                },
                dataset_kwargs_extra,
            ),
        },
        "record": [
            {
                "class": "SignalRecord",
                "module_path": "qlib.workflow.record_temp",
                "kwargs": {"model": "<MODEL>", "dataset": "<DATASET>"},
            },
            {
                "class": "SigAnaRecord",
                "module_path": "qlib.workflow.record_temp",
                "kwargs": {"ana_long_short": False, "ann_scaler": 252},
            },
            {
                "class": "PortAnaRecord",
                "module_path": "qlib.workflow.record_temp",
                "kwargs": {"config": port_analysis_config},
            },
        ],
    }


