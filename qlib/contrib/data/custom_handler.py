from __future__ import annotations

from qlib.contrib.data.handler import Alpha158, check_transform_proc, _DEFAULT_LEARN_PROCESSORS
from qlib.contrib.data.loader import Alpha158DL
from qlib.data.dataset.handler import DataHandlerLP


class TechAlpha158A(Alpha158):
    """Technical-only factor set built from daily OHLCV/VWAP fields."""

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0, 1, 2, 3, 4, 5, 9, 14, 19, 29, 39, 59],
                "feature": ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"],
            },
            "volume": {
                "windows": [0, 1, 2, 3, 4, 5, 9, 14, 19, 29, 39, 59],
            },
            "rolling": {
                "windows": [3, 5, 10, 20, 30, 60],
                "include": [
                    "ROC",
                    "MA",
                    "STD",
                    "BETA",
                    "RSQR",
                    "RESI",
                    "RANK",
                    "RSV",
                    "IMAX",
                    "IMIN",
                    "IMXD",
                    "CORR",
                    "CORD",
                    "CNTP",
                    "CNTD",
                    "SUMP",
                    "SUMD",
                    "VMA",
                    "VSTD",
                    "WVMA",
                    "VSUMP",
                    "VSUMD",
                ],
            },
        }
        return Alpha158DL.get_feature_config(conf)

    def get_label_config(self):
        return ["Ref($close, -6)/Ref($close, -1) - 1"], ["LABEL0"]


class TechAlpha158A3D(TechAlpha158A):
    """Same technical factor set with a 3-day forward return label."""

    def get_label_config(self):
        return ["Ref($close, -4)/Ref($close, -1) - 1"], ["LABEL0"]


class TechAlpha158A2D(TechAlpha158A):
    """Same technical factor set with a 2-day forward return label."""

    def get_label_config(self):
        return ["Ref($close, -3)/Ref($close, -1) - 1"], ["LABEL0"]


class TechAlpha158A4D(TechAlpha158A):
    """Same technical factor set with a 4-day forward return label."""

    def get_label_config(self):
        return ["Ref($close, -5)/Ref($close, -1) - 1"], ["LABEL0"]


class TechAlpha158B(Alpha158):
    """Enhanced technical factor set with richer rolling windows and price-volume structure."""

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0, 1, 2, 3, 4, 5, 7, 9, 14, 19, 29, 39, 59],
                "feature": ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"],
            },
            "volume": {
                "windows": [0, 1, 2, 3, 4, 5, 7, 9, 14, 19, 29, 39, 59],
            },
            "rolling": {
                "windows": [2, 3, 5, 7, 10, 15, 20, 30, 40, 60],
                "include": [
                    "ROC",
                    "MA",
                    "STD",
                    "BETA",
                    "RSQR",
                    "RESI",
                    "MAX",
                    "LOW",
                    "QTLU",
                    "QTLD",
                    "RANK",
                    "RSV",
                    "IMAX",
                    "IMIN",
                    "IMXD",
                    "CORR",
                    "CORD",
                    "CNTP",
                    "CNTN",
                    "CNTD",
                    "SUMP",
                    "SUMN",
                    "SUMD",
                    "VMA",
                    "VSTD",
                    "WVMA",
                    "VSUMP",
                    "VSUMN",
                    "VSUMD",
                ],
            },
        }
        return Alpha158DL.get_feature_config(conf)

    def get_label_config(self):
        return ["Ref($close, -4)/Ref($close, -1) - 1"], ["LABEL0"]


class TechAlpha158B2D(TechAlpha158B):
    """Enhanced technical factor set with a 2-day forward return label."""

    def get_label_config(self):
        return ["Ref($close, -3)/Ref($close, -1) - 1"], ["LABEL0"]


class TechAlpha158B4D(TechAlpha158B):
    """Enhanced technical factor set with a 4-day forward return label."""

    def get_label_config(self):
        return ["Ref($close, -5)/Ref($close, -1) - 1"], ["LABEL0"]


class TechAlpha158A3DFundamental(DataHandlerLP):
    """Technical Alpha158A3D features plus external Tushare daily PIT fundamentals."""

    DEFAULT_FUNDAMENTAL_PATH = (
        "/Users/blueswhen/source_code/work/DL/tushare/processed/training/"
        "csi300_daily_fundamental_features_qlib.parquet"
    )

    @staticmethod
    def _technical_feature_config():
        conf = {
            "kbar": {},
            "price": {
                "windows": [0, 1, 2, 3, 4, 5, 9, 14, 19, 29, 39, 59],
                "feature": ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"],
            },
            "volume": {
                "windows": [0, 1, 2, 3, 4, 5, 9, 14, 19, 29, 39, 59],
            },
            "rolling": {
                "windows": [3, 5, 10, 20, 30, 60],
                "include": [
                    "ROC",
                    "MA",
                    "STD",
                    "BETA",
                    "RSQR",
                    "RESI",
                    "RANK",
                    "RSV",
                    "IMAX",
                    "IMIN",
                    "IMXD",
                    "CORR",
                    "CORD",
                    "CNTP",
                    "CNTD",
                    "SUMP",
                    "SUMD",
                    "VMA",
                    "VSTD",
                    "WVMA",
                    "VSUMP",
                    "VSUMD",
                ],
            },
        }
        return Alpha158DL.get_feature_config(conf)

    @staticmethod
    def _technical_label_config():
        return ["Ref($close, -4)/Ref($close, -1) - 1"], ["LABEL0"]

    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        fundamental_feature_path: str | None = None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        technical_loader = {
            "class": "QlibDataLoader",
            "module_path": "qlib.data.dataset.loader",
            "kwargs": {
                "config": {
                    "feature": self._technical_feature_config(),
                    "label": self._technical_label_config(),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        fundamental_loader = {
            "class": "StaticDataLoader",
            "module_path": "qlib.data.dataset.loader",
            "kwargs": {
                "config": fundamental_feature_path or self.DEFAULT_FUNDAMENTAL_PATH,
            },
        }
        data_loader = {
            "class": "NestedDataLoader",
            "module_path": "qlib.data.dataset.loader",
            "kwargs": {
                "dataloader_l": [technical_loader, fundamental_loader],
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


class TechAlpha158A3DAnnouncement(TechAlpha158A3DFundamental):
    """Technical Alpha158A3D features plus external daily announcement event factors."""

    DEFAULT_ANNOUNCEMENT_PATH = (
        "/home/blueswhen/DL/qlib/tushare/processed/training/"
        "csi300_daily_announcement_features_qlib.parquet"
    )

    def __init__(self, *args, announcement_feature_path: str | None = None, **kwargs):
        super().__init__(
            *args,
            fundamental_feature_path=announcement_feature_path or self.DEFAULT_ANNOUNCEMENT_PATH,
            **kwargs,
        )


class TechAlpha158A3DFundAnnouncement(DataHandlerLP):
    """Technical Alpha158A3D + external fundamental rank-pct + announcement rank-pct factors."""

    DEFAULT_FUNDAMENTAL_PATH = (
        "/home/blueswhen/DL/qlib/tushare/processed/training/"
        "csi300_daily_fundamental_features_rankpct_qlib.parquet"
    )
    DEFAULT_ANNOUNCEMENT_PATH = (
        "/home/blueswhen/DL/qlib/tushare/processed/training/"
        "csi300_daily_announcement_features_rankpct_qlib.parquet"
    )

    @staticmethod
    def _technical_feature_config():
        return TechAlpha158A3DFundamental._technical_feature_config()

    @staticmethod
    def _technical_label_config():
        return TechAlpha158A3DFundamental._technical_label_config()

    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        fundamental_feature_path: str | None = None,
        announcement_feature_path: str | None = None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        technical_loader = {
            "class": "QlibDataLoader",
            "module_path": "qlib.data.dataset.loader",
            "kwargs": {
                "config": {
                    "feature": self._technical_feature_config(),
                    "label": self._technical_label_config(),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        fundamental_loader = {
            "class": "StaticDataLoader",
            "module_path": "qlib.data.dataset.loader",
            "kwargs": {
                "config": fundamental_feature_path or self.DEFAULT_FUNDAMENTAL_PATH,
            },
        }
        announcement_loader = {
            "class": "StaticDataLoader",
            "module_path": "qlib.data.dataset.loader",
            "kwargs": {
                "config": announcement_feature_path or self.DEFAULT_ANNOUNCEMENT_PATH,
            },
        }
        data_loader = {
            "class": "NestedDataLoader",
            "module_path": "qlib.data.dataset.loader",
            "kwargs": {
                "dataloader_l": [technical_loader, fundamental_loader, announcement_loader],
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


class FundamentalGrowthQuality3D(DataHandlerLP):
    """Fundamental-only PIT feature set with a 3-day forward return label."""

    DEFAULT_FUNDAMENTAL_PATH = (
        "/Users/blueswhen/source_code/work/DL/tushare/processed/training/"
        "csi300_daily_fundamental_features_growth_quality_qlib.parquet"
    )

    @staticmethod
    def _label_config():
        return ["Ref($close, -4)/Ref($close, -1) - 1"], ["LABEL0"]

    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        fundamental_feature_path: str | None = None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        label_loader = {
            "class": "QlibDataLoader",
            "module_path": "qlib.data.dataset.loader",
            "kwargs": {
                "config": {
                    "label": self._label_config(),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        fundamental_loader = {
            "class": "StaticDataLoader",
            "module_path": "qlib.data.dataset.loader",
            "kwargs": {
                "config": fundamental_feature_path or self.DEFAULT_FUNDAMENTAL_PATH,
            },
        }
        data_loader = {
            "class": "NestedDataLoader",
            "module_path": "qlib.data.dataset.loader",
            "kwargs": {
                "dataloader_l": [label_loader, fundamental_loader],
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


class FundamentalGrowthQuality20D(FundamentalGrowthQuality3D):
    """Fundamental-only PIT feature set with a 20-day forward return label for stock filtering."""

    @staticmethod
    def _label_config():
        return ["Ref($close, -21)/Ref($close, -1) - 1"], ["LABEL0"]


class FundamentalGrowthQuality63D(FundamentalGrowthQuality3D):
    """Fundamental-only PIT feature set with a 63-day forward return label."""

    @staticmethod
    def _label_config():
        return ["Ref($close, -64)/Ref($close, -1) - 1"], ["LABEL0"]


class FundamentalGrowthQuality126D(FundamentalGrowthQuality3D):
    """Fundamental-only PIT feature set with a 126-day forward return label."""

    @staticmethod
    def _label_config():
        return ["Ref($close, -127)/Ref($close, -1) - 1"], ["LABEL0"]
