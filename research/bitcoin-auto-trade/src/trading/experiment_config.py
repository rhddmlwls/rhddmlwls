"""Strict, typed TOML settings for the reusable offline research pipeline."""
from dataclasses import asdict,dataclass,field
from pathlib import Path
from datetime import date
from itertools import product
import math
import re
import tomllib


FEATURE_KINDS = {"log_return","efficiency","taker_imbalance","volatility","volume_z","funding_mean","oi_change"}


@dataclass(frozen=True)
class FeatureSpec:
    kind: str
    window: int


@dataclass(frozen=True)
class ObservationSpec:
    name: str
    features: tuple[FeatureSpec,...]


@dataclass(frozen=True)
class ModelSpec:
    name: str = "hmm"
    kind: str = "hmm"
    train_window: int = 365
    retrain_every: int = 30
    naming_window: int = 60
    decode_window: int = 61
    iterations: int = 100
    restarts: int = 10
    seeds: tuple[int,...] = (42,137)
    sticky_kappa: float = 0.


@dataclass(frozen=True)
class DecisionSpec:
    ratios: tuple[float,...] = (1.,2.,4.)
    include_viterbi: bool = True
    smoothing_windows: tuple[int,...] = ()


@dataclass(frozen=True)
class TradingSpec:
    initial_cash: float = 10000.
    fees: float = .0005
    slippage: float = .0005
    bull_target: int = 1
    side_target: int = 0
    bear_target: int = -1
    close_final: bool = True
    funding_mode: str = "actual"


@dataclass(frozen=True)
class StrategySpec:
    name: str
    bull: str = "long"
    side: str = "flat"
    bear: str = "short"
    ema_fast: int = 10
    ema_slow: int = 30
    ema_filter: int = 200
    donchian_entry: int = 20
    donchian_exit: int = 10
    breakout_k: float = .5
    breakout_funding_timing: str = "conservative_daily"
    breakout_final_day: str = "skip_entry"


@dataclass(frozen=True)
class SignalSpec:
    name: str
    kind: str
    parameters: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SelectionSpec:
    enabled: bool = False
    selection_end: str = "2023-12-31"
    validation_end: str = "2024-12-31"
    test_start: str = "2025-01-01"
    shortlist_size: int = 10
    min_trades: int = 5
    min_validation_trades: int = 1
    target_sharpe: float = 1.0
    required_seeds: tuple[int,...] = (42,137)
    walk_forward_first_year: int = 2023


@dataclass(frozen=True)
class EvaluationSpec:
    start: str = "2021-03-01"
    end: str = "2026-09-30"
    prefix_bars: int = 100
    reference_labels: str = ""
    reference_case: str = "obs20_hmm_viterbi"
    require_reference_match: bool = False
    sharpe_benchmark: str = "zero"


@dataclass(frozen=True)
class PlotSpec:
    enabled: bool = True
    dpi: int = 150
    recent_start: str = "2024-01-01"
    bull_color: str = "#a4d9bc"
    side_color: str = "#e0e4e9"
    bear_color: str = "#eea8aa"


@dataclass(frozen=True)
class DiagnosticSpec:
    bear_regimes: bool = False
    horizons: tuple[int,...] = (1,5,20)


@dataclass(frozen=True)
class LabelValidationSpec:
    enabled: bool = False
    baseline_window: int = 120
    baseline_threshold: float = 0.0
    profile_windows: tuple[int,...] = (20,60,120)
    horizons: tuple[int,...] = (1,5,20)
    primary_horizon: int = 5
    bootstrap_reps: int = 2000
    block_lengths: tuple[int,...] = (20,60,120)
    primary_block_length: int = 60
    confidence: float = .95
    bootstrap_seed: int = 20261030
    min_group_samples: int = 30
    min_effective_blocks: int = 10
    recent_start: str = "2025-01-01"
    figures: bool = True


@dataclass(frozen=True)
class DownsidePredictionSpec:
    enabled: bool = False
    horizon: int = 5
    prediction_start: str = "2023-01-01"
    train_window: int = 365
    retrain_every: int = 30
    history_lags: int = 61
    c_values: tuple[float,...] = (.1,1.,10.)
    primary_c: float = 1.
    max_iter: int = 3000
    tol: float = 1e-6
    min_class_samples: int = 20
    models: tuple[str,...] = ("base_rate","raw","raw_hmm_delta","history","history_hmm_delta")
    primary_left: str = "history_hmm_delta"
    primary_right: str = "history"
    threshold: float = .5
    calibration_bins: int = 5
    reset_delta_on_refit: bool = True
    bootstrap_reps: int = 2000
    block_lengths: tuple[int,...] = (20,60,120)
    primary_block_length: int = 60
    confidence: float = .95
    bootstrap_seed: int = 20261030
    min_effective_blocks: int = 10
    recent_start: str = "2025-01-01"
    figures: bool = True


@dataclass(frozen=True)
class SmoothingValidationSpec:
    enabled: bool = False
    windows: tuple[int,...] = (1,14)
    primary_window: int = 14
    strides: tuple[int,...] = (1,5)
    primary_stride: int = 5
    anchor: str = "2021-04-30"
    profile_windows: tuple[int,...] = (5,14,120)
    horizons: tuple[int,...] = (1,5,20)
    primary_horizon: int = 5
    baseline_window: int = 14
    baseline_threshold: float = 0.
    bootstrap_reps: int = 2000
    block_lengths: tuple[int,...] = (20,60,120)
    primary_block_length: int = 60
    confidence: float = .95
    bootstrap_seed: int = 20261030
    min_group_samples: int = 30
    min_effective_blocks: int = 10
    recent_start: str = "2025-01-01"
    include_viterbi: bool = True
    figures: bool = True


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    config_path: Path
    data_dir: Path
    output_root: Path
    cache_dir: Path
    observations: tuple[ObservationSpec,...]
    models: tuple[ModelSpec,...]
    decisions: DecisionSpec = field(default_factory=DecisionSpec)
    trading: TradingSpec = field(default_factory=TradingSpec)
    evaluation: EvaluationSpec = field(default_factory=EvaluationSpec)
    plots: PlotSpec = field(default_factory=PlotSpec)
    strategies: tuple[StrategySpec,...] = ()
    signals: tuple[SignalSpec,...] = ()
    selection: SelectionSpec = field(default_factory=SelectionSpec)
    diagnostics: DiagnosticSpec = field(default_factory=DiagnosticSpec)
    label_validation: LabelValidationSpec = field(default_factory=LabelValidationSpec)
    downside_prediction: DownsidePredictionSpec = field(default_factory=DownsidePredictionSpec)
    smoothing_validation: SmoothingValidationSpec = field(default_factory=SmoothingValidationSpec)
    symbol: str = "BTCUSDT"

    def serializable(self):
        result=asdict(self)
        for name in ["config_path","data_dir","output_root","cache_dir"]:
            result[name]=str(result[name])
        return result


def _strict(cls,values,where):
    allowed=set(cls.__dataclass_fields__)
    extra=set(values)-allowed
    if extra:
        raise ValueError(f"{where}: unknown settings {sorted(extra)}")
    try:
        return cls(**values)
    except TypeError as exc:
        raise ValueError(f"{where}: {exc}") from exc


def _name(value,where):
    if not isinstance(value,str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}",value):
        raise ValueError(f"{where}: use a short ASCII name beginning with a letter")


def _label_validation(values):
    where="label_validation"
    if not isinstance(values,dict):
        raise ValueError(f"{where} must be a TOML table")
    values=dict(values)
    for name in ["profile_windows","horizons","block_lengths"]:
        if name in values:
            if not isinstance(values[name],list):
                raise ValueError(f"{where}.{name} must be an array")
            values[name]=tuple(values[name])
    spec=_strict(LabelValidationSpec,values,where)
    if type(spec.enabled) is not bool or type(spec.figures) is not bool:
        raise ValueError(f"{where}: enabled and figures must be boolean")
    sizes={"baseline_window":(spec.baseline_window,2),
           "primary_horizon":(spec.primary_horizon,1),
           "primary_block_length":(spec.primary_block_length,1),
           "bootstrap_reps":(spec.bootstrap_reps,100),
           "bootstrap_seed":(spec.bootstrap_seed,0),
           "min_group_samples":(spec.min_group_samples,2),
           "min_effective_blocks":(spec.min_effective_blocks,2)}
    for name,(value,minimum) in sizes.items():
        if type(value) is not int or value<minimum:
            raise ValueError(f"{where}.{name} must be an integer >= {minimum}")
    for name,minimum in [("profile_windows",2),("horizons",1),("block_lengths",1)]:
        sequence=getattr(spec,name)
        if (not sequence or any(type(value) is not int or value<minimum for value in sequence)
                or len(set(sequence))!=len(sequence)):
            raise ValueError(f"{where}.{name} must contain distinct integers >= {minimum}")
    if spec.primary_horizon not in spec.horizons:
        raise ValueError(f"{where}.primary_horizon must be included in horizons")
    if spec.primary_block_length not in spec.block_lengths:
        raise ValueError(f"{where}.primary_block_length must be included in block_lengths")
    if type(spec.baseline_threshold) not in (int,float) or not math.isfinite(spec.baseline_threshold):
        raise ValueError(f"{where}.baseline_threshold must be finite")
    if (type(spec.confidence) not in (int,float) or not math.isfinite(spec.confidence)
            or not 0<spec.confidence<1):
        raise ValueError(f"{where}.confidence must lie strictly between 0 and 1")
    try:
        if not isinstance(spec.recent_start,str) or date.fromisoformat(spec.recent_start).isoformat()!=spec.recent_start:
            raise ValueError()
    except ValueError:
        raise ValueError(f"{where}.recent_start must be a YYYY-MM-DD date") from None
    return spec


def _downside_prediction(values):
    where="downside_prediction"
    if not isinstance(values,dict):
        raise ValueError(f"{where} must be a TOML table")
    values=dict(values)
    for name in ["c_values","models","block_lengths"]:
        if name in values:
            if not isinstance(values[name],list):
                raise ValueError(f"{where}.{name} must be an array")
            values[name]=tuple(values[name])
    spec=_strict(DownsidePredictionSpec,values,where)
    for name in ["enabled","reset_delta_on_refit","figures"]:
        if type(getattr(spec,name)) is not bool:
            raise ValueError(f"{where}.{name} must be boolean")
    minima={"horizon":1,"train_window":2,"retrain_every":1,"history_lags":1,
            "max_iter":1,"min_class_samples":2,"calibration_bins":2,
            "bootstrap_reps":100,"primary_block_length":1,"bootstrap_seed":0,
            "min_effective_blocks":2}
    for name,minimum in minima.items():
        value=getattr(spec,name)
        if type(value) is not int or value<minimum:
            raise ValueError(f"{where}.{name} must be an integer >= {minimum}")
    def finite_number(value):
        if type(value) not in (int,float):
            return False
        try:
            return math.isfinite(value)
        except OverflowError:
            return False
    for name in ["primary_c","tol"]:
        value=getattr(spec,name)
        if not finite_number(value) or value<=0:
            raise ValueError(f"{where}.{name} must be finite and positive")
    if (not spec.c_values or any(not finite_number(value) or value<=0 for value in spec.c_values)
            or len(set(spec.c_values))!=len(spec.c_values)):
        raise ValueError(f"{where}.c_values must contain distinct finite positive numbers")
    if spec.primary_c not in spec.c_values:
        raise ValueError(f"{where}.primary_c must be included in c_values")
    registered={"base_rate","raw","raw_hmm_delta","history","history_hmm_delta"}
    if (not spec.models or any(not isinstance(model,str) or model not in registered for model in spec.models)
            or len(set(spec.models))!=len(spec.models)):
        raise ValueError(f"{where}.models must contain distinct registered downside prediction models")
    if "base_rate" not in spec.models:
        raise ValueError(f"{where}.models must include base_rate")
    for name in ["primary_left","primary_right"]:
        value=getattr(spec,name)
        if not isinstance(value,str) or value not in spec.models or value=="base_rate":
            raise ValueError(f"{where}.{name} must be included in models and cannot be base_rate")
    if spec.primary_left==spec.primary_right:
        raise ValueError(f"{where}: primary_left and primary_right must differ")
    for name in ["threshold","confidence"]:
        value=getattr(spec,name)
        if not finite_number(value) or not 0<value<1:
            raise ValueError(f"{where}.{name} must lie strictly between 0 and 1")
    if (not spec.block_lengths or any(type(value) is not int or value<1 for value in spec.block_lengths)
            or len(set(spec.block_lengths))!=len(spec.block_lengths)):
        raise ValueError(f"{where}.block_lengths must contain distinct positive integers")
    if spec.primary_block_length not in spec.block_lengths:
        raise ValueError(f"{where}.primary_block_length must be included in block_lengths")
    for name in ["prediction_start","recent_start"]:
        value=getattr(spec,name)
        try:
            if not isinstance(value,str) or date.fromisoformat(value).isoformat()!=value:
                raise ValueError()
        except ValueError:
            raise ValueError(f"{where}.{name} must be a YYYY-MM-DD date") from None
    return spec


def _smoothing_validation(values):
    where="smoothing_validation"
    if not isinstance(values,dict):
        raise ValueError(f"{where} must be a TOML table")
    values=dict(values)
    for name in ["windows","strides","profile_windows","horizons","block_lengths"]:
        if name in values:
            if not isinstance(values[name],list):
                raise ValueError(f"{where}.{name} must be an array")
            values[name]=tuple(values[name])
    spec=_strict(SmoothingValidationSpec,values,where)
    for name in ["enabled","include_viterbi","figures"]:
        if type(getattr(spec,name)) is not bool:
            raise ValueError(f"{where}.{name} must be boolean")
    minima={"primary_window":2,"primary_stride":1,"primary_horizon":1,
            "baseline_window":2,"bootstrap_reps":100,"primary_block_length":1,
            "bootstrap_seed":0,"min_group_samples":2,"min_effective_blocks":2}
    for name,minimum in minima.items():
        value=getattr(spec,name)
        if type(value) is not int or value<minimum:
            raise ValueError(f"{where}.{name} must be an integer >= {minimum}")
    for name,minimum in [("windows",1),("strides",1),("profile_windows",2),
                         ("horizons",1),("block_lengths",1)]:
        sequence=getattr(spec,name)
        if (not sequence or any(type(value) is not int or value<minimum for value in sequence)
                or len(set(sequence))!=len(sequence)):
            raise ValueError(f"{where}.{name} must contain distinct integers >= {minimum}")
    if 1 not in spec.windows:
        raise ValueError(f"{where}.windows must include 1 as the unsmoothed control")
    for primary,sequence in [("primary_window","windows"),("primary_stride","strides"),
                             ("primary_horizon","horizons"),("primary_block_length","block_lengths")]:
        if getattr(spec,primary) not in getattr(spec,sequence):
            raise ValueError(f"{where}.{primary} must be included in {sequence}")
    def finite_number(value):
        if type(value) not in (int,float):
            return False
        try:
            return math.isfinite(value)
        except OverflowError:
            return False
    if not finite_number(spec.baseline_threshold):
        raise ValueError(f"{where}.baseline_threshold must be finite")
    if not finite_number(spec.confidence) or not 0<spec.confidence<1:
        raise ValueError(f"{where}.confidence must lie strictly between 0 and 1")
    for name in ["anchor","recent_start"]:
        value=getattr(spec,name)
        try:
            if not isinstance(value,str) or date.fromisoformat(value).isoformat()!=value:
                raise ValueError()
        except ValueError:
            raise ValueError(f"{where}.{name} must be a YYYY-MM-DD date") from None
    return spec


def load_experiment(path):
    path=Path(path).resolve()
    raw=tomllib.loads(path.read_text(encoding="utf-8"))
    extra=set(raw)-{"name","paths","data","observations","models","decisions","trading","evaluation","plots","strategies","signals","selection","strategy_grids","diagnostics","label_validation","downside_prediction","smoothing_validation"}
    if extra:
        raise ValueError(f"Unknown top-level settings: {sorted(extra)}")
    name=raw.get("name",path.stem)
    _name(name,"name")
    paths=raw.get("paths",{})
    if set(paths)-{"data_dir","output_root","cache_dir"}:
        raise ValueError("Unknown path setting")
    data=raw.get("data",{})
    if not isinstance(data,dict) or set(data)-{"symbol"}:
        raise ValueError("data must be a table containing only symbol")
    symbol=data.get("symbol","BTCUSDT")
    if not isinstance(symbol,str) or not re.fullmatch(r"[A-Z0-9]+USDT",symbol):
        raise ValueError("data.symbol must be an uppercase USDT symbol, for example ETHUSDT")
    base=path.parent
    resolve=lambda key,default:(base/paths.get(key,default)).resolve()
    observations=[]
    for item in raw.get("observations",[]):
        if set(item)-{"name","features"}:
            raise ValueError("Unknown observation setting")
        _name(item.get("name"),"observation name")
        features=tuple(_strict(FeatureSpec,f,"feature") for f in item.get("features",[]))
        if not features or len(set(features))!=len(features):
            raise ValueError("Each observation set needs nonempty distinct features")
        for feature in features:
            if feature.kind not in FEATURE_KINDS or type(feature.window) is not int or feature.window<2:
                raise ValueError(f"Invalid feature: {feature}")
        observations.append(ObservationSpec(item["name"],features))
    models=[]
    for item in raw.get("models",[{}]):
        item=dict(item)
        if "seeds" in item:
            item["seeds"]=tuple(item["seeds"])
        model=_strict(ModelSpec,item,"model")
        _name(model.name,"model name")
        if model.kind not in {"hmm","hsmm_poisson"}:
            raise ValueError(f"Unknown model kind: {model.kind}")
        ints=[model.train_window,model.retrain_every,model.naming_window,model.decode_window,model.iterations,model.restarts,*model.seeds]
        if any(type(x) is not int for x in ints) or min(ints[:6])<1 or model.train_window<60:
            raise ValueError("Model sizes and seeds must be valid integers")
        if model.decode_window>model.train_window+1 or not model.seeds or len(set(model.seeds))!=len(model.seeds) or min(model.seeds)<0:
            raise ValueError("Invalid decode window or seed groups")
        if not math.isfinite(model.sticky_kappa) or model.sticky_kappa<0 or (model.kind=="hsmm_poisson" and model.sticky_kappa):
            raise ValueError("Invalid sticky_kappa for model")
        models.append(model)
    if not observations or not models or len({o.name for o in observations})!=len(observations) or len({m.name for m in models})!=len(models):
        raise ValueError("Need nonempty observation/model sets with unique names")
    prefixes=[f"{o.name}_{m.name}" for o in observations for m in models]
    if len(set(prefixes))!=len(prefixes):
        raise ValueError("Observation/model names produce ambiguous case names")
    decision_data=dict(raw.get("decisions",{}))
    if "ratios" in decision_data:
        decision_data["ratios"]=tuple(decision_data["ratios"])
    if "smoothing_windows" in decision_data:
        if not isinstance(decision_data["smoothing_windows"],list):
            raise ValueError("decisions.smoothing_windows must be an array")
        decision_data["smoothing_windows"]=tuple(decision_data["smoothing_windows"])
    decisions=_strict(DecisionSpec,decision_data,"decisions")
    if (any(type(w) is not int or w<1 for w in decisions.smoothing_windows)
            or len(set(decisions.smoothing_windows))!=len(decisions.smoothing_windows)):
        raise ValueError("decisions.smoothing_windows must contain distinct positive integer day counts")
    if type(decisions.include_viterbi) is not bool or (not decisions.ratios and not decisions.include_viterbi and not decisions.smoothing_windows) or any(not math.isfinite(r) or r<1 for r in decisions.ratios) or len(set(decisions.ratios))!=len(decisions.ratios):
        raise ValueError("Decision ratios must be distinct finite numbers >= 1")
    from .strategy_targets import STRATEGY_IDS
    signals=[]
    if raw.get("signals"):
        from .technical_strategies import validate_technical_params
        for item in raw["signals"]:
            signal=_strict(SignalSpec,item,"signal")
            _name(signal.name,"signal name")
            if signal.name in STRATEGY_IDS:
                raise ValueError("Signal names cannot replace existing strategies")
            params=validate_technical_params(signal.kind,signal.parameters)
            signals.append(SignalSpec(signal.name,signal.kind,params))
        if len({s.name for s in signals})!=len(signals):
            raise ValueError("Signal names must be unique")
    strategy_items=list(raw.get("strategies",[]))
    for grid in raw.get("strategy_grids",[]):
        if set(grid)!={"name","bull","side","bear"}:
            raise ValueError("Strategy grid requires exactly name/bull/side/bear")
        _name(grid["name"],"strategy grid name")
        if any(not isinstance(grid[key],list) or not grid[key] or len(set(grid[key]))!=len(grid[key]) for key in ["bull","side","bear"]):
            raise ValueError("Each grid regime needs nonempty distinct choices")
        for i,(bull,side,bear) in enumerate(product(grid["bull"],grid["side"],grid["bear"]),1):
            strategy_items.append(dict(name=f"{grid['name']}_{i:03d}",bull=bull,side=side,bear=bear))
    strategies=[]
    for item in strategy_items:
        strategy=_strict(StrategySpec,item,"strategy")
        _name(strategy.name,"strategy name")
        choices={strategy.bull,strategy.side,strategy.bear}
        if not choices.issubset(STRATEGY_IDS|{s.name for s in signals}):
            raise ValueError("Strategy choices must be registered builtins or configured signals")
        if "williams" in choices and not choices.issubset({"williams","flat"}):
            raise ValueError("Intraday Williams profiles can only combine williams and flat")
        if type(strategy.breakout_k) not in (int,float) or not math.isfinite(strategy.breakout_k) or not 0<strategy.breakout_k<=1:
            raise ValueError("breakout_k must be finite and in (0, 1]")
        if strategy.breakout_funding_timing not in {"conservative_daily","favorable_daily"}:
            raise ValueError("Unknown breakout funding timing bound")
        if strategy.breakout_final_day!="skip_entry":
            raise ValueError("breakout_final_day must be skip_entry (exit at next available open)")
        periods=[strategy.ema_fast,strategy.ema_slow,strategy.ema_filter,strategy.donchian_entry,strategy.donchian_exit]
        if any(type(p) is not int or p<1 for p in periods) or strategy.ema_fast>=strategy.ema_slow:
            raise ValueError("Invalid strategy indicator periods")
        strategies.append(strategy)
    if len({s.name for s in strategies})!=len(strategies):
        raise ValueError("Strategy names must be unique")
    trading=_strict(TradingSpec,raw.get("trading",{}),"trading")
    if type(trading.funding_mode) is not str or trading.funding_mode not in {"actual","disabled"}:
        raise ValueError("trading.funding_mode must be 'actual' or 'disabled'")
    if not (math.isfinite(trading.initial_cash) and trading.initial_cash>0 and 0<=trading.fees<1 and 0<=trading.slippage<1):
        raise ValueError("Invalid trading cash or costs")
    if any(type(t) is not int or t not in [-1,0,1] for t in [trading.bull_target,trading.side_target,trading.bear_target]) or type(trading.close_final) is not bool:
        raise ValueError("Trading targets must be -1/0/1; close_final must be boolean")
    evaluation=_strict(EvaluationSpec,raw.get("evaluation",{}),"evaluation")
    if evaluation.sharpe_benchmark not in {"zero","btc_price"}:
        raise ValueError("sharpe_benchmark must be zero or btc_price")
    if symbol!="BTCUSDT" and evaluation.sharpe_benchmark=="btc_price":
        raise ValueError("btc_price is BTC-only; use zero for other data.symbol values")
    if type(evaluation.prefix_bars) is not int or evaluation.prefix_bars<1:
        raise ValueError("prefix_bars must be a positive integer")
    if type(evaluation.require_reference_match) is not bool:
        raise ValueError("require_reference_match must be boolean")
    plots=_strict(PlotSpec,raw.get("plots",{}),"plots")
    if type(plots.enabled) is not bool or type(plots.dpi) is not int or not 60<=plots.dpi<=400:
        raise ValueError("Invalid plot settings")
    for color in [plots.bull_color,plots.side_color,plots.bear_color]:
        if not re.fullmatch(r"#[0-9a-fA-F]{6}",color):
            raise ValueError("Plot colors must use #RRGGBB")
    diagnostic_data=raw.get("diagnostics",{})
    if not isinstance(diagnostic_data,dict):
        raise ValueError("diagnostics must be a TOML table")
    diagnostic_data=dict(diagnostic_data)
    if "horizons" in diagnostic_data:
        if not isinstance(diagnostic_data["horizons"],list):
            raise ValueError("diagnostics.horizons must be an array")
        diagnostic_data["horizons"]=tuple(diagnostic_data["horizons"])
    diagnostics=_strict(DiagnosticSpec,diagnostic_data,"diagnostics")
    if type(diagnostics.bear_regimes) is not bool:
        raise ValueError("diagnostics.bear_regimes must be boolean")
    if not diagnostics.horizons or any(type(h) is not int or h<1 for h in diagnostics.horizons) or len(set(diagnostics.horizons))!=len(diagnostics.horizons):
        raise ValueError("diagnostics.horizons must be distinct positive integer bar counts")
    label_validation=_label_validation(raw.get("label_validation",{}))
    downside_prediction=_downside_prediction(raw.get("downside_prediction",{}))
    smoothing_validation=_smoothing_validation(raw.get("smoothing_validation",{}))
    selection_data=dict(raw.get("selection",{}))
    if "required_seeds" in selection_data:
        selection_data["required_seeds"]=tuple(selection_data["required_seeds"])
    selection=_strict(SelectionSpec,selection_data,"selection")
    if type(selection.enabled) is not bool:
        raise ValueError("selection.enabled must be boolean")
    if selection.enabled:
        if not strategies:
            raise ValueError("Chronological selection requires configured strategies")
        selection_end=date.fromisoformat(selection.selection_end)
        validation_end=date.fromisoformat(selection.validation_end)
        test_start=date.fromisoformat(selection.test_start)
        if not selection_end<validation_end<test_start:
            raise ValueError("Selection/validation/test dates must be ordered")
        if any(type(x) is not int or x<1 for x in [selection.shortlist_size,selection.min_trades,selection.min_validation_trades,selection.walk_forward_first_year]):
            raise ValueError("Selection sizes and minimum trades must be positive integers")
        if not math.isfinite(selection.target_sharpe) or selection.target_sharpe<=0:
            raise ValueError("Selection target Sharpe must be finite and positive")
        if not selection.required_seeds or any(type(s) is not int or s<0 for s in selection.required_seeds) or len(set(selection.required_seeds))!=len(selection.required_seeds):
            raise ValueError("Selection required_seeds must be unique nonnegative integers")
        if any(tuple(m.seeds)!=selection.required_seeds for m in models):
            raise ValueError("All selection models must use the exact required seed group")
    return ExperimentSpec(name,path,resolve("data_dir","../data/extended"),resolve("output_root","../reports/experiments"),
        resolve("cache_dir","../.research-cache"),tuple(observations),tuple(models),decisions,trading,evaluation,plots,tuple(strategies),tuple(signals),selection,diagnostics,label_validation,downside_prediction,smoothing_validation,symbol)
