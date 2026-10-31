"""One-command research: validated config -> cached fits -> decisions -> PnL -> reports."""
from dataclasses import asdict
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .data import daily_source,load_bars,timestamps,validate_funding_coverage
from .decisions import probability_hysteresis,probability_moving_average
from .directional import DirectionalResult,simulate_directional
from .experiment_config import load_experiment
from .feature_engine import build_feature_set
from .features import compute_cum_return
from .posterior import make_cached_labeler
from .portfolio_metrics import additional_metrics,benchmark_relative_metrics,METRIC_DEFINITIONS
from .regime_diagnostics import comparison_metrics,regime_spells as _spells
from .research_cache import (atomic_json,commit_manifest,file_digest,fingerprint,frame_digest,read_manifest,save_frame)


def _load_frame(path):
    return pd.read_csv(path,index_col=0,parse_dates=True,float_precision="round_trip")


def _trained_probabilities(features,y,model,seed,cache,versions,log,counters):
    source=["hmm.py","posterior.py"]+(["hsmm.py"] if model.kind=="hsmm_poisson" else [])
    definition={k:v for k,v in asdict(model).items() if k not in {"name","seeds"}}
    key=fingerprint(dict(model=definition,seed=seed,features=frame_digest(features),
        naming=frame_digest(y.to_frame()),versions=versions,
        engine={name:file_digest(Path(__file__).with_name(name)) for name in source}))
    folder=cache/"probabilities"/key
    files=["probabilities.csv","fits.csv","normalization.csv"]
    metadata=read_manifest(folder,key,files)
    if metadata is not None:
        counters["probability_hits"]+=1
        log(f"  probabilities: cache hit (seed {seed})")
        return _load_frame(folder/files[0]),pd.read_csv(folder/files[1]),pd.read_csv(folder/files[2]),key
    counters["probability_misses"]+=1
    labeler=make_cached_labeler(model,seed,cache,versions)
    count=[0]
    def progress(i,audit):
        count[0]+=1
        if count[0]==1 or count[0]%10==0:
            log(f"  seed {seed}: fit {count[0]} at {features.index[i].date()} (reused {labeler.fit_cache_hits}, new {labeler.fit_cache_misses})")
    labeler.on_fit=progress
    viterbi=labeler.fit_predict(features,y)
    posterior=pd.DataFrame(labeler.probability_audit).set_index("position")
    result=pd.DataFrame({"viterbi":viterbi},index=features.index)
    result[["bear","side","bull"]]=np.nan
    result.iloc[posterior.index,result.columns.get_indexer(["bear","side","bull"])]=posterior[["bear","side","bull"]].to_numpy()
    fits=pd.DataFrame(labeler.audit).drop(columns="config",errors="ignore")
    normalization=pd.DataFrame(labeler.normalization_audit)
    if fits.empty:
        raise ValueError("No complete training windows for observation set")
    counters["fit_hits"]+=labeler.fit_cache_hits
    counters["fit_misses"]+=labeler.fit_cache_misses
    for name,frame in zip(files,[result,fits,normalization]):
        save_frame(folder/name,frame,index=name=="probabilities.csv")
    commit_manifest(folder,key,files)
    return result,fits,normalization,key


def _execution_funding(funding,mode):
    """Change execution cashflows only; observations retain the original data."""
    if mode=="actual":
        return funding
    if mode!="disabled":
        raise ValueError("Unknown execution funding mode")
    result=funding.copy()
    result["last_funding_rate"]=0.
    return result


def _backtest_cached(bars,targets,funding,trading,cache,counters,strategy=None,setups=None):
    definition=asdict(trading)
    funding_mode=definition.pop("funding_mode","actual")
    # Preserve every pre-existing actual-funding portfolio cache key.
    if funding_mode!="actual":
        definition["funding_mode"]=funding_mode
    funding=_execution_funding(funding,funding_mode)
    key=fingerprint(dict(bars=frame_digest(bars),targets=frame_digest(targets.to_frame()),
        funding=frame_digest(funding),trading=definition,engine=file_digest(Path(__file__).with_name("directional.py"))))
    if setups is not None:
        key=fingerprint(dict(bars=frame_digest(bars),setups=frame_digest(setups),funding=frame_digest(funding),
            trading=definition,funding_timing=strategy.breakout_funding_timing,final_day=strategy.breakout_final_day,
            engine=file_digest(Path(__file__).with_name("williams.py"))))
    folder=cache/"backtests"/key
    files=["equity.csv","orders.csv","positions.csv","stats.json"]
    if read_manifest(folder,key,files) is not None:
        counters["backtest_hits"]+=1
        stats=json.loads((folder/"stats.json").read_text())
        return DirectionalResult(_load_frame(folder/files[0]),pd.read_csv(folder/files[1]),pd.read_csv(folder/files[2]),stats),key
    counters["backtest_misses"]+=1
    if setups is not None:
        from .williams import simulate_williams
        result=simulate_williams(bars,setups,funding,trading.fees,trading.slippage,trading.initial_cash,
            strategy.breakout_funding_timing,strategy.breakout_final_day)
    else:
        result=simulate_directional(bars,targets.to_numpy(),funding,trading.fees,trading.slippage,
            trading.initial_cash,trading.close_final)
    save_frame(folder/files[0],result.curve)
    save_frame(folder/files[1],result.orders,index=False)
    save_frame(folder/files[2],result.positions,index=False)
    atomic_json(folder/files[3],result.stats)
    commit_manifest(folder,key,files)
    return result,key


def _reference(bars,window):
    returns=bars.close/bars.close.shift(window)-1
    path=bars.close.diff().abs().rolling(window,min_periods=window).sum()
    er=(bars.close.diff(window).abs()/path.replace(0.,np.nan)).mask(path.eq(0.),0.)
    threshold=.05 if window==20 else .10
    reference=pd.Series(np.where(returns>threshold,2,np.where(returns< -threshold,0,1)),index=bars.index)
    reference=reference.mask(er<.3,1).where(returns.notna(),-1)
    return reference.astype(int),returns


def _strategy_selection_identity(strategy,observation,decoder,model="hmm"):
    """Deduplicate equivalent hypotheses without merging different parameters."""
    choices={strategy.bull,strategy.side,strategy.bear}
    signature=f"{strategy.bull}/{strategy.side}/{strategy.bear}"
    if "ema" in choices:
        signature+=f" | EMA{strategy.ema_fast}/{strategy.ema_slow} MA{strategy.ema_filter}"
    if "donchian" in choices:
        signature+=f" | Donchian{strategy.donchian_entry}/{strategy.donchian_exit}"
    if "williams" in choices:
        signature+=f" | k={strategy.breakout_k} {strategy.breakout_funding_timing} {strategy.breakout_final_day}"
    prefix="global/" if len(choices)==1 else observation+"/"+model+"/"+decoder+"/"
    return prefix+signature,observation+" | "+signature


def _decision_specs(decisions):
    """Keep legacy names while registering causal posterior-mean decoders."""
    values=[("viterbi",None,None)] if decisions.include_viterbi else []
    for ratio in decisions.ratios:
        name=f"filter_r{int(ratio)}" if float(ratio).is_integer() else "filter_r"+str(float(ratio)).replace(".","p")
        values.append((name,ratio,None))
    values.extend((f"sma{window}",None,window) for window in decisions.smoothing_windows)
    return values


def run_experiment(config_path,check=False):
    cfg=load_experiment(config_path)
    # The public core executes fixed HMM/strategy profiles. Optional research
    # stages are deliberately excluded rather than silently ignored.
    unsupported = [name for name, enabled in {
        "HSMM": any(model.kind != "hmm" for model in cfg.models),
        "chronological selection": cfg.selection.enabled,
        "bear diagnostics": cfg.diagnostics.bear_regimes,
        "label validation": cfg.label_validation.enabled,
        "downside prediction": cfg.downside_prediction.enabled,
        "smoothing validation": cfg.smoothing_validation.enabled,
    }.items() if enabled]
    if unsupported:
        raise ValueError("Not included in the public core: " + ", ".join(unsupported))
    required=[cfg.data_dir/f"{cfg.symbol}_1d.csv",cfg.data_dir/f"{cfg.symbol}_funding_events.csv"]
    kinds={f.kind for o in cfg.observations for f in o.features}
    if "funding_mean" in kinds: required.append(cfg.data_dir/f"{cfg.symbol}_funding_daily.csv")
    if "oi_change" in kinds: required.append(cfg.data_dir/f"{cfg.symbol}_metrics_fields_daily.csv")
    for path in required:
        if not path.is_file(): raise ValueError(f"Missing input: {path}")
    inputs={str(p):file_digest(p) for p in required}
    bars=load_bars(required[0])
    registered_signals={}
    if cfg.signals:
        from .technical_strategies import technical_signal
        signal_cutoff=min(len(bars),max(250,cfg.evaluation.prefix_bars))
        for spec in cfg.signals:
            values=technical_signal(bars,spec.kind,spec.parameters)
            prefix=technical_signal(bars.iloc[:signal_cutoff],spec.kind,spec.parameters)
            pd.testing.assert_series_equal(values.iloc[:signal_cutoff],prefix)
            registered_signals[spec.name]=values
    start=pd.Timestamp(cfg.evaluation.start,tz="UTC")
    end=pd.Timestamp(cfg.evaluation.end,tz="UTC")
    if start>end or start>bars.index[-1] or end<bars.index[0]:
        raise ValueError("Evaluation dates do not overlap price data")
    pd.Timestamp(cfg.plots.recent_start,tz="UTC")
    funding_daily=daily_source(cfg.data_dir/f"{cfg.symbol}_funding_daily.csv") if "funding_mean" in kinds else None
    metrics=daily_source(cfg.data_dir/f"{cfg.symbol}_metrics_fields_daily.csv") if "oi_change" in kinds else None
    features={o.name:build_feature_set(bars,o,funding_daily,metrics) for o in cfg.observations}
    funding=pd.read_csv(required[1])
    funding["datetime"]=timestamps(funding.datetime)
    funding["funding_interval_hours"]=pd.to_numeric(funding.funding_interval_hours,errors="raise")
    if funding.datetime.duplicated().any() or not np.isfinite(funding.funding_interval_hours).all() or not funding.funding_interval_hours.gt(0).all():
        raise ValueError("Invalid funding intervals/dates")
    funding["last_funding_rate"]=pd.to_numeric(funding.last_funding_rate,errors="raise")
    if not np.isfinite(funding.last_funding_rate).all(): raise ValueError("Invalid funding rates")
    expected={}
    expected_decisions={}
    decision_specs=_decision_specs(cfg.decisions)
    max_smoothing=max(cfg.decisions.smoothing_windows,default=1)
    expected_common=pd.Series(True,index=bars.index)
    for obs in cfg.observations:
        for model in cfg.models:
            naming=compute_cum_return(bars,model.naming_window)
            finite=np.isfinite(features[obs.name]).all(axis=1)&np.isfinite(naming)
            available=finite&finite.shift(1,fill_value=False).rolling(model.train_window,min_periods=model.train_window).sum().eq(model.train_window)
            if not (available&(bars.index>=start)&(bars.index<=end)).any():
                raise ValueError(f"{obs.name}/{model.name}: no usable evaluation windows")
            expected[f"{obs.name}/{model.name}"]=int(available.sum())
            decision_available=available.rolling(max_smoothing,min_periods=max_smoothing).sum().eq(max_smoothing)
            expected_decisions[f"{obs.name}/{model.name}"]=int(decision_available.sum())
            expected_common &= decision_available
    expected_common &= (bars.index>=start)&(bars.index<=end)
    common_indices=np.flatnonzero(expected_common)
    if not len(common_indices) or not expected_common.iloc[common_indices[0]:common_indices[-1]+1].all():
        raise ValueError("Evaluation contains unavailable gaps; do not compress time")
    expected_dates=bars.index[common_indices]
    if cfg.selection.enabled:
        for boundary in [cfg.selection.selection_end,cfg.selection.validation_end,cfg.selection.test_start]:
            if not expected_dates[0]<=pd.Timestamp(boundary,tz="UTC")<=expected_dates[-1]:
                raise ValueError("Chronological selection boundaries must lie inside the common evaluation dates")
    funding_coverage=validate_funding_coverage(funding,expected_dates[0],expected_dates[-1]+pd.Timedelta("1D"))
    count=len(cfg.observations)*len(cfg.models)*len(decision_specs)
    count*=max(1,len(cfg.strategies))
    plan=dict(name=cfg.name,symbol=cfg.symbol,market="Binance USD-M perpetual",cases=count,observation_sets=len(cfg.observations),model_specs=len(cfg.models),
        seed_groups={m.name:list(m.seeds) for m in cfg.models},available_rows=expected,
        decision_available_rows=expected_decisions,
        data_start=str(bars.index[0]),data_end=str(bars.index[-1]),
        evaluation_start=str(expected_dates[0]),evaluation_end=str(expected_dates[-1]),
        evaluation_days=len(expected_dates),trading=asdict(cfg.trading),strategies=[asdict(s) for s in cfg.strategies],
        signals=[asdict(s) for s in cfg.signals],selection=asdict(cfg.selection),diagnostics=asdict(cfg.diagnostics),
        label_validation=asdict(cfg.label_validation),downside_prediction=asdict(cfg.downside_prediction),
        smoothing_validation=asdict(cfg.smoothing_validation),funding_coverage=funding_coverage)
    requested_dates=bars.index[(bars.index>=start)&(bars.index<=end)]
    alignment=dict(requested_start=str(start),requested_end=str(end),
        actual_start=str(expected_dates[0]),actual_end=str(expected_dates[-1]),
        requested_days=len(requested_dates),evaluation_days=len(expected_dates),
        start_excluded_days=int((requested_dates<expected_dates[0]).sum()),
        end_excluded_days=int((requested_dates>expected_dates[-1]).sum()),
        decision_smoothing_windows=list(cfg.decisions.smoothing_windows),
        reason="Common causal HMM and decision-window availability; no internal gaps compressed")
    plan["evaluation_alignment"]=alignment
    if cfg.downside_prediction.enabled:
        prediction_start=pd.Timestamp(cfg.downside_prediction.prediction_start,tz="UTC")
        if not expected_dates[0]<=prediction_start<=expected_dates[-1]:
            raise ValueError("downside_prediction.prediction_start must lie inside the explicit evaluation dates")
    if check:
        print(json.dumps(plan,indent=2),flush=True)
        return plan
    versions={p:importlib.metadata.version(p) for p in ["numpy","pandas","hmmlearn","scikit-learn"]}
    code_names=["pipeline.py","experiment_config.py","data.py","features.py","feature_engine.py","decisions.py","posterior.py","hmm.py","directional.py","experiment_plots.py","research_cache.py","portfolio_metrics.py","regime_diagnostics.py"]
    if any(m.kind=="hsmm_poisson" for m in cfg.models): code_names.append("hsmm.py")
    if cfg.strategies: code_names += ["strategies.py","strategy_targets.py","strategy_plots.py"]
    if cfg.signals: code_names.append("technical_strategies.py")
    if cfg.selection.enabled: code_names += ["strategy_selection.py","search_plots.py"]
    if cfg.diagnostics.bear_regimes: code_names.append("bear_diagnostics.py")
    if cfg.label_validation.enabled:
        code_names += ["label_validation.py","label_validation_runner.py","bear_diagnostics.py"]
        if cfg.label_validation.figures: code_names.append("label_validation_plots.py")
    if cfg.downside_prediction.enabled:
        code_names += ["downside_prediction.py","downside_prediction_runner.py",
                       "label_validation.py","bear_diagnostics.py"]
        if cfg.downside_prediction.figures: code_names.append("downside_prediction_plots.py")
    if cfg.smoothing_validation.enabled:
        code_names += ["smoothing_validation.py","smoothing_validation_runner.py",
                       "label_validation.py","bear_diagnostics.py"]
        if cfg.smoothing_validation.figures: code_names.append("smoothing_validation_plots.py")
    from .williams import uses_williams
    has_williams=any(uses_williams(s) for s in cfg.strategies)
    if has_williams: code_names.append("williams.py")
    code={name:file_digest(Path(__file__).with_name(name)) for name in code_names}
    run_key=fingerprint(dict(config=cfg.serializable(),inputs=inputs,code=code,versions=versions))
    out=cfg.output_root/f"{cfg.name}-{run_key[:10]}"
    out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(cfg.config_path,out/"config.toml")
    def log(message):
        print(message,flush=True)
        with (out/"run.log").open("a",encoding="utf-8") as handle:
            handle.write(datetime.now(timezone.utc).isoformat()+" "+message+"\n")
    counters=dict(probability_hits=0,probability_misses=0,fit_hits=0,fit_misses=0,backtest_hits=0,backtest_misses=0,plot_hits=0,plot_misses=0)
    log(f"Experiment {cfg.name}: {cfg.symbol}, {count} cases | {out}")
    log(f"Common evaluation: {expected_dates[0].date()} to {expected_dates[-1].date()} ({len(expected_dates)} days); requested {start.date()} to {end.date()}")
    if cfg.decisions.smoothing_windows and (alignment["start_excluded_days"] or alignment["end_excluded_days"]):
        log(f"Explicit decision warm-up alignment: excluded {alignment['start_excluded_days']} leading and {alignment['end_excluded_days']} trailing requested dates; see evaluation_alignment")
    atomic_json(out/"status.json",dict(status="running",run_key=run_key,plan=plan))
    streams={}
    stream_fits={}
    fit_rows=[]
    validations=[]
    for obs in cfg.observations:
        for model in cfg.models:
            y=compute_cum_return(bars,model.naming_window)
            for seed in model.seeds:
                log(f"Training stage {obs.name}/{model.name}, seed {seed}")
                posterior,fits,norm,key=_trained_probabilities(features[obs.name],y,model,seed,cfg.cache_dir,versions,log,counters)
                valid=posterior.viterbi.ge(0)
                if not posterior.index.equals(bars.index) or not np.isfinite(posterior.loc[valid,["bear","side","bull"]]).all().all():
                    raise ValueError("Invalid cached probability timeline")
                np.testing.assert_allclose(posterior.loc[valid,["bear","side","bull"]].sum(axis=1),1.,atol=1e-8)
                if not (fits.train_end<fits.fit_at).all() or not np.allclose(norm.normalized_mean,0,atol=1e-10) or not np.allclose(norm.normalized_std_population,norm.std_population/norm.scale,atol=1e-10):
                    raise ValueError("Training boundary or normalization audit failed")
                first=int(np.flatnonzero(valid)[0])
                cutoff=min(first+cfg.evaluation.prefix_bars,len(bars))
                checker=make_cached_labeler(model,seed,cfg.cache_dir,versions)
                prefix=checker.fit_predict(features[obs.name].iloc[:cutoff],y.iloc[:cutoff])
                np.testing.assert_array_equal(prefix,posterior.viterbi.iloc[:cutoff])
                prefix_p=pd.DataFrame(checker.probability_audit).set_index("position")
                np.testing.assert_allclose(prefix_p[["bear","side","bull"]],posterior.iloc[prefix_p.index][["bear","side","bull"]],atol=1e-10)
                validations.append(dict(observation=obs.name,model=model.name,seed=seed,normalization=True,prefix_invariance=True,
                    selected_fits=fits.fit_at.nunique(),nonconverged_selected_fits=int((~fits.drop_duplicates("fit_at").converged).sum())))
                folder=out/"training"/f"{obs.name}_{model.name}"/f"seed_{seed}"
                save_frame(folder/"probabilities.csv",posterior)
                save_frame(folder/"fits.csv",fits,index=False)
                save_frame(folder/"normalization.csv",norm,index=False)
                streams[(obs.name,model.name,seed)]=posterior
                if (cfg.diagnostics.bear_regimes or cfg.label_validation.enabled
                        or cfg.downside_prediction.enabled or cfg.smoothing_validation.enabled):
                    stream_fits[(obs.name,model.name,seed)]=fits
                fit_rows.append(fits.assign(observation=obs.name,model=model.name,seed_group=seed))
    # Every seed and decoder is evaluated on the same contiguous dates.
    moving_labels={}
    moving_audits={}
    common=pd.Series(True,index=bars.index)
    for identity,posterior in streams.items():
        common &= posterior.viterbi.ge(0)
        for window in cfg.decisions.smoothing_windows:
            path,audit=probability_moving_average(posterior,window)
            moving_labels[(*identity,window)]=path
            moving_audits[(*identity,window)]=audit
            common &= path.ge(0)
            first=int(np.flatnonzero(posterior.viterbi.ge(0))[0])
            cutoff=min(first+cfg.evaluation.prefix_bars,len(bars))
            prefix,_=probability_moving_average(posterior.iloc[:cutoff],window)
            np.testing.assert_array_equal(prefix.to_numpy(),path.iloc[:cutoff].to_numpy())
            validations.append(dict(observation=identity[0],model=identity[1],seed=identity[2],
                smoothing_window=window,smoothing_prefix_invariance=True,
                first_available=str(path.index[path.ge(0)][0]) if path.ge(0).any() else None))
    common &= (bars.index>=start)&(bars.index<=end)
    indices=np.flatnonzero(common)
    if not len(indices) or not common.iloc[indices[0]:indices[-1]+1].all():
        raise ValueError("Evaluation contains unavailable gaps; do not compress time")
    evaluation=bars.iloc[indices[0]:indices[-1]+1]
    pd.testing.assert_index_equal(evaluation.index,expected_dates)
    # These exact dates already passed event-level funding coverage validation.
    funding=funding[(funding.datetime>=evaluation.index[0])&(funding.datetime<evaluation.index[-1]+pd.Timedelta("1D"))]
    labels=pd.DataFrame(index=bars.index)
    alt_labels={}
    metadata={}
    for obs in cfg.observations:
        for model in cfg.models:
            for decoder,ratio,smoothing_window in decision_specs:
                case=f"{obs.name}_{model.name}_{decoder}"
                title=f"{obs.name} / {model.name} / "+(
                    "Endpoint argmax" if smoothing_window==1 else f"Trailing mean {smoothing_window}d" if smoothing_window is not None
                    else "Viterbi" if ratio is None else f"filter x{ratio:g}")
                metadata[case]=dict(observation=obs.name,model=model.name,decoder=decoder,ratio=ratio,title=title)
                if smoothing_window is not None:
                    metadata[case]["smoothing_window"]=smoothing_window
                for j,seed in enumerate(model.seeds):
                    posterior=streams[(obs.name,model.name,seed)]
                    if smoothing_window is not None:
                        result=moving_labels[(obs.name,model.name,seed,smoothing_window)]
                        audit=moving_audits[(obs.name,model.name,seed,smoothing_window)]
                        save_frame(out/"cases"/case/f"seed_{seed}"/"decisions.csv",audit,index=False)
                    elif ratio is None:
                        result=posterior.viterbi.astype(int)
                    else:
                        result,audit=probability_hysteresis(posterior[["bear","side","bull"]],ratio)
                        if j==0: save_frame(out/"cases"/case/"decisions.csv",audit,index=False)
                    # Threshold state is carried from its original first valid
                    # date, not restarted at a later evaluation boundary.
                    alt_labels[(case,seed)]=result
                    if j==0: labels[case]=result
    diagnostic_report=dict(enabled=cfg.diagnostics.bear_regimes,streams_expected=0,
        streams_completed=0,streams_failed=0,streams=[],failures=[])
    if cfg.diagnostics.bear_regimes:
        from .bear_diagnostics import analyze_bear_regimes
        diagnostic_report["streams_expected"]=len(alt_labels)
        log(f"Descriptive bear diagnostics: {len(alt_labels)} label/seed streams")
        for (label_case,seed),series in alt_labels.items():
            info=metadata[label_case]
            folder=out/"diagnostics"/label_case/f"seed_{seed}"
            identity=dict(label_case=label_case,seed=seed,observation=info["observation"],
                model=info["model"],decoder=info["decoder"],ratio=info["ratio"])
            try:
                diagnostic=analyze_bear_regimes(bars,series,evaluation.index[0],evaluation.index[-1],
                    horizons=cfg.diagnostics.horizons,
                    fits=stream_fits[(info["observation"],info["model"],seed)])
            except (ValueError,RuntimeError) as exc:
                failed=dict(identity,error=str(exc),audit_path=(folder/"audit.json").relative_to(out).as_posix())
                atomic_json(folder/"audit.json",dict(status="failed",**failed))
                diagnostic_report["failures"].append(failed)
                log(f"  {label_case}, seed {seed}: rejected diagnostic: {exc}")
                continue
            paths={}
            for name in ["daily","by_state","yearly","spells"]:
                path=folder/f"{name}.csv"
                save_frame(path,diagnostic[name],index=name=="daily")
                paths[name]=path.relative_to(out).as_posix()
            paths["audit"]=(folder/"audit.json").relative_to(out).as_posix()
            atomic_json(folder/"audit.json",dict(status="complete",**identity,**diagnostic["audit"]))
            diagnostic_report["streams"].append(dict(identity,paths=paths,
                counts={name:len(diagnostic[name]) for name in ["daily","by_state","yearly","spells"]},
                audit=diagnostic["audit"]))
        diagnostic_report["streams_completed"]=len(diagnostic_report["streams"])
        diagnostic_report["streams_failed"]=len(diagnostic_report["failures"])
    # Disabled reports retain the original report schema without importing
    # optional diagnostic implementations that are outside this core bundle.
    def disabled_report():
        return dict(enabled=False, streams_expected=0, streams_completed=0,
            streams_failed=0, streams=[], failures=[], plot_failures=[],
            cache=dict(analysis_hits=0, analysis_misses=0, plot_hits=0, plot_misses=0))
    label_validation_report=disabled_report()
    downside_prediction_report=disabled_report()
    downside_prediction_report["cache"].update(
        forecast_fit_hits=0,forecast_fit_misses=0,forecast_fit_rejected=0)
    smoothing_validation_report=disabled_report()
    reference_match=None
    reference_path=(cfg.config_path.parent/cfg.evaluation.reference_labels).resolve() if cfg.evaluation.reference_labels else None
    if reference_path is not None and reference_path.exists() and cfg.evaluation.reference_case in labels:
        old=_load_frame(reference_path).regime
        reference_match=bool(np.array_equal(old.reindex(labels.index),labels[cfg.evaluation.reference_case]))
        if not reference_match:
            if cfg.evaluation.require_reference_match:
                raise ValueError("Configured legacy baseline does not reproduce saved labels")
            log("Legacy reference differs; retained as a diagnostic for changed configurations")
    trading_cases={}
    strategy_profiles={s.name:s for s in cfg.strategies}
    for label_case,info in metadata.items():
        for strategy in cfg.strategies or (None,):
            case=f"{label_case}__{strategy.name}" if strategy else label_case
            trading_cases[case]=dict(info,label_case=label_case,strategy=strategy.name if strategy else "configured_targets",
                title=info["title"]+(f" / {strategy.name}" if strategy else ""))
            if strategy is not None and cfg.selection.enabled:
                group,title=_strategy_selection_identity(strategy,info["observation"],info["decoder"],info["model"])
                if cfg.decisions.smoothing_windows:
                    title+=f" | {info['decoder']}"
                trading_cases[case].update(bull=strategy.bull,side=strategy.side,bear=strategy.bear,
                    selection_group=group,title=title)
    all_summary=[]
    side_rows=[]
    year_rows=[]
    spell_rows=[]
    curve_primary={}
    selection_curves={}
    selection_positions={}
    profile_rows=[]
    failures=[]
    references={window:_reference(bars,window) for window in [20,60]}
    for case,info in trading_cases.items():
        label_case=info["label_case"]
        model=next(m for m in cfg.models if m.name==info["model"])
        primary=labels.loc[evaluation.index,label_case]
        runs=_spells(primary)
        spell_rows.append(runs.assign(case=case))
        switches=int(primary.ne(primary.shift()).iloc[1:].sum())
        completed=~(runs.left_censored|runs.right_censored)
        for state,name in enumerate(["bear","side","bull"]):
            mask=common&labels[label_case].eq(state)
            profile_rows.append(dict(case=case,regime=name,bars=int(mask.sum()),
                return20_median=float(np.log(bars.close/bars.close.shift(20)).loc[mask].median()) if mask.any() else None,
                return60_median=float(np.log(bars.close/bars.close.shift(60)).loc[mask].median()) if mask.any() else None))
        for j,seed in enumerate(model.seeds):
            series=alt_labels[(label_case,seed)].loc[evaluation.index]
            semantic={}
            for window,(ref,returns) in references.items():
                cmp=comparison_metrics(alt_labels[(label_case,seed)],ref,returns,common)
                semantic.update({f"reference{window}_agreement":cmp["agreement"],f"return{window}_spearman":cmp["spearman"]})
            seed_runs=_spells(series)
            seed_completed=~(seed_runs.left_censored|seed_runs.right_censored)
            targets=series.map({0:cfg.trading.bear_target,1:cfg.trading.side_target,2:cfg.trading.bull_target})
            strategy=None
            setups=None
            if cfg.strategies:
                from .strategy_targets import build_strategy_targets
                strategy=next(s for s in cfg.strategies if s.name==info["strategy"])
                if uses_williams(strategy):
                    from .williams import build_williams_setups
                    full_setups=build_williams_setups(bars,alt_labels[(label_case,seed)],strategy)
                    cutoff=min(len(bars),indices[0]+cfg.evaluation.prefix_bars)
                    pd.testing.assert_frame_equal(build_williams_setups(bars.iloc[:cutoff],alt_labels[(label_case,seed)].iloc[:cutoff],strategy),full_setups.iloc[:cutoff])
                    setups=full_setups.loc[evaluation.index].copy()
                    targets=setups.eligible.astype(int).rename("eligible_at_open")
                    validations.append(dict(case=case,seed=seed,williams_prefix_invariance=True,
                        gate="prior close",range="prior completed day",funding_timing=strategy.breakout_funding_timing))
                else:
                    targets=build_strategy_targets(bars,alt_labels[(label_case,seed)],strategy,registered_signals).loc[evaluation.index]
            try:
                result,key=_backtest_cached(evaluation,targets,funding,cfg.trading,cfg.cache_dir,counters,strategy,setups)
            except RuntimeError as exc:
                failures.append(dict(case=case,seed=seed,error=str(exc)))
                log(f"  {case}, seed {seed}: rejected backtest: {exc}")
                continue
            result.stats.update(additional_metrics(result.curve,result.orders,cfg.trading.initial_cash))
            result.stats.update(benchmark_relative_metrics(result.curve,evaluation.close))
            if cfg.symbol!="BTCUSDT" and "sizing" in result.stats:
                result.stats["sizing"]=result.stats["sizing"].replace("BTC quantity",f"{cfg.symbol.removesuffix('USDT')} quantity")
            if cfg.selection.enabled:
                from .strategy_selection import trade_metrics
                result.stats.update(trade_metrics(result.positions))
                selection_curves[(case,seed)]=result.curve
                selection_positions[(case,seed)]=result.positions
            result.stats["sharpe_zero"]=result.stats["sharpe"]
            result.stats["sharpe_benchmark"]=cfg.evaluation.sharpe_benchmark
            if cfg.evaluation.sharpe_benchmark=="btc_price":
                result.stats["sharpe"]=result.stats["sharpe_btc"]
            folder=out/"cases"/case/f"seed_{seed}"
            if cfg.strategies: save_frame(folder/"targets.csv",targets.to_frame())
            if setups is not None:
                audit=setups.assign(trigger_reached=evaluation.high.ge(setups.trigger),
                    entry_allowed_by_final_day=evaluation.index!=evaluation.index[-1])
                audit["entry_expected"]=audit.eligible & audit.trigger_reached & audit.entry_allowed_by_final_day
                if int(audit.entry_expected.sum())!=len(result.positions):
                    raise ValueError("Williams entry ledger does not match setup audit")
                save_frame(folder/"setups.csv",audit)
            save_frame(folder/"equity.csv",result.curve)
            save_frame(folder/"orders.csv",result.orders,index=False)
            save_frame(folder/"positions.csv",result.positions,index=False)
            for direction in [1,-1]:
                positions=result.positions[result.positions.direction.eq(direction)]
                side_rows.append(dict(case=case,seed=seed,direction=direction,positions=len(positions),
                    gross_pnl=float(positions.price_pnl.sum()),funding_cost=float(positions.funding_cost.sum()),
                    fees=float(positions.fees.sum()),slippage=float(positions.slippage.sum()),net_pnl=float(positions.net_pnl.sum())))
            previous=result.curve.equity.shift(1,fill_value=cfg.trading.initial_cash)
            for year in sorted(evaluation.index.year.unique()):
                mask=result.curve.index.year==year
                frame=result.curve.loc[mask]
                opening=float(previous.loc[mask].iloc[0])
                year_rows.append(dict(case=case,seed=seed,year=int(year),net_pnl=float(frame.equity.iloc[-1]-opening),
                    total_return=float(frame.equity.iloc[-1]/opening-1),long_pnl=float(frame.long_pnl.sum()),short_pnl=float(frame.short_pnl.sum())))
            row=dict(case=case,seed=seed,primary=j==0,observation=info["observation"],model=info["model"],decoder=info["decoder"],strategy=info["strategy"],
                ratio=info["ratio"],label_switches=int(series.ne(series.shift()).iloc[1:].sum()),
                median_spell_days=float(seed_runs.days.median()),seed_label_agreement=float(series.eq(primary).mean()),
                short_complete_spells_le2=int((seed_completed&seed_runs.days.le(2)).sum()),
                **semantic,**result.stats)
            if info.get("smoothing_window") is not None:
                row["smoothing_window"]=info["smoothing_window"]
            all_summary.append(row)
            if j==0:
                curve_primary[case]=result.curve
                log(f"  {case}: {switches} switches | median {row['median_spell_days']:g}d | net {row['total_return']:+.2%}")
    summary=pd.DataFrame(all_summary)
    if summary.empty: raise RuntimeError("Every backtest was rejected; inspect logs")
    primary_summary=summary.loc[summary.primary].copy()
    if cfg.signals:
        skipped_global_strategies=[]
        checked_global_strategies=[]
        for strategy in cfg.strategies:
            if len({strategy.bull,strategy.side,strategy.bear})==1:
                group=summary[summary.strategy.eq(strategy.name)]
                if group.empty:
                    skipped_global_strategies.append(strategy.name)
                    continue
                checked_global_strategies.append(strategy.name)
                for column in ["final_equity","max_drawdown","transaction_cost","funding_cost","positions"]:
                    np.testing.assert_allclose(group[column],group[column].iloc[0],rtol=1e-12,atol=1e-9)
        validations.append(dict(technical_prefix_invariance=True,
            global_strategy_seed_observation_invariance=True if checked_global_strategies else None,
            global_strategy_invariance_scope="completed portfolios only; fully rejected strategies are retained in backtest_rejections",
            global_strategy_invariance_checked=checked_global_strategies,
            global_strategy_invariance_skipped=skipped_global_strategies,
            technical_signal_count=len(cfg.signals)))
    save_frame(out/"summary.csv",primary_summary,index=False)
    save_frame(out/"all_seed_results.csv",summary,index=False)
    save_frame(out/"all_labels.csv",labels)
    save_frame(out/"pnl_by_side.csv",pd.DataFrame(side_rows),index=False)
    save_frame(out/"yearly.csv",pd.DataFrame(year_rows),index=False)
    save_frame(out/"spells.csv",pd.concat(spell_rows,ignore_index=True),index=False)
    save_frame(out/"state_profiles.csv",pd.DataFrame(profile_rows),index=False)
    save_frame(out/"all_fits.csv",pd.concat(fit_rows,ignore_index=True),index=False)
    selection_result=None
    if cfg.selection.enabled:
        from .strategy_selection import select_candidates
        settings=asdict(cfg.selection)
        settings.pop("enabled")
        settings["primary_seed"]=cfg.models[0].seeds[0]
        selection_result=select_candidates(selection_curves,selection_positions,trading_cases,
            evaluation.close,cfg.trading.initial_cash,settings)
        for name in ["period_metrics","rankings","walk_forward"]:
            save_frame(out/f"{name}.csv",selection_result[name],index=False)
        atomic_json(out/"selection.json",dict(selected=selection_result["selected"],audit=selection_result["audit"]))
        log("Chronological selection: "+str(selection_result["selected"]))
    image_files=[]
    if cfg.plots.enabled:
        plot_key=fingerprint(dict(labels=frame_digest(labels.loc[evaluation.index]),summary=frame_digest(primary_summary),
            settings=asdict(cfg.plots),engine=code.get("search_plots.py",code.get("strategy_plots.py",code["experiment_plots.py"]))))
        image_folder=out/"images"
        manifest_path=image_folder/"manifest.json"
        try:
            existing=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        except (OSError,ValueError):
            existing={}
        expected_images=list(existing.get("files",{}))
        if expected_images and read_manifest(image_folder,plot_key,expected_images) is not None:
            counters["plot_hits"]+=1
            image_files=expected_images
        else:
            counters["plot_misses"]+=1
            log("Rendering regime backgrounds and portfolio curves")
            if selection_result is not None:
                from .search_plots import render_search_report
                chosen=selection_result["selected"]
                show=list(dict.fromkeys(([chosen["winner"]] if chosen.get("winner") else [])+chosen.get("shortlist",[])[:2]))
                controls=[]
                for strategy_name in ["long_ema_flat","ema_global","buy_hold"]:
                    candidates=[case for case,info in trading_cases.items() if info["strategy"]==strategy_name and case in curve_primary]
                    if candidates: controls.append(next((c for c in candidates if trading_cases[c]["observation"]=="obs120"),candidates[0]))
                labels_for_plot={case:info["title"] for case,info in trading_cases.items()}
                plot_metrics=selection_result["period_metrics"].loc[selection_result["period_metrics"].seed.eq(chosen["primary_seed"])]
                image_files=render_search_report(plot_metrics,curve_primary,show,controls,
                    image_folder,cfg.trading.initial_cash,cfg.selection.selection_end,cfg.selection.validation_end,
                    cfg.plots,labels=labels_for_plot)
            elif cfg.strategies:
                from .strategy_plots import render_strategy_comparison
                image_files=render_strategy_comparison(primary_summary,curve_primary,image_folder,cfg.plots,cfg.trading.initial_cash)
            else:
                from .experiment_plots import render_experiment
                image_files=render_experiment(evaluation,labels.loc[evaluation.index],metadata,primary_summary,
                    curve_primary,image_folder,cfg.plots,cfg.trading.initial_cash)
            commit_manifest(image_folder,plot_key,image_files)
    # JSON serialization uses null for absent diagnostic cells, never NaN.
    clean_summary=json.loads(primary_summary.to_json(orient="records"))
    metric_definitions=dict(METRIC_DEFINITIONS)
    if cfg.symbol!="BTCUSDT":
        for metric in ["sharpe_btc","btc_annualized_excess_mean","btc_tracking_error"]:
            metric_definitions[metric]=metric_definitions[metric].replace("BTC",cfg.symbol)
    if cfg.selection.enabled:
        from .strategy_selection import SELECTION_METRIC_DEFINITIONS
        metric_definitions.update(SELECTION_METRIC_DEFINITIONS)
    if cfg.evaluation.sharpe_benchmark=="btc_price":
        metric_definitions["sharpe"]=METRIC_DEFINITIONS["sharpe_btc"]
    status="complete" if not failures else "complete_with_backtest_rejections"
    if (diagnostic_report["streams_failed"] or label_validation_report["streams_failed"]
            or label_validation_report["plot_failures"] or downside_prediction_report["streams_failed"]
            or downside_prediction_report["plot_failures"] or smoothing_validation_report["streams_failed"]
            or smoothing_validation_report["plot_failures"]):
        status="complete_with_backtest_and_diagnostic_rejections" if failures else "complete_with_diagnostic_rejections"
    report=dict(status=status,run_key=run_key,symbol=cfg.symbol,market="Binance USD-M perpetual",
        config=cfg.serializable(),inputs=inputs,source_sha256=code,versions=versions,
        evaluation_start=str(evaluation.index[0]),evaluation_end=str(evaluation.index[-1]),evaluation_days=len(evaluation),
        evaluation_alignment=alignment,funding_coverage=funding_coverage,
        cases=trading_cases,summary=clean_summary,validation=validations,legacy_baseline_matches=reference_match,
        cache=counters,backtest_rejections=failures,images=image_files,bear_diagnostics=diagnostic_report,label_validation=label_validation_report,
        downside_prediction=downside_prediction_report,
        smoothing_validation=smoothing_validation_report,
        metric_definitions=metric_definitions,
        interpretation=("Exploratory regime/strategy mapping search on previously studied dates; ranking is in-sample and not an untouched test. EMA/Donchian are existing long/flat strategies, computed on full causal price history and gated by close-time regimes; execution is next open. Indicator state continues across regimes."
            if cfg.strategies else "Exploratory fixed-threshold comparison on previously studied dates; no optimal threshold selected. Probability hysteresis uses only endpoint probabilities, not future smoothing. Ratios are model-state evidence, not expected profit."),
        execution="Previous close -> next open; configured signed targets, fixed quantity after 1x entry; final-close costs as configured. Actual funding rates; daily open proxies missing event mark prices. Exchange liquidation not modeled.",
        references="20/60-bar return-and-ER proxy rules share inputs and are descriptive, not independent true regimes.")
    if has_williams:
        report["execution"]="Williams: prior-close regime gate, current-open + k * prior-day high-low stop entry, next-open exit. Other strategies retain close -> next-open execution. Williams skips final-day entries; sizing and missing-mark proxies follow the existing 1x convention."
        report["interpretation"]="Exploratory comparison on previously studied dates, not an untouched test. k=0.5 is the declared baseline; k=0.3/0.7 are sensitivity checks, not selected optima. Daily OHLC assumes stop fill at threshold plus configured costs and cannot resolve intraday jumps or entry timestamps. Entry-day funding uses explicit conservative/favorable timing bounds conditional on daily-open mark proxies; exit-boundary funding is charged before exit."
    if cfg.signals:
        report["technical_signals"]=[asdict(s) for s in cfg.signals]
        report["technical_prefix_invariance"]=True
        report["interpretation"]="Exploratory technical-signal and HMM regime mapping search on historical data previously inspected. Signals use completed closes and execute at the next open. Intraday VWAP is not available: rolling_vwap uses rolling daily quote-volume / base-volume. Sharpe_zero is conventional Rf=0 Sharpe; sharpe_btc is BTC-relative information ratio. All tried candidates and seed groups are retained."
    report["price_benchmark"]=dict(symbol=cfg.symbol,source="same input perpetual closing prices; excludes fees and funding",
        legacy_metric_names=["sharpe_btc","btc_annualized_excess_mean","btc_tracking_error"])
    if cfg.symbol!="BTCUSDT":
        report["interpretation"]=report["interpretation"].replace("BTC-relative",f"{cfg.symbol}-relative")
        report["price_benchmark"]["legacy_name_note"]="The retained *_btc / btc_* column names refer to this asset's own price benchmark, not an external BTC series."
    if selection_result is not None:
        report["chronological_selection"]=selection_result["selected"]
        report["selection_audit"]=selection_result["audit"]
        report["interpretation"]+=" Selection is chronological but retrospectively designed on an already-studied dataset, not an untouched test. The 2025+ segment is report-only for the declared 2024-selected winner. Multiple candidate search can inflate the best observed Sharpe; no guaranteed target attainment."
    report["funding_policy"]=dict(mode=cfg.trading.funding_mode,
        execution_cashflows="signed actual funding payments and credits" if cfg.trading.funding_mode=="actual" else "both funding payments and credits disabled; event dates and coverage retained",
        model_observations="original source data, including any actual funding_mean feature; unchanged by execution funding mode")
    if cfg.trading.funding_mode=="disabled":
        report["execution"]=report["execution"].replace("Actual funding rates;","Funding cashflows disabled;")
        report["execution"]+=" Both funding payments and credits are zero; fees, slippage, execution timing and sizing are retained."
        report["interpretation"]+=" Funding-disabled sensitivity with the same original HMM observations and decision rules; positions are resized from the resulting equity, not obtained by adding historical funding to a fixed portfolio."
    atomic_json(out/"report.json",report)
    atomic_json(out/"status.json",dict(status=report["status"],run_key=run_key,cache=counters,
        diagnostic_streams_completed=diagnostic_report["streams_completed"],diagnostic_streams_failed=diagnostic_report["streams_failed"],
        label_validation_streams_completed=label_validation_report["streams_completed"],
        label_validation_streams_failed=label_validation_report["streams_failed"]))
    atomic_json(cfg.output_root/f"{cfg.name}-latest.json",dict(output=str(out),run_key=run_key,status=report["status"]))
    log(primary_summary[["case","label_switches","median_spell_days","total_return","max_drawdown"]].to_string(index=False))
    log(f"Complete | cache {counters} | {out}")
    return dict(output=str(out),**report)
