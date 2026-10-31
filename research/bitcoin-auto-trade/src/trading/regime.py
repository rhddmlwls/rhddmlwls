"""Past-only HMM fitting, daily endpoint probabilities and trailing averages."""
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .cache import array_digest, atomic_json, atomic_npz, commit_manifest, file_digest, fingerprint, read_manifest

REGIMES = ("bear", "side", "bull")


def _fit(raw, naming, settings, seed):
    """Choose an initialization by training likelihood, never trading profit."""
    scaler = StandardScaler().fit(raw)
    observations = scaler.transform(raw)
    best, errors = None, []
    for restart in range(settings["restarts"]):
        try:
            model = GaussianHMM(n_components=3, covariance_type="diag",
                n_iter=settings["iterations"], random_state=seed + restart,
                transmat_prior=np.ones((3, 3)))
            model.fit(observations)
            likelihood = float(model.score(observations))
            states = model.predict(observations)
            counts = np.bincount(states, minlength=3)
            if not np.isfinite(likelihood) or (counts == 0).any():
                raise ValueError("Nonfinite likelihood or unobserved state")
            means = [float(naming[states == state].mean()) for state in range(3)]
            if best is None or likelihood > best[0]:
                best = (likelihood, model, counts, means, seed + restart)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            errors.append(dict(seed=seed + restart, error=str(exc)))
    if best is None:
        raise RuntimeError(f"No usable 3-state HMM fit: {errors}")
    likelihood, model, counts, means, selected_seed = best
    mapping = {int(raw_state): rank for rank, raw_state in enumerate(np.argsort(means, kind="stable"))}
    history = list(model.monitor_.history)
    change = history[-1] - history[-2] if len(history) > 1 else None
    audit = [dict(raw_state=state, regime=REGIMES[mapping[state]],
        mean_trailing_log_return=means[state], count=int(counts[state]),
        likelihood=likelihood, seed=selected_seed, iterations=model.monitor_.iter,
        converged=bool(change is not None and 0 <= change < model.tol),
        last_likelihood_change=change, self_transition=float(model.transmat_[state, state]),
        tied_state_means=len(set(means)) < 3, rejected_restarts=len(errors),
        restart_errors=json.dumps(errors)) for state in range(3)]
    return scaler, model, mapping, audit


def _cached_fit(raw, naming, settings, seed, cache, versions):
    definition = {key: settings[key] for key in ("iterations", "restarts")}
    key = fingerprint(dict(model=definition, seed=seed, observations=array_digest(raw),
        naming=array_digest(naming), versions=versions,
        source={name: file_digest(Path(__file__).with_name(name)) for name in ("regime.py", "cache.py")}))
    folder = Path(cache) / "fits" / key
    metadata = read_manifest(folder, key, ["parameters.npz"])
    if metadata is not None:
        with np.load(folder / "parameters.npz", allow_pickle=False) as values:
            scaler = StandardScaler()
            scaler.mean_, scaler.var_, scaler.scale_ = [values[name].copy() for name in ("mean", "variance", "scale")]
            scaler.n_features_in_, scaler.n_samples_seen_ = len(scaler.mean_), len(raw)
            model = GaussianHMM(n_components=3, covariance_type="diag", init_params="", params="")
            model.means_, model.covars_ = values["means"].copy(), values["covariances"].copy()
            model.startprob_, model.transmat_ = values["start"].copy(), values["transition"].copy()
        return scaler, model, {int(key): value for key, value in metadata["mapping"].items()}, metadata["audit"], True
    try:
        scaler, model, mapping, audit = _fit(raw, naming, settings, seed)
    except RuntimeError as exc:
        atomic_json(folder / "failure.json", dict(seed=seed, model=definition, error=str(exc)))
        raise
    atomic_npz(folder / "parameters.npz", mean=scaler.mean_, variance=scaler.var_, scale=scaler.scale_,
        means=model.means_, covariances=model._covars_, start=model.startprob_, transition=model.transmat_)
    clean = [{key: None if isinstance(value, float) and not np.isfinite(value) else value
              for key, value in row.items()} for row in audit]
    commit_manifest(folder, key, ["parameters.npz"], mapping=mapping, audit=clean)
    return scaler, model, mapping, audit, False


def compute_regimes(features, naming_returns, settings, seed, cache):
    """Refit on preceding observations; retain only each close's endpoint."""
    observations = features.to_numpy(dtype=float)
    naming = np.asarray(naming_returns, dtype=float)
    if (naming.shape != (len(features),) or not features.index.is_unique
            or not features.index.is_monotonic_increasing):
        raise ValueError("Features need a sorted unique index and aligned naming returns")
    train, refit, decode = (settings[key] for key in ("train_window", "retrain_every", "decode_window"))
    if train < 60 or min(refit, decode, settings["iterations"], settings["restarts"]) < 1 or decode > train + 1:
        raise ValueError("Invalid HMM training, refit or decode window")
    valid = np.isfinite(observations).all(axis=1) & np.isfinite(naming)
    probabilities = pd.DataFrame(np.nan, index=features.index, columns=REGIMES)
    probabilities["raw_state"], probabilities["viterbi"] = -1, -1
    fits, normalization = [], []
    model = scaler = mapping = next_fit = None
    versions = {name: version(name) for name in ("numpy", "pandas", "hmmlearn", "scikit-learn", "threadpoolctl")}
    with threadpool_limits(limits=1):
        for position in range(train, len(observations)):
            if not valid[position - train:position].all() or not valid[position]:
                model = next_fit = None
                continue
            if model is None or position == next_fit:
                raw = observations[position - train:position]
                scaler, model, mapping, audit, cached = _cached_fit(
                    raw, naming[position - train:position], settings, seed, cache, versions)
                transformed = scaler.transform(raw)
                for feature, mean, variance, scale, values in zip(
                        features.columns, scaler.mean_, scaler.var_, scaler.scale_, transformed.T):
                    normalization.append(dict(fit_at=position, train_start=position - train,
                        train_end=position - 1, feature=feature, mean=float(mean),
                        std_population=float(np.sqrt(variance)), scale=float(scale),
                        normalized_mean=float(values.mean()), normalized_std_population=float(values.std(ddof=0))))
                fits.extend(dict(row, fit_at=position, train_start=position - train,
                    train_end=position - 1, train_rows=train, cached=cached) for row in audit)
                next_fit = position + refit
            window = scaler.transform(observations[max(0, position - decode + 1):position + 1])
            raw_state = int(model.predict(window)[-1])
            endpoint = model.predict_proba(window)[-1]
            for raw_state_index, regime in mapping.items():
                probabilities.iloc[position, regime] = endpoint[raw_state_index]
            probabilities.iloc[position, 3:] = raw_state, mapping[raw_state]
    return probabilities, pd.DataFrame(fits), pd.DataFrame(normalization)


def trailing_average(probabilities, window):
    """Average complete t-window+1...t probabilities; exact ties choose bear first."""
    if type(window) is not int or window < 1:
        raise ValueError("Smoothing window must be a positive integer")
    index = probabilities.index
    if (not isinstance(index, pd.DatetimeIndex) or index.hasnans or not index.is_unique
            or not index.is_monotonic_increasing
            or not index.to_series().diff().dropna().eq(pd.Timedelta("1D")).all()):
        raise ValueError("Smoothing requires consecutive daily dates")
    values = probabilities[list(REGIMES)].astype(float)
    array = values.to_numpy()
    known, missing = np.isfinite(array).all(axis=1), np.isnan(array).all(axis=1)
    if (not (known | missing).all() or (array[known] < 0).any() or (array[known] > 1).any()
            or not np.allclose(array[known].sum(axis=1), 1, atol=1e-8, rtol=0)):
        raise ValueError("Each probability row must be a valid simplex or entirely unavailable")
    averaged = values.rolling(window, min_periods=window).mean()
    available = averaged.notna().all(axis=1)
    labels = pd.Series(-1, index=index, name="regime", dtype="int8")
    labels.loc[available] = averaged.loc[available].to_numpy().argmax(axis=1).astype(np.int8)
    return labels, averaged
