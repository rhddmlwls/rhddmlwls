"""Compact comparison of configured regime-to-strategy assignments."""
from pathlib import Path

import numpy as np


def _strategy_label(name):
    controls = {"directional": "Long / flat / short", "long_only": "Long / flat / flat",
                "ema_global": "EMA in all regimes", "donchian_global": "Donchian in all regimes",
                "buy_hold": "Buy and hold", "williams_k050":"Williams k=0.5 (conservative)",
                "williams_k030":"Williams k=0.3", "williams_k070":"Williams k=0.7",
                "williams_bull_k050":"Williams k=0.5 | bull only",
                "williams_nonbear_k050":"Williams k=0.5 | bull + side",
                "williams_k050_favorable":"Williams k=0.5 (favorable bound)"}
    if name in controls:
        return controls[name]
    words = {"ema": "EMA", "donchian": "Donchian", "long": "Long", "short": "short", "flat": "flat"}
    return " / ".join(words.get(part, part) for part in name.split("_"))


def render_strategy_metrics(summary, out, settings):
    """Show every primary-seed strategy with risk, return and trading intensity."""
    required = {"annual_turnover_gross", "calmar", "sharpe", "max_drawdown", "cagr"}
    if summary.empty or not required.issubset(summary.columns):
        return []
    import os
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((out / ".matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    observations = list(summary.observation.drop_duplicates())
    btc_relative = "sharpe_benchmark" in summary and summary.sharpe_benchmark.iloc[0] == "btc_price"
    ratio_label = "Sharpe vs BTC" if btc_relative else "Sharpe"
    row_count = max(int((summary.observation == obs).sum()) for obs in observations)
    fig, axes = plt.subplots(1, len(observations), figsize=(10 * len(observations), .38 * row_count + 3.8), squeeze=False)
    has_williams=summary.strategy.str.startswith("williams").any()
    headings = ["Strategy" if has_williams else "Bull / side / bear", "Sharpe\nvs BTC" if btc_relative else ratio_label,
                "MDD", "Turnover / yr", "Calmar", "CAGR"]
    controls = {"ema_global", "donchian_global", "buy_hold"}
    for ax, observation in zip(axes[0], observations):
        group = summary[summary.observation == observation].sort_values("sharpe", ascending=False)
        rows = []
        for row in group.itertuples():
            rows.append([_strategy_label(row.strategy), f"{row.sharpe:.2f}", f"{row.max_drawdown:.1%}",
                         f"{row.annual_turnover_gross:.1f}x", f"{row.calmar:.2f}", f"{row.cagr:.1%}"])
        ax.axis("off")
        ranking = "BTC-relative ratio" if btc_relative else "Sharpe"
        ax.set_title(f"{observation} | all strategies ranked by {ranking}", loc="left", fontsize=14, fontweight="bold", pad=14)
        table = ax.table(cellText=rows, colLabels=headings, cellLoc="right", colLoc="right",
                         colWidths=[.41, .10, .11, .16, .10, .12], bbox=[0, .015, 1, .97])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("white")
            cell.PAD = .12
            if c == 0:
                cell.set_text_props(ha="left")
            if r == 0:
                cell.set_facecolor("#263449")
                cell.set_text_props(color="white", weight="bold")
            else:
                is_control = group.iloc[r - 1].strategy in controls
                cell.set_facecolor("#e0f1e8" if r <= 3 else ("#eaf0f8" if is_control else ("#f3f5f7" if r % 2 else "#ffffff")))
                if is_control and c == 0:
                    cell.set_text_props(weight="bold")
    seed = f" | primary seed {int(summary.seed.iloc[0])}" if "seed" in summary else ""
    fig.suptitle(f"Net portfolio metrics{seed}", x=.035, ha="left", fontsize=18, fontweight="bold")
    if btc_relative:
        fig.text(.035, .095, "BTC-relative ratio: mean(strategy daily return - BTC daily return) / sample std of that difference * sqrt(365.25).", fontsize=10)
        fig.text(.035, .072, "Information ratio vs BTC close-to-close returns; timestamps aligned, first return pair excluded | MDD: daily closing equity | Calmar: CAGR / |MDD|.", fontsize=9)
    else:
        fig.text(.035, .072, "Sharpe: daily returns, Rf = 0, annualized with 365.25 days | MDD: daily closing equity | Calmar: CAGR / |MDD|.", fontsize=10)
    fig.text(.035, .049, f"Turnover / yr: sum of absolute order notional / pre-order NAV, divided by years; both reversal legs counted. Green: top 3 {ratio_label}; blue: global controls.", fontsize=9)
    fig.text(.035, .026, "Same evaluation dates; net of fees, slippage and funding | Full-period exploratory ranking; no held-out selection.", fontsize=9)
    fig.subplots_adjust(left=.035, right=.985, top=.83, bottom=.16, wspace=.08)
    name = "strategy_metrics_comparison.png"
    fig.savefig(out / name, dpi=settings.dpi, facecolor="white")
    plt.close(fig)
    return [name]


def render_strategy_comparison(summary, curves, out, settings, initial):
    import os
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str((out / ".matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import FuncFormatter, NullFormatter

    if summary.empty:
        return []
    observations = list(summary.observation.drop_duplicates())
    strategies = list(summary.strategy.drop_duplicates())
    execution_note=("Williams: intraday stop -> next open; controls: next open; daily funding timing bounds."
                    if any(s.startswith("williams") for s in strategies) else "Next-open execution; net of costs and funding.")
    table = summary.pivot(index="strategy", columns="observation", values="total_return").reindex(index=strategies, columns=observations)
    values = table.to_numpy(dtype=float) * 100
    bound = max(float(np.nanmax(np.abs(values))), 1.0)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(10, max(6, .34 * len(strategies) + 2)))
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#dddddd")
    chart = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap,
                      norm=TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound))
    ax.set_xticks(range(len(observations)), observations)
    ax.set_yticks(range(len(strategies)), [_strategy_label(s) for s in strategies])
    for row in range(len(strategies)):
        for col in range(len(observations)):
            value = values[row, col]
            ax.text(col, row, f"{value:+.1f}%" if np.isfinite(value) else "Unavailable",
                    ha="center", va="center", fontsize=9,
                    color="white" if np.isfinite(value) and abs(value) > .7 * bound else "#202631")
    ax.set(title="Cumulative net return | same evaluation dates", xlabel="Observation horizon")
    fig.colorbar(chart, ax=ax, pad=.025, label="Cumulative return (%)")
    fig.text(.015, .025, "Global controls ignore regimes. Exploratory comparison, net of costs and funding.\n"+execution_note, fontsize=9)
    fig.subplots_adjust(left=.36, right=.91, top=.93, bottom=.12)
    fig.savefig(out / "strategy_return_comparison.png", dpi=settings.dpi)
    plt.close(fig)

    available = summary[summary.case.isin(curves)]
    shown_observations = [obs for obs in observations if (available.observation == obs).any()]
    files = ["strategy_return_comparison.png"]
    if not shown_observations:
        return files + render_strategy_metrics(summary, out, settings)
    fig, axes = plt.subplots(len(shown_observations), 1, figsize=(15, 4.8 * len(shown_observations)), squeeze=False)
    palette = plt.get_cmap("tab10")
    colors = {strategy: palette(i % 10) for i, strategy in enumerate(strategies)}
    colors.update(buy_hold="#202631", directional="#777777", ema_global="#0072B2", donchian_global="#D55E00",
                  williams_k050="#CC79A7",williams_bull_k050="#009E73",williams_nonbear_k050="#56B4E9")
    for ax, observation in zip(axes[:, 0], shown_observations):
        group = available[available.observation == observation]
        selected = list(group.sort_values("total_return", ascending=False).head(3).case)
        for strategy in ("directional", "ema_global", "donchian_global", "buy_hold", "williams_k050", "williams_bull_k050", "williams_nonbear_k050"):
            selected.extend(case for case in group.loc[group.strategy == strategy, "case"] if case not in selected)
        for case in selected:
            row = group.set_index("case").loc[case]
            curve = curves[case]
            ax.plot(curve.index, curve.equity / initial, lw=1.5, color=colors[row.strategy],
                    ls="--" if row.strategy == "buy_hold" else "-",
                    label=f"{_strategy_label(row.strategy)}: {row.total_return:+.1%}; MDD {row.max_drawdown:.1%}")
        ax.axhline(1, color="#999999", lw=.7, ls=":")
        subtitle="baseline, regime gates and controls" if any(s.startswith("williams") for s in strategies) else "top 3 full-period returns and fixed controls"
        ax.set(yscale="log", ylabel="Equity / initial capital (log)", title=f"{observation} | {subtitle}")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}x"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.legend(fontsize=8, loc="upper left", ncol=2)
        ax.grid(alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.text(.07, .025, "Exploratory same-period comparison; no held-out selection | "+execution_note, fontsize=9)
    fig.subplots_adjust(left=.07, right=.99, top=.95, bottom=.09, hspace=.3)
    fig.savefig(out / "strategy_equity_comparison.png", dpi=settings.dpi)
    plt.close(fig)
    return files + ["strategy_equity_comparison.png"] + render_strategy_metrics(summary, out, settings)
