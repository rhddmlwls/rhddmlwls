"""Generic plots for any configured observation/model/decision matrix."""
from pathlib import Path

import numpy as np
import pandas as pd


def render_experiment(bars,labels,metadata,summary,curves,out,settings,initial):
    import os
    os.environ.setdefault("MPLCONFIGDIR",str((Path(out)/".matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as dates
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter,NullFormatter,PercentFormatter

    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    colors={0:settings.bear_color,1:settings.side_color,2:settings.bull_color,-1:"#ffffff"}
    legend=[Patch(facecolor=colors[i],label=s) for i,s in [(2,"Bull"),(1,"Side / middle-ranked"),(0,"Bear")]]
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"figure.facecolor":"white"})
    def panel(ax,case,left,right):
        mask=(bars.index>=left)&(bars.index<=right)
        frame=bars.loc[mask]
        values=labels.loc[mask,case].to_numpy()
        x=dates.date2num(frame.index.to_pydatetime())
        boundaries=np.r_[0,np.flatnonzero(values[1:]!=values[:-1])+1,len(values)]
        for a,b in zip(boundaries[:-1],boundaries[1:]):
            ax.axvspan(x[a]-.5,x[b-1]+.5,color=colors[int(values[a])],lw=0,zorder=0)
        ax.plot(frame.index,frame.close,color="#202631",lw=1.1,zorder=3)
        ax.set(xlim=(x[0]-.5,x[-1]+.5),yscale="log",ylim=(frame.close.min()*.9,frame.close.max()*1.12))
        low,high=ax.get_ylim()
        ax.set_yticks([v for v in [10000,15000,20000,30000,40000,60000,100000,150000] if low<=v<=high])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v,p:f"${v/1000:g}k"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(dates.YearLocator())
        ax.xaxis.set_major_formatter(dates.DateFormatter("%Y"))
        ax.grid(axis="y",color="white",alpha=.7,lw=.7)
        ax.spines[["top","right"]].set_visible(False)
        title=metadata[case]["title"]
        switches=np.count_nonzero(values[1:]!=values[:-1])
        median=np.median(np.diff(boundaries))
        ax.set_title(f"{title} | shown: {switches} switches; median {median:g}d",loc="left",fontsize=10,fontweight="bold")
        ax.set_ylabel("BTCUSDT (log)")
    first,last=bars.index[0],bars.index[-1]
    for case in labels:
        fig,ax=plt.subplots(figsize=(16,5.5))
        panel(ax,case,first,last)
        fig.legend(handles=legend,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,.03))
        fig.text(.07,.012,"Close-time labels only | probabilities unchanged by switching threshold | full-height regime backgrounds",fontsize=9)
        fig.subplots_adjust(left=.07,right=.99,top=.9,bottom=.17)
        fig.savefig(out/f"regime_{case}.png",dpi=settings.dpi)
        plt.close(fig)
    groups={}
    for case in labels:
        group=(metadata[case]["observation"],metadata[case]["model"])
        groups.setdefault(group,[]).append(case)
    pages=[list(groups.items())[i:i+3] for i in range(0,len(groups),3)]
    for period,left in [("full",first),("recent",max(first,pd.Timestamp(settings.recent_start,tz="UTC")))]:
        if left>last:
            continue
        for number,page in enumerate(pages):
            rows=max(len(cases) for group,cases in page)
            fig,axes=plt.subplots(rows,len(page),figsize=(10.5*len(page),3.1*rows+1),squeeze=False)
            for column,(group,cases) in enumerate(page):
                for row in range(rows):
                    ax=axes[row,column]
                    if row<len(cases): panel(ax,cases[row],left,last)
                    else: ax.axis("off")
            fig.suptitle("Same dates and prices | observation horizon and evidence threshold",fontsize=16,y=.99)
            fig.legend(handles=legend,loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.5,.008))
            fig.subplots_adjust(left=.055,right=.99,top=.95,bottom=.065,hspace=.35,wspace=.14)
            suffix="" if len(pages)==1 else f"_{number+1}"
            fig.savefig(out/f"comparison_{period}{suffix}.png",dpi=settings.dpi)
            plt.close(fig)
    if curves:
        fig,axes=plt.subplots(2,1,figsize=(16,9),gridspec_kw={"height_ratios":[2.3,1]})
        palette=plt.get_cmap("tab10")
        for i,(case,curve) in enumerate(curves.items()):
            row=summary.set_index("case").loc[case]
            color=palette(i%10)
            axes[0].plot(curve.index,curve.equity,color=color,lw=1.7,
                label=f"{metadata[case]['title']}: {row.total_return:+.1%}")
            dd=curve.equity/curve.equity.cummax().clip(lower=initial)-1
            axes[1].plot(curve.index,dd,color=color,lw=1.1)
        axes[0].set(yscale="log",ylabel="Equity (USDT)",title="Bull long / bear short / side flat | net of costs and funding")
        low,high=axes[0].get_ylim()
        axes[0].set_yticks([v for v in [100,500,1000,2000,4000,6000,10000,15000,20000,40000,100000] if low<=v<=high])
        axes[0].yaxis.set_major_formatter(FuncFormatter(lambda v,p:f"{v:,.0f}"))
        axes[0].yaxis.set_minor_formatter(NullFormatter())
        axes[0].axhline(initial,color="#888888",lw=.7,ls=":")
        axes[0].legend(ncol=2,fontsize=8,loc="best")
        axes[1].set(ylabel="Close drawdown",xlabel="UTC date | next-open execution")
        axes[1].yaxis.set_major_formatter(PercentFormatter(1))
        for ax in axes:
            ax.xaxis.set_major_locator(dates.YearLocator())
            ax.xaxis.set_major_formatter(dates.DateFormatter("%Y"))
            ax.set_xlim(first,last)
            ax.grid(alpha=.15)
            ax.spines[["top","right"]].set_visible(False)
        fig.subplots_adjust(left=.08,right=.99,top=.94,bottom=.09,hspace=.25)
        fig.savefig(out/"equity_comparison.png",dpi=settings.dpi)
        plt.close(fig)
    return [p.name for p in out.glob("*.png")]
