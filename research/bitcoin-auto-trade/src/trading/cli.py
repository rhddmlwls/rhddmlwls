"""Config-driven entry point for the published fixed-strategy HMM core."""
import argparse
from .pipeline import run_experiment


def main():
    parser=argparse.ArgumentParser(description="Offline HMM regime/strategy research; no orders")
    commands=parser.add_subparsers(dest="command",required=True)
    experiment=commands.add_parser("experiment",help="Validate, train and backtest a TOML config")
    experiment.add_argument("--config",default="configs/btc.toml")
    experiment.add_argument("--check",action="store_true",help="Validate source inputs without model training")
    args=parser.parse_args()
    try:
        report=run_experiment(args.config,check=args.check)
    except (ValueError,TypeError,FileNotFoundError) as exc:
        parser.error(str(exc))
    if report.get("backtest_rejections"):
        raise SystemExit(2)


if __name__=="__main__":
    main()
