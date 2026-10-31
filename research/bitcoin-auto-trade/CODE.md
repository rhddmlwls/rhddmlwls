# 코드 구성과 실행

[프로젝트 개요](README.md) / [설정](configs/README.md) / [입력 자료](data/README.md)

다섯 관측 변수, Gaussian HMM, MACD 국면 전략과 롱 보유 비교를 구현함.

## 코드 구성

| 파일 | 역할 |
| --- | --- |
| [data.py](src/trading/data.py) | 가격·펀딩 검증과 관측 변수 계산 |
| [regime.py](src/trading/regime.py) | HMM 학습, 일별 국면 확률과 후행 평균 |
| [strategy.py](src/trading/strategy.py) | MACD 신호와 국면별 매매 규칙 |
| [backtest.py](src/trading/backtest.py) | 다음 시가 체결, 비용·펀딩과 성과 계산 |
| [run.py](src/trading/run.py) | TOML 설정 검사와 전체 실행, 보고서 저장 |
| [cache.py](src/trading/cache.py) | 학습·확률·손익 캐시와 체크섬 검증 |

## 실행

프로젝트 폴더에서 Python 3.12 환경을 준비하고 [입력 CSV](data/README.md)를 설정 경로에 둠.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\run-experiment.ps1 -Config configs/btc.toml -CheckOnly
.\run-experiment.ps1 -Config configs/btc.toml
```

다른 운영체제에서는 가상환경에 설치하고 활성화한 뒤 `btc-research --config configs/btc.toml --check`로 검사하고, `--check`를 빼고 실행함. `CheckOnly`도 입력 파일은 필요하며 학습은 하지 않음.

```powershell
.\.venv\Scripts\python.exe -m trading.run --verify-results
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

검산 명령은 저장된 15개 자산곡선에서 수익률·Sharpe·최대낙폭을 계산하므로 원시 자료가 필요하지 않음. 테스트는 [test_core.py](tests/test_core.py) 한 파일에 모음.

학습, 일별 확률과 손익 캐시는 각각 분리함. 비용이나 평균 기간을 바꿀 때 필요한 단계만 다시 계산함. 결과와 실패·미수렴 기록은 `reports/experiments/`의 실행별 폴더에 저장함. 라이브러리 버전과 입력 자료가 달라지면 새 실행 결과도 달라질 수 있음.
