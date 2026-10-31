# 코드 읽는 순서와 실행

[프로젝트 개요](README.md) / [설정](configs/README.md) / [입력 자료](data/README.md)

대표 결과에 쓰인 다섯 관측 변수, Gaussian HMM, MACD 국면 전략과 롱 보유 비교만 남김. 전체 연구 원본과 기존 결과는 별도로 보존함.

## 면접 준비용 읽는 순서

아래 네 파일이 연구 내용을 설명하는 핵심임.

| 순서 | 파일 | 설명할 내용 |
| --- | --- | --- |
| 1 | [data.py](src/trading/data.py) | 가격·펀딩 검증, 다섯 관측 변수와 자료 이용 가능 시각 |
| 2 | [regime.py](src/trading/regime.py) | 과거 자료로 HMM 학습, 상태 이름, 매일 국면 확률과 10일 평균 |
| 3 | [strategy.py](src/trading/strategy.py) | MACD와 SMA 신호를 국면에 따라 적용하는 규칙 |
| 4 | [backtest.py](src/trading/backtest.py) | 다음 시가 체결, 수량·비용·펀딩, 고갈 거부와 성과 지표 |

[run.py](src/trading/run.py)는 TOML 설정을 읽어 위 네 단계를 연결하고 보고서를 저장함. [cache.py](src/trading/cache.py)는 학습·확률·손익의 체크섬과 안전한 저장을 담당함.

설명할 때는 다음 질문에 답할 수 있으면 됨.

- 왜 HMM을 썼는가? 관측 변수의 분포가 다른 세 국면을 추정하고 국면별 규칙을 비교하기 위함임. 국면 이름 자체가 미래 수익을 보장하지 않음.
- 미래 자료 사용을 어떻게 막았는가? 학습·정규화는 이전 관측만 사용하고, 오늘까지의 확률로 신호를 만든 뒤 다음 시가에 체결함.
- 10일 평균은 무엇인가? 오늘을 포함한 최근 10일의 국면 확률을 각각 평균내어 가장 큰 국면을 선택함. 판단 흔들림과 반응 지연의 절충임.
- 결과의 한계는 무엇인가? 규칙을 전체 기간 성과로 골랐고, 실제 유지증거금과 강제청산을 모델링하지 않았음.

## 실행

프로젝트 폴더에서 Python 3.12 환경을 준비하고 [입력 CSV](data/README.md)를 설정 경로에 둠.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\run-experiment.ps1 -Config configs/btc.toml -CheckOnly
.\run-experiment.ps1 -Config configs/btc.toml
```

다른 운영체제에서는 가상환경에 설치한 뒤 `btc-research --config configs/btc.toml --check`로 검사하고, `--check`를 빼고 실행함. `CheckOnly`도 입력 파일은 필요하며 학습은 하지 않음.

```powershell
btc-research --verify-results
python -m unittest discover -s tests -v
```

검산 명령은 저장된 15개 자산곡선에서 수익률·Sharpe·최대낙폭을 계산하므로 원시 자료가 필요하지 않음. 테스트는 [test_core.py](tests/test_core.py) 한 파일에 모음.

학습, 일별 확률과 손익 캐시는 각각 분리함. 비용이나 평균 기간을 바꿀 때 필요한 단계만 다시 계산함. 결과와 실패·미수렴 기록은 `reports/experiments/`의 실행별 폴더에 저장함. 라이브러리 버전과 입력 자료가 달라지면 새 실행 결과도 달라질 수 있음.
