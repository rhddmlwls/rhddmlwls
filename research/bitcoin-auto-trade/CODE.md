# 핵심 코드와 실행

[프로젝트 개요](README.md) / [비교 설정](configs/README.md) / [입력 자료](data/README.md)

대표 전략의 계산 경로를 공개함. 원래 연구 코드의 계산과 캐시 구조를 유지하고, 전체 후보 탐색과 부가 진단 명령은 제외함.

| 흐름 | 코드 |
| --- | --- |
| 일봉과 펀딩 입력 검증 | [data.py](src/trading/data.py) |
| 관측 변수 계산 | [feature_engine.py](src/trading/feature_engine.py), [features.py](src/trading/features.py) |
| 과거 자료로 HMM 학습과 일별 확률 산출 | [hmm.py](src/trading/hmm.py), [posterior.py](src/trading/posterior.py) |
| 국면 확률의 후행 평균과 국면 판단 | [decisions.py](src/trading/decisions.py) |
| 국면별 신호와 목표 포지션 | [technical_strategies.py](src/trading/technical_strategies.py), [strategy_targets.py](src/trading/strategy_targets.py) |
| 다음 시가 체결, 비용과 펀딩 | [directional.py](src/trading/directional.py) |
| 설정 검사와 실행, 캐시와 성과 | [experiment_config.py](src/trading/experiment_config.py), [pipeline.py](src/trading/pipeline.py), [research_cache.py](src/trading/research_cache.py), [portfolio_metrics.py](src/trading/portfolio_metrics.py) |

## 실행

프로젝트 폴더에서 Python 3.12 환경을 사용함.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[plots]"
.\run-experiment.ps1 -Config configs/btc.toml -CheckOnly
.\run-experiment.ps1 -Config configs/btc.toml
```

`CheckOnly`도 실제 입력 파일이 필요함. [자료 형식](data/README.md)에 맞춰 CSV를 준비하고 TOML의 `[paths].data_dir`을 지정함. ETH 예시는 `configs/transfer.toml`이며, 다른 코인은 심볼과 입력 폴더를 변경함.

다른 운영체제에서는 Python 3.12 가상환경에 설치한 뒤 `btc-research experiment --config configs/btc.toml --check`로 검사하고, `--check`를 빼고 실행함.

학습은 과거 관측만 사용함. 종가 국면과 신호로 다음 시가에 매매하며, 실제 펀딩, 반전의 양쪽 체결 비용과 최종 청산을 반영함. 결과와 실패·미수렴 기록은 `[paths].output_root` 아래 새 실행 폴더에 저장함. 캐시는 학습, 일별 확률과 손익별로 구분함.

## 저장된 결과 검산

```powershell
python verify_results.py
```

이 명령은 공개 CSV에서 성과 지표만 다시 계산하므로 원시 자료와 학습이 필요하지 않음. 저장된 결과와 새 실행 결과는 구분함. 원시 자료의 기간, 입력값이나 라이브러리 버전이 달라지면 새 결과가 달라질 수 있음.

핵심 단위 테스트와 작은 파이프라인 검사는 `python -m unittest discover -s tests -v`로 실행함.
