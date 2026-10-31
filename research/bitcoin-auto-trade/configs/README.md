# 설정 예시

[프로젝트 개요](../README.md) / [검산 데이터](../results/README.md)

공통 조건은 두 예시에 모음. 원래 실행기에 사용한 설정을 그대로 보관한 것이며, HMM 재학습 실행기는 이번 공개 범위에 포함하지 않음.

| 설정 | 평가 조건 |
| --- | --- |
| [btc.toml](btc.toml) | BTC 사후 조합과 롱 보유, 2021-05-09~2026-09-30, seed 42 |
| [transfer.toml](transfer.toml) | 공통 기간 비교의 ETH 예시, 2022-03-01~2026-09-30, seed 42 / 137 |

다른 코인의 공통 기간 실험도 같은 모델과 매매 조건을 사용함. 원본 설정 간 차이는 실험 이름, 아래 심볼과 입력 폴더임.

| 자산 | `[data].symbol` | `[paths].data_dir` |
| --- | --- | --- |
| BTC | `BTCUSDT` | `../data/cross_asset/BTCUSDT` |
| ETH | `ETHUSDT` | `../data/cross_asset/ETHUSDT` |
| SOL | `SOLUSDT` | `../data/cross_asset/SOLUSDT` |
| XRP | `XRPUSDT` | `../data/cross_asset/XRPUSDT` |

직전 365개 관측 학습, 30일 재학습, 10일 후행 확률 평균, 다음 시가 체결과 진입 후 수량 고정을 유지함. 수수료와 슬리피지는 체결당 각각 0.05%이며 실제 펀딩과 마지막 종가 청산을 반영함. 데이터와 보고서 경로는 원래 로컬 실험 환경을 가리킴.
