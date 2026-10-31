# 입력 자료

[프로젝트 개요](../README.md) / [코드 실행](../CODE.md)

원시 시장 자료는 저장소에 포함하지 않음. TOML의 `[paths].data_dir` 폴더에 아래 세 CSV를 준비함. 상대경로는 설정 파일 위치를 기준으로 함. `<SYMBOL>`은 `BTCUSDT`, `ETHUSDT`, `SOLUSDT`, `XRPUSDT` 중 실행할 심볼임.

| 파일 | 필수 열 |
| --- | --- |
| `<SYMBOL>_1d.csv` | `datetime,open,high,low,close,volume,taker_buy_volume` |
| `<SYMBOL>_funding_events.csv` | `datetime,last_funding_rate,funding_interval_hours` |
| `<SYMBOL>_funding_daily.csv` | `datetime,funding_mean,available_at` |

- 날짜와 시각은 UTC ISO 8601을 권장함. 일봉의 `datetime`은 해당 일자 00:00 UTC임. 중복·결측 일자와 미완료 봉을 허용하지 않음.
- `taker_buy_volume`은 기본자산 수량 단위이며 `volume`과 단위가 같아야 함.
- 펀딩 이벤트는 실제 부호와 정산 시각, 당시 정산 간격을 유지함. 일괄 8시간으로 가정하지 않음. `mark_price`를 추가할 수 있으며, 없으면 해당 UTC 일자 시가로 펀딩 금액을 계산함.
- 일별 `funding_mean`은 그 UTC 일자의 실제 펀딩률 산술평균임. `available_at`은 그 값이 관측 가능해진 시각임. 원래 수집기는 다음 날 00:05 UTC에 이용 가능하다고 가정함. 단순 날짜 결합이나 결측 보간을 하지 않음.

## 원본 출처와 기간

Binance USDⓈ-M 무기한선물의 공식 월별 일봉과 펀딩 아카이브를 사용함. ZIP을 풀어 위 CSV 형식으로 변환하고, 일별 펀딩 평균과 이용 가능 시각을 별도로 만듦.

- [일봉 원본 예시](https://data.binance.vision/data/futures/um/monthly/klines/ETHUSDT/1d/ETHUSDT-1d-2020-10.zip)
- [펀딩 원본 예시](https://data.binance.vision/data/futures/um/monthly/fundingRate/ETHUSDT/ETHUSDT-fundingRate-2026-09.zip)

BTC 대표 실행의 원시 일봉은 2020-01-01~2026-09-30, 코인별 공통 기간 실행의 원시 일봉은 2020-10-01~2026-09-30임. 앞부분은 지표 계산과 HMM 학습에 필요하므로 평가 시작일보다 이전 자료까지 준비해야 함. 원래 수집에서는 아카이브 체크섬을 확인하고, 월별 파일의 누락 봉은 공식 일별 파일에서 복구했음.
