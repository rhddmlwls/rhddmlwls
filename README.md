# rhddmlwls

Quantitative Research

## Research

### [Bitcoin Auto Trade Research](https://github.com/rhddmlwls/rhddmlwls/tree/main/research/bitcoin-auto-trade)

BTCUSDT 데이터 수집부터 HMM 국면 분석, 거래비용과 펀딩을 반영한 백테스트, 결과 검증까지 연결한 Python 연구 파이프라인을 구축함.

상승장과 하락장에 같은 매매 규칙을 적용하는 대신, 시장을 상승, 횡보, 하락으로 나누고 국면마다 다른 전략을 쓰는 방식을 실험함. 모델의 국면 구분이 그럴듯한지와 실제 비용 차감 손익이 좋은지를 따로 평가함.

#### 구현한 내용

- 데이터: Binance USD-M 선물 일봉 2,465일과 펀딩 이벤트 7,395건 수집. 가격, 펀딩과 파생지표 원본 ZIP 2,383개 체크섬과 결측, 중복 검사
- 국면 분석: 과거 365봉으로 학습하고 30봉마다 재학습. 각 종가까지의 정보로만 HMM 상태와 확률 추정
- 백테스트: 국면별 롱, 현금, EMA, Momentum 배치. 다음 시가 체결, 수수료, 슬리피지, 실제 펀딩과 반전 비용 반영
- 실험 관리: TOML로 관측 변수와 모델, 전략을 바꾸고 학습, 확률, 매매, 그림 캐시를 분리
- 검증: Top 3 두 seed 재현, 원장 24개 CSV의 SHA256 일치 확인. 당시 작은 파이프라인 통합시험을 포함한 117개 테스트 통과

#### 주요 결과

관측 변수와 모델, 전환 방식, 전략 배치를 비교하고, 동일 조건의 417개 유효 설정을 291개 전략군으로 묶어 Top 3를 정리함. 두 고정 seed 42/137 중 낮은 Sharpe 순으로 선정함.

| 전략군 | 상승 / 횡보 / 하락 | 최저 seed Sharpe | 순수익, seed 42 / 137 |
| --- | --- | ---: | ---: |
| 펀딩 5변수 HMM | 롱 / EMA / 현금 | 1.031 | +289.28% / +315.49% |
| 60일 3변수 HMM | EMA / EMA / 현금 | 0.787 | +199.02% / +162.45% |
| 120일 3변수 HMM | 모멘텀90 / EMA / 현금 | 0.776 | +209.39% / +168.46% |

평가기간은 2021-04-30~2026-09-30 UTC, 초기자산 10,000 USDT, 편도 수수료와 슬리피지 각각 5 bps, 실제 펀딩과 최종 청산 비용을 반영한 결과임.

전체 기간을 살펴본 뒤 고른 사후 탐색 순위임. 1위의 2025년 이후 Sharpe는 0.106 / 0.400으로 약했고, 해당 후보의 선택 학습은 1/66회와 4/66회 미수렴함. 미사용 테스트 성과는 아님.

#### 실제로 확인한 차이

상승 롱 / 횡보 EMA / 하락 현금 규칙을 고정하고 기존 3변수에 변동성과 펀딩을 추가한 짝 비교에서, 두 seed의 전체 기간 Sharpe가 1.003 / 0.715에서 1.031 / 1.082로 높아짐. 다만 최근 구간 성과는 기존 3변수보다 약했음.

아래는 같은 가격과 날짜에 3변수와 5변수 HMM이 붙인 국면을 비교한 예시임. 검은 선은 BTC 종가, 초록은 상승, 회색은 중간 상태, 빨강은 하락임. 색은 각 종가에서 추정한 상대적인 상태이며, 미래 가격의 정답 라벨은 아님.

![3변수와 5변수 HMM의 두 seed 국면 비교](https://raw.githubusercontent.com/rhddmlwls/rhddmlwls/main/research/bitcoin-auto-trade/results/images/observation_regime_zoom.png)

[구현과 분석 흐름](https://github.com/rhddmlwls/rhddmlwls/tree/main/research/bitcoin-auto-trade) / [실험별 비교와 해석](https://github.com/rhddmlwls/rhddmlwls/blob/main/research/bitcoin-auto-trade/EXPERIMENTS.md) / [전체 성과와 낙폭](https://github.com/rhddmlwls/rhddmlwls/tree/main/research/bitcoin-auto-trade/results)

## 다음 작업

오프라인 데이터 수집, 국면 분석과 백테스트까지 구현한 상태임. 다음 단계는 기존 일봉 신호를 실제 호가 환경에서 어떻게 실행할지 확인하는 작업임.

- [ ] 오더북 수집 범위와 체결 평가 기준 결정
- [ ] 스냅샷과 증분 이벤트로 호가 복원, 저장과 재생 구현
- [ ] 가상 주문, 부분 체결, 포지션과 자산을 대사하는 paper trading 연결

오더북과 paper 실행은 구현 전임. 단계별 완료 기준은 [로드맵](https://github.com/rhddmlwls/rhddmlwls/blob/main/research/bitcoin-auto-trade/ROADMAP.md)에 정리함.

## Tools

Python 3.12, pandas, NumPy, scikit-learn, hmmlearn, Matplotlib, uv

실험과 검증 수치는 2026-10-20~22 연구 기록 기준임.
