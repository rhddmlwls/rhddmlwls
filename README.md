# rhddmlwls

Quantitative Research · Bitcoin · Python

비트코인 시장의 국면을 모델링하고, 매매 아이디어를 재현 가능한 실험으로 검증합니다.
연구 질문부터 결과와 한계, 다음 작업까지 기록하는 공간입니다.

---

## Current Focus

- **Market Regimes** — HMM으로 상승·횡보·하락 국면을 나누고 관측 변수와 상태 지속시간 비교
- **Strategy Research** — 국면별 전략과 거래비용·펀딩·초기값에 따른 성과 차이 검증
- **Reproducible Experiments** — TOML 설정, 단계별 캐시, 결과 원장과 감사 기록으로 실험 관리
- **Next Research** — 오더북 수집·재생과 paper trading 연결 준비

---

## Featured Research

| Project | Question | Methods | Status |
| --- | --- | --- | --- |
| [Bitcoin Auto Trade Research](https://github.com/rhddmlwls/rhddmlwls/tree/main/research/bitcoin-auto-trade) | 시장 국면에 따라 전략을 바꾸면 어떤 차이가 생길까? | HMM · Sticky HMM · HSMM · EMA · Momentum · Backtesting | 오프라인 연구 완료 · 후속 구현 예정 |

BTCUSDT 일봉·펀딩으로 국면 모델과 전략을 비교하고, 동일 조건의 **291개 전략군에서 Top 3**를 정리했습니다. 두 seed의 결과, 비용, 최근 구간 성과와 미수렴 기록을 함께 공개합니다.

[연구 개요](https://github.com/rhddmlwls/rhddmlwls/tree/main/research/bitcoin-auto-trade) · [결과와 한계](https://github.com/rhddmlwls/rhddmlwls/tree/main/research/bitcoin-auto-trade/results) · [다음 작업](https://github.com/rhddmlwls/rhddmlwls/blob/main/research/bitcoin-auto-trade/ROADMAP.md)

---

## Current Status

| Stage | Status | Next Action |
| --- | --- | --- |
| 일봉·펀딩 데이터와 HMM 전략 비교 | 완료 | 기존 결과를 후속 연구의 기준선으로 보존 |
| Top 3 결과·설정·검증 기록 공개 | 완료 | 새 연구가 생기면 같은 형식으로 추가 |
| 오더북 수집·복원·재생 | 예정 | 일봉 전략의 체결 개선과 별도 호가 신호 중 연구 목적 정의 |
| Paper trading | 예정 | 재생 경로 검증 후 가상 주문·체결·포지션 대사 연결 |

실험·검증 기준일은 **2026-10-22**, 이 프로필 정리일은 **2026-10-30**입니다. 현재 구현 범위는 오프라인 연구이며, 실시간 주문 연결은 예정 단계입니다.

---

## Recent Updates

| Date | Update |
| --- | --- |
| 2026-10-30 | 연구 포트폴리오와 진행 현황·업데이트 템플릿 정리 |
| 2026-10-22 | 3→5변수 HMM 비교, Top 3 두 seed 재현과 검증 기록 정리 |
| 2026-10-21 | 국면 지속시간·기술적 전략 조합·거래비용·하락 라벨 점검 |
| 2026-10-20 | 장기 일봉·펀딩 자료 확보와 초기 HMM 검증 |

[전체 업데이트 기록](https://github.com/rhddmlwls/rhddmlwls/blob/main/UPDATES.md)

---

## Research Toolkit

**Python:** pandas · NumPy · scikit-learn · hmmlearn · Matplotlib  
**Research:** 시계열 · 시장 국면 · 백테스트 · 비용·펀딩 분석 · seed 민감도  
**Workflow:** uv · TOML · unittest · GitHub · 체크섬·원장 대사

---

## Notes & Templates

[프로젝트 템플릿](https://github.com/rhddmlwls/rhddmlwls/blob/main/templates/PROJECT.md) · [업데이트 템플릿](https://github.com/rhddmlwls/rhddmlwls/blob/main/templates/UPDATE.md) · [관리 방법](https://github.com/rhddmlwls/rhddmlwls/blob/main/MAINTENANCE.md)

연구 결과는 과거 데이터를 반복해서 살펴본 사후 탐색입니다. 기간·비용·seed와 한계를 함께 기록하며, 미사용 테스트 성과와 구분합니다.
