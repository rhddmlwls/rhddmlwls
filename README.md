# rhddmlwls

Quantitative Research

## Research

| 프로젝트 | 내용 |
| --- | --- |
| [Bitcoin Auto Trade](https://github.com/rhddmlwls/rhddmlwls/tree/main/research/bitcoin-auto-trade) | HMM 국면 분석과 국면별 매매 전략 비교 |

두 seed의 전체 성과와 최근 성과, 미수렴 기록은 [상세 결과](https://github.com/rhddmlwls/rhddmlwls/tree/main/research/bitcoin-auto-trade/results)에 정리함. 전체 기간을 본 뒤 고른 사후 탐색 결과이며, 미사용 테스트 성과는 아님.

현재 구현 범위는 데이터 수집, 국면 분석, 백테스트까지임.

## 다음 작업

- [ ] 오더북 데이터의 용도와 수집 범위 결정
- [ ] 스냅샷과 증분 이벤트로 호가 복원, 저장과 재생 구현
- [ ] 가상 주문과 체결을 연결하고 포지션과 자산 대사

오더북과 paper trading은 구현 전임. 단계별 완료 기준은 [로드맵](https://github.com/rhddmlwls/rhddmlwls/blob/main/research/bitcoin-auto-trade/ROADMAP.md)에 정리함.

## Tools

Python, pandas, NumPy, scikit-learn, hmmlearn, Matplotlib

TOML 설정 기반 실험, uv 환경 관리, unittest 검증
