# Top 3 설정

2026-10-22에 재현한 세 TOML의 원본 사본임.

| 설정 | 전략군 |
| --- | --- |
| [top3_01_funding.toml](top3_01_funding.toml) | 펀딩 5변수 / 롱, EMA, 현금 |
| [top3_02_ema60.toml](top3_02_ema60.toml) | 60일 3변수 / EMA, EMA, 현금 |
| [top3_03_momentum120.toml](top3_03_momentum120.toml) | 120일 3변수 / 모멘텀90, EMA, 현금 |

입력 경로 `../data/extended` 등은 원래 연구 프로젝트의 디렉터리 기준임. 이 프로필 저장소에는 실행 코드와 데이터가 없으므로 TOML만으로 학습과 재현 실행은 불가함.

원래 프로젝트의 표준 명령은 `./run-experiment.ps1 -Config configs/long_regimes.toml`임. 설정 검사만 할 때는 `-CheckOnly`를 추가함. 개별 후보는 해당 Top 3 설정을 지정함. 과거 실행 결과, 해시, 원장 대사는 [재현 기록](../results/reproduction.json)에서 확인 가능함.

일반 조건은 TOML에서 먼저 변경함. 학습, 확률, 매매, 그림 캐시를 분리하는 기존 연구 흐름을 유지함. 게시 사본의 해시는 [PUBLICATION.json](../PUBLICATION.json)에 기록함.
