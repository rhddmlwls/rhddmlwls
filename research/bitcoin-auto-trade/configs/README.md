# Top 3 Configuration Snapshots

2026-10-22에 재현한 세 TOML을 원본 바이트 그대로 보관합니다.

| 설정 | 전략군 |
| --- | --- |
| [top3_01_funding.toml](top3_01_funding.toml) | 펀딩 5변수 / 롱·EMA·현금 |
| [top3_02_ema60.toml](top3_02_ema60.toml) | 60일 3변수 / EMA·EMA·현금 |
| [top3_03_momentum120.toml](top3_03_momentum120.toml) | 120일 3변수 / 모멘텀90·EMA·현금 |

입력 경로 `../data/extended` 등은 원래 연구 프로젝트의 디렉터리 기준입니다. 실행 코드와 데이터는 이 프로필 저장소에 포함되지 않으므로, 여기에서 TOML만으로 학습·재현을 실행할 수는 없습니다.

원래 프로젝트의 표준 실행은 `./run-experiment.ps1 -Config configs/long_regimes.toml`, 설정 검사만 할 때는 `-CheckOnly`를 붙입니다. 개별 후보는 해당 Top 3 설정을 지정합니다. 과거 실행 결과·해시·원장 대사는 [재현 기록](../results/reproduction.json)에 있습니다.

일반 조건 변경은 TOML을 먼저 수정하고 학습·확률·매매·그림 캐시를 분리하는 기존 연구 흐름을 유지합니다. 게시 사본의 해시는 [PUBLICATION.json](../PUBLICATION.json)에서 확인합니다.
