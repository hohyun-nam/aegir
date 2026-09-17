# 실전 입력 학습 파이프라인 (2026-09-15)

기본 학습 입력은 관측 RGB/HDR, 카메라 정보, 허용된 valid/emitter mask다. GT depth/normal/material 및 Blender 조명 bbox/두께/strength를 학습 입력으로 자동 선택하지 않는다. Conference 분할은 사용자 요청대로 기존 상태를 유지한다.

## 순서 및 기본 설정

| 단계 | 입력 → 결과 | 기본 설정 |
| --- | --- | --- |
| Depth | RGB → ZoeD_N 예측 | 미터, OpenCV axial Z-depth, GT alignment 없음 |
| Seed | 예측 depth + 카메라 + valid mask | train 선택 유지, stride4, voxel0.10m, 0.1–15m, emitter dilation2 제외 |
| Geometry view | RGB/pose + 예측 depth + seed | valid255 / invalid0 / emitter-exempt64 |
| 2DGS | 예측 depth 및 RGB 감독 | 20K, Pearson0.2, invalid0.02, 보호폭7px, dist0, densify10K/0.0005 |
| Emitters | HDR 밝은 영역 + 다중 뷰 카메라 | 삼각측량/재투영 검증, 영상 경계로 방향·면적 복원, 동일한 긴 조명의 공선 조각 병합, GT/예측-depth 조명 위치 사용 없음 |
| AEGIR | 고정 배경 + 예측 재질 priors + 관측 이미지 | 20K, SH-DC, white0, secondary-only128/1 |
| Inspect | 학습 결과 + 기존 관측 이미지 | 128/8, SH proxy off, denoiser 기본 off/비교, 알베도 비교 |

에미터 초기화 기본값은 `hdr_multiview`다. 최소3뷰(기본 train 뷰의5%)와 최대3px 재투영 오차, 최소3도 시차 및 면적/방향 복원을 요구한다. 길고 같은 선상에 있는 조명이 밝기 단절로 여러 후보가 되면 최종3D support를 병합한다. 후보가 하나도 검증되지 않으면 오류로 멈추며 임의/GT/depth 조명 fallback을 하지 않는다. 예측 depth는 배경 seed와 2DGS 감독에 사용한다. 자세한 알고리즘과 초기화 inspect 명령은 [HDR_MULTIVIEW_EMITTER_INITIALIZATION.md](HDR_MULTIVIEW_EMITTER_INITIALIZATION.md)에 있다.

예측 depth는 모델 추정치다. 카메라 translation 좌표와 미터 단위가 호환되어야 하며, Pearson loss가 scale 오류나 조명 표면의 depth 추정 오류를 자동으로 고쳐 주지는 않는다. emitter pose 고정과 초기 scale 이하 제약은 기존 사용자 설정이므로 유지했다.

## 실행

AEGIR root에서 tmux로 다음 entrypoint를 실행한다. 새 산출물은 기존 GT 실험과 다른 `preddepth_hdrmv...v2` AEGIR 기본 경로를 사용한다.

```bash
# Blender, planar 2D emitter 기본값
bash scripts/run_blender_gtdepth_full_pipeline.sh

# Blender, ellipsoid 3D emitter
EMITTER_REPRESENTATION=ellipsoid_3d bash scripts/run_blender_gtdepth_full_pipeline.sh

# Conference, ellipsoid 3D emitter 기본값
bash scripts/run_conference_depthseed_full_pipeline.sh
```

Blender 스크립트 파일명의 `gtdepth`는 과거 명칭이다. 현재는 공통 `run_predicted_depth_full_pipeline.sh`로 연결하며 source dataset의 depth 파일을 모델 감독으로 연결하지 않는다. source의 기존 `*_intrinsics_depthfixed_v2` 이름 역시 유지한 카메라 보정 dataset 명칭이다.

사용자 경로는 `SOURCE_DATA`, `PREDICTED_DATA`, `GEOMETRY_DATA`, `SEED_PLY`, `DN_OUTPUT`, `DN_LOG`, `AEGIR_OUTPUT`, `PRIORS`로 변경할 수 있다. `DATA`는 Blender 호환 entrypoint에서 source 별칭이다. `GT_DATA` 같은 과거 중간 데이터 경로 변수는 새 기본 경로에 사용하지 않는다.

로그: `AEGIR_OUTPUT/full_pipeline.log`, 단계별 native 로그: `DN_LOG/<scene>.log`, 완료 시 timing: `AEGIR_OUTPUT/full_pipeline_timing.tsv`.

## 출처 및 재사용

- `PREDICTED_DATA/predicted_depth/manifest.json`: RGB/depth checksum, metadata signature, generator/model weights hash, 단위/convention 및 GT alignment=false.
- `SEED_PLY.provenance.json`: 출처와 voxel/stride/mask/split 설정 및 PLY checksum.
- `GEOMETRY_DATA/geometry_view_provenance.json`: prepared mask/depth/seed 입력과 출력 checksum.
- `<checkpoint>.provenance.json`: 실행 설정, 입력 및 코드 hash와 완료 checkpoint checksum.
- `<checkpoint>.pending.json`: 진행/중단 실행의 요청. 같은 입력/설정일 때만 이어간다.
- `AEGIR_OUTPUT/emitter_initialization.json`: 초기화 방법/입력 뷰와 조명별 위치·방향·크기/대응 뷰/재투영 오차. checkpoint에 진단 내용을 함께 저장한다.
- AEGIR checkpoint 내부 `dataset.input_provenance`: 데이터/geometry/prior checksum, exact-resume 검증에 사용.

검증되지 않은 기존 checkpoint와 raw/mm Zoe 디렉터리는 자동 실전 캐시로 받아들이지 않는다. 충돌하면 새 output path로 실행한다. GT 도움을 받은 과거 결과는 그대로 남으며 실전 모델을 얻으려면 depth/2DGS부터 재학습해야 한다. GT 진단 opt-in은 changelog에 설명되어 있으며 기본 풀코스는 이 옵션을 켜지 않는다.

Conference inspect는 기존처럼 학습뷰 reconstruction이다. held-out 평가 분할은 이번 변경에 포함하지 않았다.

## 이번 수정의 검증

- 전체 테스트 **137개 통과**. GT/raw-depth 위장 연결, GT seed/geometry 유입, cache 설정/내용 변경, prepared mask 변조, HDR 후보 위치 복원 및 초기 radiance 보상 회귀 검증 포함.
- 실제 Conference frame 0/33으로 cached ZoeD_N GPU 추론 실행. frame 0 median2.728m, frame33 median3.757m.
- frame33의 1-view smoke: emitter exclusion seed4,143점 → native2DGS1-step 및 checkpoint export → 기본 HDR/예측-depth emitter 초기화 → frozen AEGIR1-step 성공.
- 동일 smoke checkpoint를128/8로 기본/denoised inspect하고 비교 이미지 및 RGB/prior/학습 albedo 비교를 생성했다. SH proxy off, 기본 denoiser off를 metrics에서 확인했다.
- Shell syntax 및 변경 Python 파일 Ruff 검사 통과.

Smoke 결과는 `outputs/practical_pipeline_smoke_20260915/`에 있다. **1-step smoke는 경로 연결과 실행 검증이며 수렴/화질 평가가 아니다.** 전체191-view/20K 재학습은 이번 수정 과정에서 시작하지 않았다.

## 과거 HDR/예측-depth 반경 방식 (현재 기본값 아님)

**이 방식은 후보 부족만 해결했으며 실제 Blender에서 조명 위치가 잘못 복원되어 `hdr_multiview`로 대체됐다.** 명시적 `hdr_predicted_depth` 초기화는 동일 voxel ID 대신 **scene diagonal의1% 반경**에서 서로 다른 뷰의 고휘도 후보를 센다. 뷰당1표이며 minimum view 조건(기본 max3/5%)은 유지한다. DBSCAN에 투입하는 후보와 최종 fitted cluster 모두 다중 뷰 증거를 요구한다. `EMITTER_PERSISTENCE_RADIUS_FRACTION=0.01`로 runner에서 설정할 수 있다. Generic G-buffer 진단의 기존 voxel 모드는 유지한다.

Blender120-view 실제 초기화는8cm 동일 셀 검사에서 최대5뷰로6뷰 조건을 충족하지 못했지만,16cm 반경 검사에서 GT 없이6개 에미터를 검출했다. 완료된 native2DGS는 그대로 사용하고 AEGIR만 새 output에서 실행한다. 모델 depth의 실제 위치 오차는 별도 확인해야 한다.


## HDR 다중 뷰 초기화로 전환

실제 Blender train120뷰의 관측 HDR/카메라만으로 에미터6개를 복원했다. 대응은 조명당28–47뷰, 재투영 오차 중앙값0.29–0.48px다. 전체 테스트145개 통과. 자세한 진단/overlay/PLY는 `outputs/blender_practical3d_20260915_051804/emitter_init_hdr_multiview_v2/`에 있다. 검증된 실전2DGS는 재사용할 수 있으며 AEGIR은 새 출력에서 처음부터 학습한다. 예전 emitter 초기화의 학습 결과는 resume로 교정되지 않는다.
