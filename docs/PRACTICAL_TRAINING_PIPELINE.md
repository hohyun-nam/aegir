# 실전 입력 학습 파이프라인 (2026-09-17)

기본 학습 입력은 관측 RGB/HDR, 카메라 정보와 valid mask다. GT
depth/normal/material 및 Blender 조명 ID/bbox/두께/strength는 사용하지 않는다.
source에 emitter mask가 있어도 새 기본 파이프라인의 2DGS, emitter 초기화,
AEGIR loss 분리는 이를 읽지 않는다. Conference 분할은 기존 상태를 유지한다.

## 순서 및 기본 설정

| 단계 | 입력 → 결과 | 기본 설정 |
| --- | --- | --- |
| Depth | RGB → ZoeD_N 예측 | 미터, OpenCV axial Z-depth, GT alignment 없음 |
| Seed | 예측 depth + 카메라 + valid mask | train 선택 유지, stride4, voxel0.10m, 0.1–15m, 에미터 포함 |
| Geometry view | RGB/pose + 예측 depth + seed | 에미터를 RGB/depth supervision에 포함 |
| 2DGS | 예측 depth 및 RGB 감독 | 20K, Pearson0.2, invalid0.02, 보호폭7px, dist0, densify10K/0.0005 |
| Emitters | HDR + inclusive 2DGS G-buffer | 상위0.5% 고휘도점, multi-view persistence, DBSCAN, PCA; GT emitter 정보 미사용 |
| 역할 분리 | PCA support + source Gaussian center | Gaussian별 `emitter_id`; 카메라 BRDF raster와 IRGS BVH에서 tagged row 제외 |
| AEGIR | 고정 배경 + 예측 재질 priors + 관측 이미지 | 5K, SH-DC, white0, secondary-only128/1 |
| Inspect | 학습 결과 + 기존 관측 이미지 | 128/8, SH proxy off, denoiser 기본 off/비교, 알베도 비교 |

에미터 초기화 기본값은 풀코스에서 `hdr_gbuffer`다. HDR 밝은 픽셀 위치는
emitter-inclusive 2DGS의 expected surface point로 얻는다. 전체 train 뷰의 기본 5%
이상에서 같은 3D voxel에 반복된 후보만 DBSCAN에 넣고, cluster별 PCA로 center,
scale, rotation을 만든다. 후보가 없으면 임의 emitter로 대체하지 않고 오류로 멈춘다.

초기화 직후 analytic support 내부의 source Gaussian에 emitter ID를 저장한다. 이
Gaussian들은 삭제되지 않아 association을 검사하거나 다시 분류할 수 있지만, 일반
BRDF G-buffer와 visibility/secondary transport BVH에는 들어가지 않는다. 따라서
analytic `L_e`와 source SH/BRDF가 같은 위치에서 중복 합산되거나 서로 가리는 현상을
막는다. emitter subset relighting에서는 켠 ID만 제외하므로 끈 조명의 tagged Gaussian은
비발광 fixture BRDF로 다시 나타난다. checkpoint 형식은 v16이다.

예측 depth는 모델 추정치다. 카메라 translation 좌표와 미터 단위가 호환되어야 하며, Pearson loss가 scale 오류나 조명 표면의 depth 추정 오류를 자동으로 고쳐 주지는 않는다. emitter pose 고정과 초기 scale 이하 제약은 기존 사용자 설정이므로 유지했다.

## 실행

AEGIR root에서 tmux로 다음 entrypoint를 실행한다. 새 산출물은 경로에
`emitter_inclusive`, `hdrgbuffer`, `5k`를 포함한다.

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

풀코스의 주요 확인 산출물은 다음과 같다.

- `AEGIR_OUTPUT/emitter_initialization/2dgs_emitter_inclusive.ply`: emitter fixture까지
  포함한 native 2DGS 원본.
- `AEGIR_OUTPUT/aegir_initialization.pt`: HDR G-buffer에서 emitter를 복원하고 source
  Gaussian에 ID를 연결한 직후의 step 0 checkpoint. 초기화 PLY를 만든 다음 optimizer와
  RNG까지 exact-resume하여 학습하므로 초기화를 두 번 계산하지 않는다.
- `AEGIR_OUTPUT/emitter_initialization/map_with_initial_emitters.ply`: tagged source row를
  뺀 배경과 초록색 analytic emitter를 합친 초기화 확인용 PLY.
- `AEGIR_OUTPUT/inspect_initialization_vs_5k/`: 동일 10개 pose의 GT, 초기화, 5K render
  비교 frame/contact sheet와 `summary.json`.

## 출처 및 재사용

- `PREDICTED_DATA/predicted_depth/manifest.json`: RGB/depth checksum, metadata signature, generator/model weights hash, 단위/convention 및 GT alignment=false.
- `SEED_PLY.provenance.json`: 출처와 voxel/stride/mask/split 설정 및 PLY checksum.
- `GEOMETRY_DATA/geometry_view_provenance.json`: prepared mask/depth/seed 입력과 출력 checksum.
- `<checkpoint>.provenance.json`: 실행 설정, 입력 및 코드 hash와 완료 checkpoint checksum.
- `<checkpoint>.pending.json`: 진행/중단 실행의 요청. 같은 입력/설정일 때만 이어간다.
- `AEGIR_OUTPUT/emitter_initialization.json`: 초기화 방법/입력 뷰와 조명별 위치·방향·크기/대응 뷰/재투영 오차. checkpoint에 진단 내용을 함께 저장한다.
- `AEGIR_OUTPUT/emitter_gaussian_association.json`: emitter별 tagged Gaussian 수와 support margin.
- AEGIR checkpoint 내부 `dataset.input_provenance`: 데이터/geometry/prior checksum, exact-resume 검증에 사용.

검증되지 않은 기존 checkpoint와 raw/mm Zoe 디렉터리는 자동 실전 캐시로 받아들이지 않는다. 충돌하면 새 output path로 실행한다. GT 도움을 받은 과거 결과는 그대로 남으며 실전 모델을 얻으려면 depth/2DGS부터 재학습해야 한다. GT 진단 opt-in은 changelog에 설명되어 있으며 기본 풀코스는 이 옵션을 켜지 않는다.

Conference inspect는 기존처럼 학습뷰 reconstruction이다. held-out 평가 분할은 이번 변경에 포함하지 않았다.

## 20K mean-only geometry 비교 실험

기본 5K 결과와 별개로 5K 이후 background Gaussian scale 학습을 확인할 때는 다음 전용
entrypoint를 사용한다.

```bash
bash scripts/run_conference_inclusive2dgs_mean_only_20k_full_pipeline.sh
```

이 스크립트는 이번에 새로 완료된 예측 depth, stride-4/voxel-0.10m point cloud와
emitter-inclusive native 2DGS 20K를 검증 후 재사용한다. AEGIR에서는 scale/rotation/
opacity와 densification/pruning을 고정하고 3D center인 means만 LR `1e-6`으로 학습한다.
기본 material/emitter curriculum은 유지하므로 means LR은 5K부터 줄어 15K에 0이 되며
emitter는 20K까지 학습된다. 명시적인 mean anchor loss는 넣지 않아 geometry-fixed와
비교할 때 위치 최적화 자체의 효과를 분리한다.

`inspect_checkpoint_progression/comparison/`에는 동일한 10개 pose에서 렌더한
initialization/5K/10K/15K/20K 비교 frame, contact sheet, `summary.json`이 생성된다.
렌더 설정은 BRDF/light 128, secondary 8, denoiser off다. 초기화 이전 native 결과와
analytic emitter 초기화는 각각 `emitter_initialization/2dgs_emitter_inclusive.ply`와
`emitter_initialization/map_with_initial_emitters.ply`로 확인한다.

Blender에서 같은 조건을 새 predicted-depth seed와 native 2DGS부터 실행하는 entrypoint는
`scripts/run_blender_inclusive2dgs_mean_only_20k_full_pipeline.sh`다. 과거에는 Conference 완료 뒤
자동 분석하고 Blender를 순서대로 시작하는
[서버 전용 감시 스크립트](https://github.com/hohyun-nam/aegir/blob/61cbf90860df8e7f023e519d090eefe2435dc1c0/scripts/watch_conference_then_run_blender_mean_only_20k.sh)를 사용했다.
감시기는 Conference의 최종 checkpoint, progression summary, 완료 marker를 모두 확인한
경우에만 Blender를 시작했다. `scripts/analyze_mean_only_progression.py`는 각 checkpoint의
PSNR/SSIM/LPIPS와 초기 2DGS 대비 background mean 이동량의 mean/median/p90/p95/p99/max,
고정해야 하는 scale/rotation/opacity의 최대 변화량을 JSON으로 기록한다.

## 현재 구현 검증

- 비 OptiX 전체 테스트 146개와 OptiX tracer 테스트 12개 통과.
- CUDA 2DGS에서 tagged row는 일반 raster의 radius가 0이고, 초기화용 inclusive
  raster에서는 다시 나타나는 것을 확인.
- Conference의 기존 emitter-inclusive 2DGS 213,856개에서 GT emitter 정보 없이
  emitter 2개를 복원. 각각 6,142개/5,115개를 연결하고 202,599개를 배경으로 유지.
- 같은 장면의 모든 191뷰에서 tagged row를 제외해도 뒤쪽 배경 G-buffer가 유효하며
  analytic emitter camera projection이 정상 생성되는 것을 확인.
- 새 전체 2DGS20K + AEGIR5K 수렴 결과는 아직 실행 전이며, 위 검증은 알고리즘,
  저장, raster/BVH 연결에 대한 회귀/smoke 검증이다.

## 2026-09-15 practical 수정 당시 검증

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
