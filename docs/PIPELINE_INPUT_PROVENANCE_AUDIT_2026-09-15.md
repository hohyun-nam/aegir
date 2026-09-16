# 파이프라인 입력 출처 및 실전성 점검 — 2026-09-15

> **수정 상태 (2026-09-15):** 사용자 요청에 따라 항목 1–6, 8은 코드/실행 경로에서 수정했다. 항목 7의 Conference 평가 분할은 변경하지 않았다. 아래 내용은 수정 전 진단을 보존한다. 현재 기본값과 검증 범위는 [실전 파이프라인 문서](PRACTICAL_TRAINING_PIPELINE.md) 및 changelog의 동일 날짜 교정 항목을 참고한다. 이전 GT-assisted checkpoint 자체는 교정된 결과로 재분류하지 않는다.


사용자 요구: 카메라 정보와 허용된 마스크, 관측 RGB/HDR를 사용한다. Geometry는 depth 모델의 예측으로 학습하고, 에미터는 고휘도 HDR RGB 후보에서 초기화한다. 데이터셋 GT depth와 Blender 조명 정답 geometry를 기본 학습 입력으로 사용하지 않는다.

이번 작업은 점검과 기록이다. 아래 문제의 동작 수정이나 재학습은 아직 수행하지 않았다. 과거 사용자가 요청한 GT-depth 포인트 초기화 실험 자체와, 그 실험 설정을 실전 기본 파이프라인으로 일반화한 실수를 구분한다.

## 판정

현재 Blender GT-depth full-course 및 새 Conference depthseed full-course는 GT geometry 정보를 사용한 실험이다. RGB/HDR와 depth-model 예측만으로 map/lighting을 복원하는 파이프라인으로 설명하면 안 된다.

- `scripts/run_blender_gtdepth_full_pipeline.sh`
- `scripts/run_conference_depthseed_full_pipeline.sh`

원래 RGB-derived ZoeDepth + Pearson 경로는 `scripts/run_blender_my_room_full_pipeline.sh`에 남아 있다. 그러나 이 오래된 runner 전체를 그대로 재사용하면 최신 마스크 처리, 초기화 정책, 입력 출처 검증이 자동으로 해결되는 것은 아니다.

## 확인한 문제와 증거

### 1. Point cloud와 2DGS 모두 데이터셋 depth를 사용한다

Conference runner는 원본 `depths`에서 seed를 만들고, 같은 디렉터리를 준비 데이터의 `mono_depth`로 symlink한다 (`run_conference_depthseed_full_pipeline.sh:88`). 2DGS 단계에는 `DEPTH_LOSS_TYPE=l1`, `LAMBDA_DEPTH=0.2`가 전달된다. Blender GT-depth runner도 같은 구조다 (`run_blender_gtdepth_full_pipeline.sh:119,157`).

이는 GT depth를 초기 point cloud에서만 사용하는 것이 아니다. Geometry 최적화 전체가 GT depth 감독을 받는다. 디렉터리 이름이 `mono_depth`라고 해서 모델 예측 depth가 되는 것은 아니다.

수정 방향: 학습용 depth 파일의 출처를 모델 예측으로 명시하고, seed와 depth loss 모두 그 출처를 검증한다. 예측의 오차와 scale/shift 특성에 맞춰 원래 Pearson 경로 또는 별도의 정합된 loss를 선택한다. GT를 이용한 depth alignment도 기본 경로에서 금지해야 한다.

### 2. HDR 에미터 후보 검출보다 마스크 + 데이터셋 depth 초기화가 먼저 실행된다

`aegir/trainer.py:202`가 `seed_planar_emitters_from_masks`를 먼저 호출하고, 성공하면 즉시 반환한다. 이 함수는 `frame.metric_depth`를 역투영해서 에미터 위치와 모양을 얻는다 (`aegir/initialization.py:28`). 현재 두 full-course 모두 emitter mask와 dataset depth를 제공하므로 HDR/G-buffer 초기화 경로가 우회된다.

HDR는 이 경로에서 주로 초기 RGB radiance를 정하는 데 쓰인다. 위치 후보와 개수를 고휘도 HDR 이미지에서 검출하는 것은 아니다. 마스크 사용이 허용되더라도, 마스크 + GT depth 초기화를 HDR 기반 초기화로 설명하면 안 된다.

수정 방향: 초기화 방식을 명시적으로 선택하도록 만들고, 실전 경로는 관측 HDR의 밝은 후보를 사용하는 방식으로 고정한다. 마스크는 허용된 보조 역할을 수행하되 위치의 GT depth 역투영 경로를 자동 선택하지 않는다.

### 3. Blender 조명 bounding box와 두께가 학습 초기화에 사용된다

`aegir/data.py:235`가 데이터 디렉터리의 `emitters.json`을 자동 로드한다. 마스크 초기화는 이 bbox를 평면/패널 초기화에 사용할 수 있고, 3D support 복원은 법선 방향 반경을 bbox 두께에서 얻는다 (`aegir/initialization.py:186`). 값이 없는 cluster에 metadata color × strength를 넣는 fallback도 코드에 있다. 이번 학습에서 그 intensity fallback까지 실제 사용됐다고 단정하지는 않는다.

수정 방향: 학습용 HDR 초기화 경로는 GT bbox/두께/강도를 읽지 않는다. 평가용 GT 조명과, 사용자가 직접 정의한 relighting 조명 입력은 별도 경로로 유지한다. 이미 GT bbox로 초기화된 checkpoint는 로더만 바꿔서 실전 학습 결과로 바뀌지 않는다.

### 4. 에미터 geometry 제외와 기존 HDR/G-buffer 초기화가 충돌한다

현재 2DGS는 emitter mask 영역을 seed와 surface supervision에서 제외한다. 그런데 기존 HDR 초기화는 `frame.mask & gbuffer.valid_mask`를 요구하고 (`aegir/trainer.py:235`), 후보의 3D 위치를 G-buffer points에서 가져온다 (`aegir/initialization.py:415`). 의도대로 에미터 Gaussian이 없는 위치에서는 해당 후보를 탈락시킬 수 있다. 주변 Gaussian coverage가 남아 있다면, emitter 위치 대신 뒤쪽 표면 위치를 얻을 수도 있다.

따라서 마스크 기반 초기화 호출을 삭제하는 것만으로 해결되지 않는다.

수정 방향: emitter를 제외하기 전의 RGB-derived geometry/depth로 후보를 초기화하고 그 상태를 보존하거나, emitter-excluded map과 별개로 예측 depth 또는 다중 뷰 정합을 사용한다. 최종 배경 Gaussian 제외와 조명 위치 추정을 한 G-buffer validity 조건에 묶지 않는다.

### 5. 예측 depth 파일의 단위가 GT-depth seed 경로와 다르다

DN-Splatter depth generator는 `SCALE_FACTOR=0.001`로 모델의 meter depth를 나눠 NPY에 저장한다 (`external/dn-splatter/dn_splatter/scripts/depth_from_pretrain.py:206`, `utils/utils.py:31`). 현재 Conference의 raw Zoe `frame_0000.npy`는 min 818.749, median 2727.981, max 6644.797이다. meter 단위로는 각각 약 0.819, 2.728, 6.645다.

`prepare_zoe_raw_depth_dir.py`는 이 파일을 단위 변환 없이 symlink한다. Pearson loss는 양의 전역 배율에 불변이므로 이 단위 차이가 눈에 띄지 않을 수 있다. 하지만 현재 seed의 0.1–15m depth filter나 metric L1에 그대로 넣으면 1000배 오류가 된다.

수정 방향: 예측 depth를 meter로 명시적으로 변환하고 카메라 depth convention, 좌표계 scale, 메타데이터를 함께 검증한다. 모든 추정 depth를 GT에 정합해야 한다는 뜻은 아니다.

### 6. 입력 출처 검증과 캐시 재사용 검증이 부족하다

준비 스크립트는 `depth_file_path`를 제거하지만, 뒤에서 외부 디렉터리를 `mono_depth`로 연결할 수 있다. GT 파일을 연결해도 현재 금지-key 검사만으로 잡히지 않는다. Full-course는 seed PLY 또는 `transforms.json`이 존재하면 재사용하며, 모델·단위·split·mask 설정 전체를 검증하지 않는다.

현재 준비 스크립트의 금지 목록에는 `normal_world_file_path`, `metallic_gt_file_path`도 빠져 있다. 현재 2DGS 명령에는 `--normal_supervision`이 없고 AEGIR 재질은 DiffusionRenderer prior를 읽으므로, 이 metadata 잔존을 현재 활성 GT normal/material loss라고 주장하지 않는다. 하지만 sanitization이 완전하다고 볼 수는 없다.

수정 방향: depth/seed/emitter 초기화에 provenance manifest를 저장하고, GT/model 구분을 실행 전에 검사한다. 캐시 재사용은 출처와 설정이 일치할 때만 허용한다. 비실전 GT 실험은 별도 명시적 모드로 둔다.

### 7. Conference 평가가 held-out NVS 평가가 아니다

현재 원본 및 준비 Conference metadata에는 train/val/test split이 없다. 완료된 새 AEGIR checkpoint는 `num_views=191`, `split=train`, `frame_indices=0..190`을 기록한다. 후속 inspect runner는 `--split all --max-views 30`을 사용한다.

이 결과는 학습뷰 reconstruction이다. Held-out map generalization 성능으로 해석하면 안 된다. Depth, geometry, emitter initialization, material-prior temporal chunks에도 공통 train split을 적용해야 한다.

`build_gt_depth_point_cloud.py`는 프레임의 `split`만 검사하고 없으면 train으로 간주한다. 따라서 filename split만 있는 일반 데이터셋에는 별도 보완이 필요하다. 현재 Blender dataset에는 frame별 split도 있으므로 해당 Blender seed가 이미 test frame을 사용했다고 단정하지 않는다.

### 8. HDR radiance 초기값이 각도 커널의 기본 감쇠를 보상하지 않는다

현재 두 initializer 모두 channel median을 `colors`로 넣는다. 기본 spread=1, power=1, direction gains=1일 때 단위 방향의 커널은 exp(-1)이다 (`aegir/emitters.py:260`). 실제 CPU 함수 호출로 C=(1,1,1), direction=+Z에서 Le=(0.36787945,0.36787945,0.36787945)를 재현했다.

따라서 HDR median을 출력 radiance와 같게 맞춘 초기화가 아니다. 이전 2026-09-06 audit의 e배 보상 설명과 현재 코드도 일치하지 않는다. 이는 초기값의 문제이며 모든 학습 완료 결과가 항상 이 비율로 어둡다는 뜻은 아니다.

수정 방향: 관측 방향에서 초기 radiance가 HDR 관측에 맞도록 angular kernel을 보상하거나 초기 방사 profile을 정합한다. GT intensity를 넣는 해결책으로 돌아가지 않는다.

## 사용자가 선택한 설정과 구현 실수를 구분

SH-DC albedo 초기화, white loss=0, geometry freeze, secondary-only, 1-sample training/8-sample inspect, selectable 2D/3D emitter, emitter overlap exclusion, mask loss 0.02/protection 7은 사용자 선택/비교 결과에 따른 설정이다. 이 설정 자체를 이번 GT 입력 오류로 취급하거나 전부 논문 기본값으로 되돌리면 안 된다.

다만 추정 emitter center/rotation을 고정하고 scale을 초기값 이하로 제한하면 초기화 오차를 복구하기 어렵다. GT 정보 없이 초기화하는 경로에서 이 제약의 영향은 별도로 평가해야 한다.

Secondary NEE + PDF surrogate, expected-hit 집약, origin offset 0.15m, SH-DC appearance/albedo의 색 공간, 낮은 roughness GGX epsilon은 별도의 기존 rendering audit 항목이다. 이번 입력 점검만으로 그 문제의 수정이나 현재 artifact의 원인을 확정하지 않았다. 최신 코드에서도 NEE/surrogate와 GGX epsilon 구조는 확인했다.

## 정상적으로 분리되어 있는 입력

- RGB/HDR 관측값을 photometric target으로 사용하는 것은 필요한 이미지 감독이다.
- DiffusionRenderer material priors는 RGB로부터 예측된다. 현재 import는 GT depth/normal/material을 학습 감독으로 추가하지 않는다.
- 현재 2DGS runner는 GT normal supervision을 켜지 않는다. Rendered normal/depth consistency regularization과 외부 GT normal loss를 혼동하면 안 된다.
- AEGIR는 초기화 후 반복 학습에 GT depth loss를 걸지 않는다. 그렇더라도 GT depth로 학습된 고정 geometry와 GT-assisted emitter 초기화의 영향은 남는다.
- 새 조명을 사용자가 직접 정의하고 해당 파라미터를 넣는 relighting editing은 기존 조명을 이미지에서 추정하는 학습과 별도 작업이다. 이전 Blender radiance 대응 보정 실험은 calibration-assisted 결과로 구분해야 한다.

## 수정할 파이프라인

1. 공통 train/val/test split을 먼저 결정한다.
2. Train RGB로 depth 모델을 실행하고 출처·단위·convention을 기록한다.
3. 예측 depth 또는 train-view triangulation으로 seed를 생성한다. 예측 depth seed는 meter/pose scale 정합을 검증하고 voxel downsampling을 유지한다.
4. 예측 depth 감독으로 native 2DGS를 학습한다.
5. 관측 HDR의 고휘도 후보 + RGB-derived 3D geometry/depth로 emitter를 초기화한다. GT bbox/두께/강도는 사용하지 않는다.
6. Emitter geometry를 제외한 배경 map과 HDR 초기화에 필요한 위치 정보를 분리해 유지한다.
7. AEGIR는 선택한 BRDF/transport 설정으로 학습한다. 재질 prior는 train RGB 기반 예측값만 사용한다.
8. Held-out views에서 reconstruction/relighting을 평가하고, GT는 평가 target 및 사용자가 허용한 마스크 역할로 구분한다.

## 논문 확인 범위

[AEGIR §3.2](https://arxiv.org/html/2606.28635v1#S3.SS2)에는 고휘도 RGB 역투영, DBSCAN, PCA 기반 초기화가 기술되어 있다. Appendix C는 DN-Splatter 10K 초기 geometry를 기술하지만 특정 depth 모델이 ZoeDepth라고 본문에서 확인되지는 않는다. 따라서 '논문이 Zoe/Pearson을 정확히 사용했다'고 단정하지 않는다. 이번 사용자 요구의 HDR RGB 후보 검출은 관측 이미지에서 조명을 추정한다는 핵심 구조를 따른다.

논문은 조명 pose까지 공동 최적화하고 학습 indirect에는 SH proxy를 사용한다. 현재 선택한 freeze/secondary-only/방향성 확장은 논문과 다른 명시적 실험 설정이며 GT 입력 문제와 구분한다.
