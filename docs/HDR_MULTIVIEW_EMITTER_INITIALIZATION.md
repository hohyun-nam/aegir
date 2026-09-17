# HDR 다중 뷰 에미터 초기화 (2026-09-15)

학습 기본 초기화는 `hdr_multiview`다. 관측 HDR의 밝은 영역과 카메라 내부/외부 파라미터로 조명 위치·방향·면적을 복원한다. **조명 위치를 예측 depth로 역투영하지 않으며 Blender 조명 GT와 emitter ID/mask를 대응관계에 사용하지 않는다.** valid mask는 관측 픽셀 제외 용도로만 사용한다. 배경 맵은 기존의 RGB/모델 예측 depth 기반 2DGS를 사용한다.

## 이전 초기화 문제

`hdr_predicted_depth` 방식은 조명 표면의 모델 depth 오차를 에미터 위치로 옮겼다. 위치를 고정해서 학습하므로 잘못 배치된 조명은 AEGIR에서 복구되지 않았다. HDR 상위 0.5%만 후보로 자르면 거의 균일한 패널이 Monte Carlo 밝기 차이 때문에 작은 조각으로 분리되기도 했다. 다중 뷰 후보를 1% 반경으로 모으는 이전 수정은 후보 부족 오류는 해결했지만 위치 정확도를 보장하지 못했다.

이전 `aegir_hdr_radius01` 결과는 잘못된 초기화 실험으로 유지한다. 새 초기화를 적용하려면 **AEGIR을 새 출력 경로에서 처음부터 학습**한다. 검증된 실전 2DGS는 재사용할 수 있다. 잘못된 에미터를 가진 checkpoint의 exact resume는 초기화를 다시 실행하지 않는다.

## 복원과 검증

1. Train HDR 전체의 유효 luminance 99.5% 분위수를 밝기 기준으로 구하고 그 80% 이상 영역을 연결 성분으로 검출한다. 최소 12픽셀이 필요하다. 화면 가장자리에서 잘린 영역은 위치·면적 추정에 사용하지 않는다.
2. 사각형 영역은 대각선 교점을 중심 관측으로 사용한다. 서로 다른 카메라의 관측 광선을 조합해 삼각측량 후보를 만든다. 양의 깊이, 카메라 baseline, 최소 3도 시차, 광선 간 거리 및 배경 bounds 검사를 통과해야 한다.
3. 모든 뷰에서 후보를 재투영한다. 뷰당 가장 가까운 영역 하나만 대응하며, 최대 3픽셀 이내 관측이 최소 `max(3, ceil(train_views × 0.05))`개 필요하다. Robust least squares로 중심을 정밀화하고 대응을 갱신한다.
4. 사각형은 여러 뷰의 모서리를 대응시켜 평면 방향을 복원하고, 영상 경계를 평면에 역투영하여 크기를 구한다. 타원 등 모서리가 없는 영역은 여러 뷰의 silhouette 중심·공분산을 함께 맞춰 평면 타원을 복원한다. 방향·크기 복원에 실패한 후보도 제외한다.
5. 사각형은 관측 면적과 같은 면적을 가진 타원 support로 근사한다. `π s_u s_v = median(observed area)`이며 장단축비는 평면 PCA로 정한다. `planar_ellipse_2d`는 두 반경, `ellipsoid_3d`는 동일한 면내 support와 scene diagonal × 0.002의 초기 두께 반경을 사용한다. **단일 얇은 면의 두께는 HDR 영상만으로 측정되지 않으므로 이 값은 초기 prior다.**
6. 길쭉한 조명이 HDR threshold 때문에 여러 조각으로 분리되면, 서로 15도 이내로 평행하고 같은 직선·평면 위에서 연속되는 3D support를 하나로 병합한다. 조명 개수를 미리 지정하지 않으며 작고 독립적인 패널은 합치지 않는다.
7. 영역 HDR의 뷰 간 중앙값을 초기 radiance로 사용한다. 기존 방향 모델의 초기 `exp(-1)` 감쇠를 상쇄하도록 shared color에 `e`를 곱한다. 방향별 방사 학습과 2D/3D representation 선택은 유지한다.

유효한 후보가 하나도 없으면 오류로 멈춘다. GT 위치나 모델 depth 또는 임의 조명으로 fallback하지 않는다. 검출된 개수를 6개로 강제하지 않는다. 밝기 기준 이하 조명, 심하게 가려진 조명, 부족한 시차, 비평면 형상은 검출되지 않거나 초기화가 실패할 수 있다. 삼각측량도 카메라 calibration의 오차에는 영향을 받는다.

## 학습 전에 보기

AEGIR root에서 실행한다. 학습은 시작하지 않고 초기화 checkpoint, 투영 비교 이미지와 초록색 에미터를 포함한 SuperSplat PLY를 만든다.

```bash
/workspace/.conda/envs/dnsplatter/bin/python scripts/inspect_hdr_emitter_initialization.py \
  --data <certified_predicted_data> \
  --geometry <certified_2dgs_checkpoint.pt> \
  --representation ellipsoid_3d \
  --output <new_preflight_folder>
```

- `emitter_initialization.json`: 입력 checksum, 사용 뷰 수, 임계값, 조명별 중심/반경/normal/radiance, 대응 프레임·경계·재투영 오차.
- `projection_contact.png`, `projection_overlays/*.png`: 관측 RGB와 HDR 위에 검출 영역(청록), 복원 타원(초록), 중심(빨강)을 표시. 사각형과 면적을 맞춘 타원이므로 모서리까지 동일하게 겹치지는 않는다.
- `map_with_initial_emitters.ply`: 배경 SH-DC 알베도와 초록색 초기 에미터의 공간 배치.
- `initialization.pt`: step 0 시각화 checkpoint. optimizer state가 없는 preflight 파일이며 exact resume용 학습 checkpoint가 아니다.

실제 학습도 초기화 후 `output/emitter_initialization.json`을 저장하며 진단 내용을 checkpoint에 포함한다. CLI, 공통 풀코스와 AEGIR runner 기본값이 모두 `hdr_multiview`다. `hdr_predicted_depth`는 명시적으로 선택하는 과거 방식이다. 신규 기본 AEGIR 출력 경로는 `preddepth_hdrmv_...v2`로 기존 결과와 구분한다.

## 실제 Blender 검증

`outputs/blender_practical3d_20260915_051804/emitter_init_hdr_multiview_v2/`에서 train 120뷰를 사용했다. GT emitter definition과 조명 depth는 사용하지 않았다.

| ID | 중심 xyz (m, 맵 좌표) | 대응 뷰 | 재투영 오차 중앙값 (px) |
| --- | --- | --- | --- |
| 0 | 1.988, -0.661, 2.296 | 47 | 0.484 |
| 1 | -0.595, -0.676, 2.297 | 44 | 0.484 |
| 2 | 1.989, 1.434, 2.303 | 34 | 0.354 |
| 3 | -0.597, 1.415, 2.299 | 31 | 0.411 |
| 4 | -0.590, -2.752, 2.302 | 30 | 0.361 |
| 5 | 2.001, -2.737, 2.300 | 28 | 0.292 |

위치가 천장 높이의 2×3 배열로 복원되고 normal도 천장 면에 가깝게 정렬됐다. 이 결과는 **초기화 위치·형상 검증**이며 재학습 후 렌더링 품질이나 Monte Carlo 노이즈 개선을 보장하는 수렴 평가가 아니다.

합성 사각형/타원 다중 뷰 복원, 면적/방향/radiance, 패널 영역 분할, 부족한 관측/시차에서 fallback 금지 및 fresh training 연결을 포함해 전체 테스트 145개가 통과했다.

실제 Blender 배경132,417 Gaussians/초기 에미터6개로 GPU128/1 학습2-step과 exact resume1-step을 실행했다. frozen geometry/white0/SH proxy0을 확인했고 loss가 유한하며 초기화 진단이 resume 후에도 보존됐다. 이 smoke는 실행 검증이며 전체20K 재학습은 시작하지 않았다. `verification.json`에 검사 결과를 저장했다.


## Conference 초기화 시간 교정

2026-09-15의 중복 피팅 수정 후 같은 관측 ID 대응은 pass당 한 번만 형상 검증하고, 후보 점수는 에미터가 수락되어 관측이 소비될 때만 다시 계산한다. 실패 후보의 최종 재투영/크기 조건은 그대로 유지한다. 진행 로그와 diagnostic의 unique_shape_fits/duplicate_hypotheses_skipped/score_passes/elapsed_seconds로 작업 상황을 확인할 수 있다.

초기 최적화 버전은 Conference191뷰에서 약49.5초에13개 후보를 수락했다. 하지만 실제 장면의 두 긴 천장 조명이 밝기 단절 때문에 여러 조각으로 과분할된 결과였고, 그 checkpoint는 올바른2-emitter 기준으로 사용하지 않는다.

2026-09-17 교정 버전은 같은 GT-free HDR/카메라 입력에서 원시13개 후보를 **두 개의 긴 스트립 에미터**로 병합했다. 결과는 `outputs/conference_practical3d_hdrmv_20260915/emitter_init_hdr_multiview_merged_v3/`에 있으며 `projection_contact.png`에서 두 실제 스트립을 따라가는 support를 확인했다. Blender120-view 실제 회귀에서는 원시6개가 병합 후에도6개로 유지되고 중심·크기·재투영 수치가 기존 결과와 같았다. 이 검사는 초기화 형상 평가이며 최종 수렴/렌더링 품질 평가는 새2-emitter 초기화로 처음부터 학습한 뒤 수행한다.
