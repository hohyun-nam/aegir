# AEGIR 논문 대비 코드 검토 메모

검토일: 2026-08-12  
검토 대상:

- 논문: `/workspace/hohyun_ws/aegir.pdf`
- 수정 메모: `/workspace/hohyun_ws/AEGIR고칠꺼`
- 현재 구현: `/workspace/hohyun_ws/AEGIR`

## 가장 중요한 정정

논문의 iteration당 ray 수는 **2^16 = 65,536 rays**가 맞다.

PDF 텍스트 추출 과정에서 위첨자 `16`이 본문 숫자 `2`에 붙어 `216`처럼
추출되었고, 이를 216 rays로 잘못 해석했던 이전 검토 내용은 폐기한다.

현재 구현의 `rays_per_step=1,024`는 논문의 65,536 rays보다 64배 적다.
논문 설정을 재현할 때는 메모리와 실행 시간을 확인하면서 최종적으로
65,536 rays/iteration을 기준으로 맞춰야 한다.

## 결론

현재 구현은 다음 핵심 구조는 맞다.

- 하나의 shared planar Gaussian scene을 사용한다.
- 각 Gaussian이 albedo, roughness, metallic을 직접 가진다.
- 카메라마다 현재 Gaussian 파라미터로 임시 G-buffer를 즉석에서 만든다.
- 공식 2DGS `diff-surfel-rasterization`을 사용한다.
- direct shadow visibility에 IRGS OptiX tracer를 사용한다.
- emitter split 시 부모를 남기지 않고 두 자식으로 교체한다.
- DiffusionRenderer의 albedo/roughness/metallic만 material prior로 사용한다.
- AEGIR loader는 GT depth/normal을 직접 supervision으로 읽지 않는다.

하지만 아래 문제를 수정하기 전에는 최종 AEGIR 20k 학습을 시작하지 않는다.

## P0: 학습 전 반드시 수정

### 1. [완료] 회전된 anisotropic emitter의 ray intersection 축 오류

파일: `aegir/emitters.py`, `AreaEmitterModel.intersect()`

2026-08-12 수정 완료. `R`이 local-to-world 축을 column으로 저장하고 tensor가 row-vector라는
convention에 맞춰 world-to-local을 `v @ R`로 통일했다. `intersect()`와 `radiance()`를
수정했으며, 45도 회전한 scales `(3, 1, 0.5)` emitter에서 장축/중축/단축 거리가
각각 `3/1/0.5`인지 검증한다. `radiance()`와 `radiance_for_ids()` 일치 및 quaternion,
scale gradient finite 여부도 회귀 테스트로 고정했다.

### 2. train/validation/test view 누수

**상태 (2026-08-12): 구현 완료.**

- `MultiViewSceneDataset(split=train|val|test|all)`을 추가했다.
- AEGIR 학습은 명시적으로 `train`, `inspect`는 기본 `test`를 사용한다.
- split 필터를 `max_views`보다 먼저 적용하며, 원본 frame index를 보존해
  `frame_XXXX.pt` material prior와 정렬한다.
- split metadata가 전혀 없는 FIPT-Real은 `train`에서 전체 frame을 사용할 수 있지만
  `val/test`는 오류를 내므로 전체 frame metric을 held-out 결과로 잘못 보고하지 않는다.
  FIPT-Real held-out 평가는 별도 protocol을 정해 metadata를 추가해야 한다.

파일: `aegir/data.py`, `MultiViewSceneDataset`

현재 loader는 `transforms.json`의 모든 `frames`를 읽으며 top-level
`train_filenames`, `val_filenames`, `test_filenames`를 무시한다.

FIPT-Synthetic 현황:

- bathroom: train 109, validation 13
- bedroom: train 208, validation 14
- kitchen: train 202, validation 15
- livingroom: train 213, validation 14

현재 상태로 학습하면 validation view도 학습에 포함된다. HyperSim과 Isaac Sim도
같은 문제가 있다. `inspect` 역시 같은 전체 frame을 평가하므로 held-out metric이
아니다.

필요 수정:

- dataset에 명시적인 `split=train|val|test` 지원
- 학습은 train filename만 사용
- 평가는 held-out filename만 사용
- prior 파일도 원본 frame identity에 맞춰 매핑
- FIPT-Real은 현재 split metadata가 없으므로 평가 protocol을 별도로 정의

### 3. dataset mask와 2DGS valid mask 미사용

**상태 (2026-08-12): 구현 완료.**

- frame에 `mask_path`가 있으면 nearest-neighbor로 읽고, 없으면 all-valid mask를 만든다.
- metadata가 mask를 명시했는데 파일이 없거나 크기가 다르면 데이터 오류로 보고한다.
- photometric L1/SSIM, material prior, edge-aware TV와 emitter 초기화에 mask를 적용한다.
- 학습 최종 mask는 dataset mask와 rasterized 2DGS `valid_mask`의 교집합이다.
- TV는 양쪽 pixel이 모두 valid인 이웃 pair만 사용한다.

파일: `aegir/data.py`, `aegir/trainer.py`, `aegir/losses.py`

변환 데이터에는 `mask_path`가 있지만 AEGIR loader가 읽지 않는다. 2DGS가 만든
`valid_mask`도 photometric, material-prior, TV loss에 적용하지 않는다.

확인된 mask 유효 비율:

- FIPT-Synthetic: 대부분 거의 100%
- HyperSim: 일부 frame은 약 77.3%만 유효
- Isaac Sim: 일부 frame은 약 86.8%만 유효

GT mask는 invalid/이상 픽셀을 제외하는 용도이므로 사용한다. 최종 loss mask는
dataset mask와 rasterized valid mask를 결합해야 한다. TV에서는 invalid 경계를
가로질러 smoothing하지 않도록 인접 pixel pair mask가 필요하다.

### 4. 논문에 없는 learnable global ambient 제거

**상태 (2026-08-12): 구현 완료.**

- `AreaEmitterModel`과 `EmitterState`에서 ambient 파라미터를 제거했다.
- renderer의 HDR은 `direct + indirect`만 합산하며 ambient 출력 채널도 제거했다.
- ambient config, 초기화, regularizer와 새 checkpoint 저장 필드를 제거했다.
- 새 checkpoint format은 v3이며, 기존 v1/v2의 `ambient_color`는 로드 시 무시한다.
- 실제 기존 v1 checkpoint CPU load와 ambient 부재를 검증했다.

파일: `aegir/emitters.py`, `aegir/render.py`, `aegir/config.py`

현재 global ambient는 trainable이고 초기값은 0.02이며 기본 규제 가중치는 0이다.
기존 체크포인트에서 ambient가 다음처럼 증가했다.

- HyperSim: 약 `[2.01, 1.84, 1.65]`
- Isaac Sim: 약 `[1.08, 0.83, 0.53]`

따라서 넓은 조명을 explicit area emitter 대신 ambient가 설명할 수 있다. 논문
재현에서는 ambient를 제거하거나 정확히 0으로 고정한다. 단순히 약한 penalty를
추가하는 것보다 표현 자체에서 제외하는 것이 기준 구현에 가깝다.

### 5. scene normalization 또는 scene-scale-aware visibility

파일: `aegir/config.py`, `aegir/irgs_tracer.py`, `aegir/trainer.py`

논문은 scene을 정규화한 뒤 고정 visibility 파라미터를 사용한다고 설명한다.
현재 fixed-pose DN checkpoint는 identity transform, scale 1.0이며 실제 meter 좌표를
그대로 사용한다.

대략적인 camera extent diagonal:

- FIPT-Synthetic: 약 5.9~8.9 m
- HyperSim 예시: 약 8.0 m
- Isaac Sim warehouse: 약 33.0 m

그런데 모든 scene에 같은 visibility bias 0.001, steepness 80을 적용한다. scene을
논문 방식으로 정규화하거나 bias/steepness와 위치 learning rate를 scene scale에
맞게 변환해야 한다.

논문의 ray-marching transmittance 종료값은 0.05이고 현재 기본값은 0.03이므로
논문 설정으로 수정해야 한다.

### 6. secondary bounce 구현을 논문 방식으로 교체

파일: `aegir/trainer.py`, `aegir/render.py`, `aegir/inspect.py`

논문:

- 학습 시 secondary ray의 첫 2D Gaussian hit를 trace하고 그 Gaussian의 SH/color
  feature를 proxy radiance로 조회한다.
- inference 시 SH proxy를 끄고 secondary hit에서 recovered explicit emitter까지
  다시 trace한다.

현재 구현:

- 전체 surfel이 아니라 균등하게 고른 1,024개 Gaussian center만 사용
- center를 작은 sphere처럼 근사하여 point-distance hit 검사
- full SH가 아니라 DN의 DC 색상만 보존
- secondary tracing에 IRGS를 사용하지 않음
- 별도의 4개 cosine sample을 direct term과 무관하게 더함
- 임의의 `bounce_strength=0.25` 사용
- inspect와 novel-view inference에서도 계속 SH/DC proxy 사용

학습용 SH proxy transport와 최종 emitter-based inference transport를 별도 경로로
구현해야 한다. 논문의 visibility-weighted incident-light 식과 현재 독립적인
`direct + indirect` 합산 방식도 다시 대조한다.

### 7. DiffusionRenderer glue pipeline 복원/재구현

**2026-08-12 구현 완료.** 현재 workspace에 RGB/mask 입력 준비, official inverse inference,
material `.pt` import, one-command wrapper가 존재한다. bedroom 222 frame을 20-step/256으로
실제 생성했고 train/val/test 원본 frame ID 정렬과 세 material field를 검증했다. loader는
`--prior-dir` 사용 시 세 필드를 모두 강제한다. GT depth/normal은 생성하거나 읽지 않는다.

DiffusionRenderer inverse checkpoint와 다음 연결 스크립트가 현재 workspace에 준비됐다.

- `prepare_diffusionrenderer_input.py`
- `run_diffusionrenderer_priors.py`
- `import_diffusionrenderer_priors_to_dir.py`
- `run_diffusionrenderer_material_priors.sh`

이전 `export_dnsplatter_gbuffers.py`는 shared-Gaussian 구현에서 필요하지 않다. DN checkpoint가
직접 shared geometry를 초기화하며, DiffusionRenderer는 RGB에서 material prior만 예측한다.

필요 결과는 frame identity가 보존된 per-view 파일이며 AEGIR 학습에는 다음 세
prior만 전달한다.

- `prior_albedo`
- `prior_roughness`
- `prior_metallic`

DiffusionRenderer의 depth/normal 출력은 AEGIR supervision에 사용하지 않는다.

## P1: 논문 재현성과 학습 품질을 위해 수정

### 8. ray budget

- 논문: 2^16 = 65,536 rays/iteration
- 현재 기본값 및 CLI/run script: 1,024 rays/iteration
- 현재는 논문보다 64배 적음

BRDF sample 128개와 emitter sample 128개가 primary ray마다 평가되는 구조이므로,
65,536 설정의 실제 VRAM/시간을 먼저 작은 smoke run으로 측정해야 한다. OOM을
피하기 위한 chunked ray evaluation을 구현하되, 논문과 동일한 총 ray 수와 loss
정규화를 유지한다.

### 9. 색공간과 tone mapping

현재 renderer는 linear HDR에 Reinhard tone mapping만 적용한다. target은 PNG를
단순히 `[0,1]`로 읽으며 sRGB/linear 변환을 하지 않는다. HyperSim converter는
frame마다 90-percentile exposure와 gamma를 적용한다.

확인할 사항:

- target RGB가 sRGB인지 linear LDR인지 명시
- rendered linear HDR에 동일한 camera response/tone mapping 적용
- 필요하면 per-view exposure를 metadata로 보존하거나 최적화
- DiffusionRenderer basecolor prior의 sRGB/linear convention 확인
- BRDF에서 쓰는 albedo는 linear reflectance로 통일

### 10. focus map과 patch 경계

현재 focus map은 전체 target image가 아니라 선택된 patch 안에서 계산된다. patch
경계의 zero padding이 인공적인 contrast를 만들고 `max(delta)`도 patch마다 달라진다.
전체 view에서 focus map을 계산한 뒤 동일 patch를 crop하거나, 논문 구현과 동일한
sampling/normalization을 사용해야 한다.

### 11. 논문 외 emitter regularizer

현재 기본값에는 논문에 없는 다음 항목이 추가되어 있다.

- surface proximity: weight 10.0
- camera-frustum constraint: weight 1.0

논문의 기본 `area + white + bounds` 결과와 직접 비교하려면 이 둘은 기본 off로
두고 별도 ablation으로 평가한다. area/bounds loss의 scene-scale normalization도
논문 수식과 현재 비정규화 좌표계의 차이를 명시한다.

### 12. emitter adaptation optimizer state

split/prune 후 현재 코드는 emitter Adam optimizer 전체를 새로 만든다. 살아남은
emitter의 momentum도 손실된다. 문서에는 momentum을 보존한다고 적혀 있어 코드와
문서가 불일치한다. unchanged emitter state를 매핑해 보존하거나 해당 차이를
명시해야 한다.

### 13. resume와 reproducibility

현재 AEGIR checkpoint에는 model과 loss history만 있고 다음 상태가 없다.

- geometry/material/emitter optimizer state
- 현재 iteration을 기준으로 한 resume entrypoint
- CPU/CUDA RNG state
- data/patch sampler state

SSH/tmux 장애 후 exact resume가 불가능하다. 20k 학습 전에 resume 기능과 고정 seed를
추가한다.

### 14. evaluation

현재 `inspect`는 전체 dataset에 대해 PSNR만 계산하며 mask/split을 사용하지 않는다.
논문 비교를 위해 다음이 필요하다.

- held-out test views only
- invalid mask 적용
- PSNR, SSIM, LPIPS
- raw Monte Carlo output metric과 visualization-only denoised output 분리
- inference에서는 SH proxy를 완전히 비활성화
- 충분한 render sample 수와 재현 가능한 sampling seed

## 현재 테스트 상태와 필요한 회귀 테스트

2026-08-12 CPU 테스트 결과:

- 26 passed
- 3 skipped (CUDA/OptiX 경로)

추가해야 할 테스트:

- [완료] rotated anisotropic emitter intersection
- [완료] `radiance()`와 `radiance_for_ids()` local-frame 일치
- [완료] train/val/test frame selection
- [완료] dataset mask + raster valid mask loss 적용
- [완료] ambient 표현 자체가 없고 legacy ambient가 무시되는지 검증
- training SH bounce와 inference emitter bounce 분기
- scene scale 변화에 대한 visibility 동등성
- 65,536-ray chunked evaluation과 unchunked reference의 수치 동등성
- checkpoint exact resume 동등성
- held-out masked metrics

## 수정 권장 순서

1. train/val/test split과 mask 처리
2. [완료] 회전 emitter intersection 및 회귀 테스트
3. global ambient 제거
4. scene normalization/visibility threshold 정합
5. secondary bounce 학습·추론 경로 구현
6. [완료] DiffusionRenderer input/prior/import pipeline 구현 및 bedroom 222-frame 검증
7. 2^16 ray를 위한 chunked transport와 논문 hyperparameter 적용
8. 색공간/tone mapping 정합
9. exact resume와 평가 pipeline 구현
10. 전체 CUDA/OptiX smoke test 후 AEGIR 20k 학습

## 작업 원칙

- AEGIR 학습에는 GT camera pose와 invalid-pixel mask를 사용할 수 있다.
- AEGIR 또는 DN-Splatter 학습 supervision에 GT depth는 사용하지 않는다.
- DN-Splatter geometry는 RGB + fixed GT pose + COLMAP point initialization + RGB-derived
  ZoeDepth/PearsonDepth 설정을 사용한다.
- DiffusionRenderer depth/normal은 AEGIR material supervision으로 사용하지 않는다.
- material representation은 per-view trainable G-buffer가 아니라 per-Gaussian BRDF다.
- G-buffer는 매 camera/iteration마다 현재 Gaussian 상태로 즉석 rasterization한다.
