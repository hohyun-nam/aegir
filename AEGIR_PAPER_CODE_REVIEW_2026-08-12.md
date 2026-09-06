# AEGIR 논문 대비 코드 검토 메모

검토일: 2026-08-12  
검토 대상:

- 논문: `/workspace/hohyun_ws/aegir.pdf`
- 수정 메모: `/workspace/hohyun_ws/AEGIR고칠꺼`
- 현재 구현: `/workspace/hohyun_ws/AEGIR`

## 가장 중요한 정정

논문의 iteration당 총 transport ray 수는 **2^16 = 65,536 rays**가 맞다.

PDF 텍스트 추출 과정에서 위첨자 `16`이 본문 숫자 `2`에 붙어 `216`처럼
추출되었고, 이를 216 rays로 잘못 해석했던 이전 검토 내용은 폐기한다.

중요하게도 이 값은 primary pixel 수가 아니다. 논문의 point당
`N_brdf=128`, `N_light=128` 설정을 함께 적용하면 primary surface point는
`65,536 / (128 + 128) = 256`개다. 로컬 옵션 `rays_per_step`은 역사적인 이름과
달리 primary surface point 수를 뜻하므로 기본값을 256으로 설정한다.

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

**2026-08-14 구현 완료.** 현재 구현:

- DN의 frozen DC+rest full SH 16계수를 checkpoint에 보존
- training은 direct shadow와 동일한 IRGS trace의 opacity-composited SH color를 재사용
- 각 primary transport sample에서 `V*L_direct + (1-V)*L_bounce`로 결합
- inference는 SH를 요청하지 않고 first Gaussian hit의 material/normal에서 diffuse
  sampling 후 recovered explicit emitter까지 다시 trace; miss는 0
- 기존 1,024 center/sphere proxy와 `bounce_strength=0.25` 제거
- v5 checkpoint와 training/inference 분리 회귀 및 CUDA gradient test 통과

논문이 inference secondary sample count를 별도 숫자로 명시하지 않아 기본 1개로
설정했으며, 이는 inference variance-control option으로만 유지한다.

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

### 8. [완료] ray budget

- 논문: 2^16 = 65,536 total transport rays/iteration
- BRDF 128 + emitter 128 = primary point당 transport ray 256개
- 따라서 primary surface point는 iteration당 256개
- 현재 기본 config, CLI 및 production run script를 256 primary point로 통일

로컬 `--rays-per-step`은 primary surface point 수를 받는 과거 이름이다. 따라서
논문 설정은 `--rays-per-step 256 --num-brdf-samples 128 --num-light-samples 128`이다.
65,536을 이 옵션에 직접 넣으면 `65,536 x 256 = 16,777,216` transport rays가 되어
논문 budget의 256배가 된다. 2026-08-14 bedroom stress test에서 이 잘못된 대형
설정은 24,000 MiB PyTorch cap 아래 약 15.4초/iteration, 20k ETA 약 85.6시간이었다.

### 9. 색공간과 tone mapping

**2026-08-13 구현 완료.** 완전한 HDR frame이 있는 scene은 PNG를 학습 target으로 쓰지
않고 scene-linear HDR을 symmetric `log1p` domain에서 비교한다. LDR-only scene은
`aegir_color-v1` metadata의 exposure/tone mapper/transfer function을 그대로 사용하며,
정보가 없을 때만 learned scene exposure + Reinhard+sRGB fallback을 명시적으로 쓴다.
train/eval이 같은 변환 함수를 사용하고 policy와 learned exposure를 checkpoint에 저장한다.

DiffusionRenderer basecolor PNG는 기본 sRGB source로 명시하여 importer에서 한 번 linear
reflectance로 decode하고, BRDF/material-prior loss는 linear albedo만 받는다. 기존 30개
converted scene에도 관측 domain metadata를 기록했다. raw linear HDR/metric tensor와
visualization PNG를 분리했으며 sRGB 왕복, exact/fallback LDR, HDR loss, converter/importer
회귀를 포함한 전체 CPU test `68 passed`를 확인했다.

### 10. focus map과 patch 경계

**2026-08-13 구현 완료.** 전체 target view와 combined valid mask에서 7x7 local
contrast 및 view-global `max(delta)` normalization을 먼저 계산하고, target/G-buffer와
동일한 patch 좌표로 focus weight를 crop한다. 따라서 동일 source pixel의 weight는 patch
위치와 크기에 영향받지 않는다. invalid RGB는 local average와 normalization에서 제외되며
최종 masked L1에서도 제외된다. 서로 다른 두 crop의 겹치는 픽셀 일치, invalid 값 변경
불변성, precomputed crop의 photometric loss 적용 회귀 테스트를 추가했다.

### 11. 논문 외 emitter regularizer

수정 전 기본값에는 논문에 없는 다음 항목이 추가되어 있었다.

- surface proximity: weight 10.0
- camera-frustum constraint: weight 1.0

2026-08-13에 둘의 기본 가중치를 `0.0`으로 변경했다. 계산 기능은 삭제하지 않았으며,
emitter가 표면에서 과도하게 뜨거나 모든 관측 frustum 밖으로 이탈하는 경우에만
`--light-surface`와 `--light-view`로 명시적으로 다시 활성화한다. 논문의 기본
`area + white + bounds` 항은 유지한다. area/bounds loss의 scene-scale normalization도
논문 수식과 현재 비정규화 좌표계의 차이를 명시한다.

### 12. emitter adaptation optimizer state

**2026-08-13 구현 완료.** split/prune 결과가 새 행의 old-emitter provenance와 child
여부를 반환한다. unsplit survivor의 raw parameter와 Adam first/second moment를 행 단위로
정확히 보존하고, split child의 moment는 0으로 초기화한다. Adam의 `step`은 parameter
tensor당 하나인 scalar이므로 survivor 값을 보존하고 child와 공유한다. prune/split 및
adaptation 직후 finite optimizer step 회귀 테스트를 추가했다.

### 13. resume와 reproducibility

**2026-08-13 구현 완료.** trainer checkpoint를 v4로 올리고 geometry/material/emitter
Adam state, 다음 global step, Python/NumPy/CPU/CUDA RNG, 고정 scene bounds와 surface
anchor, 선택 frame identity, emitter adaptation state를 저장한다. `--resume`은 저장된
global step부터 시작하므로 learning-rate/adaptation schedule을 반복하지 않는다. 저장된
config가 기준이며 `--steps`만 최종 목표 step으로 변경한다.

정확한 재개를 위해 DN checkpoint, dataset/prior 경로, frame 목록, visible CUDA device
수가 달라지면 오류로 중단한다. 현재 full-SH schema는 v5이며 v1-v4는
inspection/inference 또는 명시적 warm start만
허용하고 exact resume에서는 거부한다. Python·NumPy·PyTorch 난수와 세 optimizer를
사용하고 중간 emitter split까지 포함한 7-step 연속 실행 대 4+3-step resume 결과가
tensor 단위로 일치했으며 CUDA RNG round-trip도 통과했다.

### 14. evaluation

**2026-08-13 부분 구현.** 현재 `inspect`는 선택한 split에서 동일 invalid mask를 적용한
PSNR, SSIM, spatial LPIPS를 계산한다. 평가 seed와 sample 수를 metadata에 저장하고,
float32 raw RGB/linear HDR 입력은 PNG와 별도 보존한다. train/evaluation split overlap은
오류로 차단하며 split provenance도 결과에 기록한다. 2026-08-13에
FIPT-Synthetic/HyperSim/Isaac의 기존 train을 보존한 채 공동 held-out pool을 원본
순서에서 교대로 배정하여 독립 val/test로 분할했다. 기본·파생 26개 metadata와
converter를 동기화하고 교집합 0, 전체 frame 완전 분할 및 idempotency를 검증했다.

최종 논문 비교를 위해 다음은 아직 필요하다.

- FIPT-Real held-out metadata 추가
- inference에서는 SH proxy를 완전히 비활성화
- 색공간 규약 확정 후 동일 checkpoint CUDA 재평가의 수치 동등성 검증

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
- [완료] 기본 설정의 `256 x (128 + 128) = 65,536` transport-ray budget 검증
- [완료] checkpoint exact resume 동등성
- held-out masked metrics

## 수정 권장 순서

1. train/val/test split과 mask 처리
2. [완료] 회전 emitter intersection 및 회귀 테스트
3. global ambient 제거
4. scene normalization/visibility threshold 정합
5. [완료] secondary bounce 학습·추론 경로 구현
6. [완료] DiffusionRenderer input/prior/import pipeline 구현 및 bedroom 222-frame 검증
7. [완료] `256 x (128 + 128) = 2^16` transport-ray budget 적용
8. [완료] 색공간/tone mapping 정합
9. [완료] exact resume 구현; evaluation protocol 마무리
10. 전체 CUDA/OptiX smoke test 후 AEGIR 20k 학습

## 작업 원칙

- AEGIR 학습에는 GT camera pose와 invalid-pixel mask를 사용할 수 있다.
- AEGIR 또는 DN-Splatter 학습 supervision에 GT depth는 사용하지 않는다.
- DN-Splatter geometry는 RGB + fixed GT pose + COLMAP point initialization + RGB-derived
  ZoeDepth/PearsonDepth 설정을 사용한다.
- DiffusionRenderer depth/normal은 AEGIR material supervision으로 사용하지 않는다.
- material representation은 per-view trainable G-buffer가 아니라 per-Gaussian BRDF다.
- G-buffer는 매 camera/iteration마다 현재 Gaussian 상태로 즉석 rasterization한다.
