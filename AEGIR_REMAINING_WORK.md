# AEGIR 남은 작업 체크리스트

최초 작성일: 2026-08-13  
작업 루트: `/workspace/hohyun_ws/AEGIR`  
서버 GPU 사용 한도: **작업당 최대 VRAM 24 GB**

이 문서는 AEGIR의 남은 수정사항과 검증 상태를 관리하는 기준 문서다.
아래 항목을 구현할 때 코드 수정, 회귀 테스트, 실행 검증을 함께 수행하고 이
문서의 상태와 검증 기록도 같은 작업에서 갱신한다.

## 상태 표기

- `[ ] 미착수`: 구현을 시작하지 않음
- `[-] 진행 중`: 일부 구현됐지만 완료 조건을 만족하지 않음
- `[x] 완료`: 구현과 필요한 검증을 모두 통과함
- `[!] 보류`: 외부 입력, 사용자 결정 또는 선행 작업이 필요함

완료 표시는 코드가 존재한다는 이유만으로 하지 않는다. 항목별 완료 조건과 테스트,
실제 실행 결과가 모두 충족되어야 한다.

---

## 완료된 기반 작업

### [x] 실제 gradient freeze 및 SH 전환 일정 분리 (2026-09-08)

- LR=0인 optimizer 그룹의 gradient를 끄고 clipping에서 제외한다.
- geometry 고정 중 BVH 반복 재생성과 scene refinement를 생략한다.
  마지막 geometry 업데이트는 첫 frozen step에서 BVH에 반영한다.
- 전체 scene이 고정된 단계의 prior/TV 계산을 생략한다.
- SH fade를 5K–10K로 분리하고 material은 15K까지 학습한다.
- 체크포인트 v12; 이전 버전은 렌더링 호환, exact resume은 v12만 허용.
- 회귀 테스트: clipping 간섭 제거, freeze 이후 emitter 미분 유지,
  regularizer 생략, 독립 일정 저장 및 과거 config 해석 검증.
- 전체 테스트 110개 통과. 실제 Conference CUDA 4-step 검사에서 geometry
  불변, frozen gradient 없음, emitter gradient 유지, BVH 재생성 0회 확인.

### [x] 외부 CUDA 소스 Git 배포 (2026-09-07)

- AEGIR `external/`에 수정된 IRGS tracer와 공식 2DGS rasterizer 및
  빌드에 필요한 헤더를 포함했다. 원본 라이선스와 출처를 유지했다.
- 두 설치 스크립트의 기본 경로를 저장소 내부로 변경했다.
- 현재 서버의 외부 소스와 복사본의 파일별 SHA-256 일치 및 설치 스크립트
  `bash -n` 통과. 생성된 바이너리와 build 디렉터리는 Git에서 제외했다.
- 새 컴퓨터에서 CUDA 확장을 빌드하는 것은 별도 환경 검증 단계이며,
  이번 배포 검증에서 수행한 것으로 간주하지 않는다.

### [x] 작업 경로 relocation

- active runner에서 옛 `/home/tako/...`, `/home/megapose/...` 절대 경로 제거
- local-prefix 환경을 `/workspace/.conda/envs/*`에서 사용
- `AEGIR/.cuda`를 현재 `dnsplatter` CUDA 13 toolkit으로 연결
- `AEGIR/.torch_extensions`를 현재 workspace 전용 cache로 연결
- `cuda_link_stubs/libcudart.so`를 현재 CUDA runtime으로 연결
- `/workspace/hohyun_ws` 아래 broken symlink 제거

검증 기록 (2026-08-13):

- 모든 active shell runner `bash -n` 통과
- Python scripts compile 통과
- COLMAP 기본 경로가 `/workspace/.conda/envs/aegir_colmap/bin/colmap`임을 확인
- AEGIR CPU test `32 passed`

### [x] 기존 핵심 안전 수정

- train/validation/test frame 선택 구현
- dataset invalid mask와 2DGS valid mask 결합
- global ambient 표현 제거
- emitter split에서 부모를 두 자식으로 교체하도록 수정
- rotated anisotropic emitter intersection/radiance 좌표계 수정
- bedroom DiffusionRenderer material prior 222 frame 검증

---

## 우선순위 작업

### 1. [x] 24 GB 기준 CUDA/OptiX smoke test

작업 내용:

- 2DGS rasterizer CUDA forward/backward
- IRGS/OptiX visibility tracing과 gradient
- 작은 Gaussian 수와 낮은 sample 수로 AEGIR 10~100 step 학습
- 단계별 peak allocated/reserved VRAM 측정
- OOM, NaN, Inf 검사

완료 조건:

- 모든 CUDA 경로가 현재 local environment에서 실행됨
- forward와 backward gradient가 finite임
- smoke training이 정상 checkpoint를 생성함
- 최대 사용량이 24 GB 미만이며 측정 명령과 결과가 기록됨
- smoke checkpoint를 final 학습에 resume하지 않음

검증 기록 (2026-08-13):

- 전용 harness: `scripts/smoke_cuda_optix_24gb.py`
- 독립 CUDA 회귀 테스트:
  - `tests/test_shared_gaussians.py::test_2dgs_rasterizer_forward_backward`
  - `tests/test_irgs_tracer.py`
  - 결과: `3 passed`
- 실제 bedroom 입력 통합 smoke:
  - DN checkpoint: ZoeDepth + Pearson step 9,999
  - DiffusionRenderer material prior 사용
  - 512 Gaussians, train view 1개, 32×32 raster
  - 64 primary rays/step
  - BRDF/light sample 각 2개, secondary sample 1개
  - 10 steps 완료, step 10 checkpoint 생성
- 2DGS 결과:
  - valid pixels 150
  - 모든 geometry/material parameter gradient 존재 및 finite
- IRGS/OptiX 결과:
  - hit visibility `0.10000002`, miss visibility `1.0`
  - opacity gradient finite 및 nonzero
- 메모리:
  - PyTorch training peak allocated `0.06399 GiB`
  - PyTorch training peak reserved `0.06641 GiB`
  - `nvidia-smi` per-process peak `804 MiB` (34 samples)
  - 작업 한도 24 GiB 미만
- finite 검사:
  - OOM 없음
  - 저장된 scene/emitter/loss에 NaN/Inf 없음
- 산출물:
  - `outputs/smoke_cuda_optix_24gb_20260813_01/smoke_result.json`
  - `outputs/smoke_cuda_optix_24gb_20260813_01/aegir_smoke.pt`

주의: 위 메모리는 CUDA/OptiX 경로 검증을 위한 축소 설정의 측정값이다. 최종
256 primary point, point당 BRDF/light 각 128 samples 설정의 VRAM을 직접 측정한
결과는 아니다.
smoke checkpoint는 final 학습에 resume하지 않는다. 컨테이너에서 보이는 전체 GPU
VRAM은 약 97.9 GB이며, 24 GB는 서버 운용상 작업 한도로 적용한다.

### 2. [x] Scene scale과 visibility 정합

완료 내용 (2026-08-13):

- scene scale `S`를 Gaussian bounds의 quantile diagonal로 한 번 계산
- 설정의 `visibility_bias=0.001`과 `visibility_steepness=80`은 정규화 좌표 기준값으로 명시
- 실제 tracer/renderer 적용값을 다음처럼 변환:
  - `bias_scene = 0.001 * S`
  - `steepness_scene = 80 / S`
- 예를 들어 `S=6 m`이면 bias `0.006 m`, steepness `13.333... m^-1`
- `S=33 m`이면 bias `0.033 m`, steepness `2.424... m^-1`
- 현재 9,999-step DN checkpoints의 실제 적용값:
  - bedroom: `S=13.2611`, bias `0.013261`, steepness `6.03268`
  - kitchen: `S=14.6656`, bias `0.014666`, steepness `5.45494`
  - livingroom: `S=11.4963`, bias `0.011496`, steepness `6.95877`
- IRGS, inspection, camera-path render, legacy point visibility가 동일한 `S`를 전달받도록 통일
- `trace_transmittance_min`을 논문값 `0.05`로 변경

논문과 비교:

- 논문 Equation 3은
  `1 - alpha_trace * sigmoid(d_light - epsilon - d_trace)`로 bias `epsilon`을 한 번 사용한다.
- 논문 Appendix A.3은 scene을 정규화한 뒤 fixed visibility parameter를 쓰며,
  scene-dependent steepness도 동등한 구현이라고 명시한다.
- 현재 코드는 scene 전체 좌표를 재작성하지 않고, 위 `S` 비례/역비례 변환으로 같은 효과를 낸다.
- 논문은 bias와 steepness의 정확한 숫자를 공개하지 않는다. 따라서 `0.001`과 `80`은
  논문 수치라고 표기하지 않고 기존 구현의 normalized-coordinate 기준값으로 유지한다.
- 논문 Appendix C의 ray-marching 종료 transmittance `0.05`는 정확히 일치시켰다.
- 참고: https://arxiv.org/html/2606.28635v2

Bias 중복 처리 수정:

- 기존에는 같은 `visibility_bias`가 ray origin 이동과 거리식에 함께 나타나 용도가 섞여 있었다.
- self-hit 방지용 ray origin offset은 별도의 수치 상수 `1e-6 * S`로 분리했다.
- OptiX trace depth를 원래 surface-origin 거리로 되돌린 후, Equation 3의
  `visibility_bias`를 `soft_visibility()`에서 정확히 한 번만 차감한다.
- BVH surfel thickness도 `1e-6 * S`로 맞춰 좌표 scale에 독립적으로 만들었다.

검증 기록 (2026-08-13):

- CPU scale/bias/transmittance 단위 테스트 추가: `tests/test_visibility_scale.py`
- CUDA IRGS scale 회귀 테스트 추가:
  `tests/test_irgs_tracer.py::test_irgs_visibility_and_opacity_gradient_are_scale_invariant`
- 1배/10배 동등 scene에서 visibility와 opacity gradient 일치
- 전체 테스트: `38 passed`
- CUDA/OptiX 전용 테스트: `3 passed`
- 24 GB 제한 bedroom 10-step 통합 smoke 재실행: 통과
  - training peak allocated `0.06397 GiB`
  - training peak reserved `0.06641 GiB`
  - `nvidia-smi` per-process peak `804 MiB`
  - OOM/NaN/Inf 없음, step 10 checkpoint finite
  - 결과: `outputs/smoke_cuda_optix_24gb_visibility_scale_20260813/smoke_result.json`

범위 메모: 위치 learning rate의 meter 단위 정합은 사용자가 보류한 학습 관련 수정에
포함한다. emitter regularization의 area/bounds 항은 scene diagonal로 정규화되어 있다.
2026-08-14 감사에서 area 항이 논문 Eq.19의 squared norm이 아니라 first-power norm으로
구현된 것을 발견해 `mean(sum((scale / scene_diagonal)^2))`로 정정했다.

### 3. [x] Secondary bounce를 논문 방식으로 재구현

완료 내용 (2026-08-14):

- DN-Splatter의 DC+rest 16개 full SH coefficient를 frozen training buffer로 보존
- 기존 primary OptiX shadow trace가 이미 계산하는 SH color/normal/depth/alpha를 재사용;
  별도 sphere proxy trace를 추가하지 않음
- 각 BRDF/light sample에서 논문 Equation 4의
  `V * L_direct + (1-V) * L_bounce`를 직접 적용
- training은 first secondary Gaussian trace의 opacity-composited full-SH radiance에서
  differentiable explicit-emitter NEE로 5K--15K 동안 교차 전환
- explicit estimator는 first hit의 normal 및 learned albedo/roughness/metallic을 사용해
  emitter surface를 직접 sampling하고 IRGS visibility를 다시 계산하므로 작은 emitter를
  무작위 hemisphere ray가 우연히 맞혀야 하는 high-miss 문제가 없음
- inference는 `training_proxy=False`로 SH contribution을 강제로 0으로 만들고 동일한
  explicit-emitter NEE만 사용; 재귀적인 multi-bounce는 수행하지 않음
- `bounce_strength=0.25`, 1,024-center proxy, sphere radius/max-distance heuristic 제거
- 논문은 secondary sample 수를 별도 수치로 명시하지 않으므로 기본 1개로 두고
  `num_bounce_samples`를 train/inference variance-control 설정으로 유지
- explicit NEE cross-fade가 추가된 checkpoint schema를 v7로 올리고 v1-v6 exact resume은 거부;
  legacy checkpoint inference는 저장된 DC proxy를 degree-0 SH로 변환해 계속 로드

검증 기록 (2026-08-14):

- 수식 회귀: visibility 0.25에서 direct가 정확히 0.25배가 되고 weighted SH bounce가
  임의 strength 없이 그대로 결합됨
- inference callback이 SH를 요청하지 않으며 emitter miss=0, explicit-emitter hit>0 검증
- CUDA/OptiX full-SH 방향성, inference SH=0, first-hit material/point finite,
  opacity gradient finite/nonzero 검증
- 전체 test suite: `75 passed`
- bedroom 512 Gaussian, 32x32, 2-step 통합 smoke 통과
  - v5 checkpoint 및 모든 저장 tensor finite
  - `nvidia-smi` process peak `760 MiB`
  - 결과: `outputs/smoke_secondary_bounce_v5_20260814/smoke_result.json`
- 별도 inference smoke: SH `(512,16,3)`, degree 3, HDR/direct/indirect 모두 finite,
  indirect HDR nonzero, `HDR = direct + indirect` 일치

#### Conference direct-light 회귀 수정 (2026-08-14)

- 기존 conference 20k checkpoint의 frame 0을 분해하면 training HDR의 96.5%를 frozen
  DN SH proxy가 설명했고, inference에서 남는 learned direct HDR mean은 target의 약
  3.5%뿐이었다. 따라서 SuperSplat albedo geometry는 정상이어도 physical render가
  검게 보였다.
- 공식 IRGS의 `light_t_min=0.05`와 동일하게 shadow ray origin을 surface에서 `0.05 m`
  이동한다. 이는 light endpoint의 scene-scale `visibility_bias`와 별개이며 그 bias는
  계속 거리식에서 한 번만 차감한다.
- ray origin offset과 2D Gaussian BVH proxy 두께를 분리했다. proxy 두께는 기존의
  `scene_scale * 1e-6`로 유지되어 surfel geometry를 5 cm 두께로 부풀리지 않는다.
- frozen SH proxy는 step 5,000까지 weight 1, step 5,000--15,000에서 선형으로 0까지
  감소한다. 동시에 explicit secondary NEE weight를 0에서 1까지 올리므로 간접광 자체가
  사라지지 않으며 마지막 5,000 step은 explicit NEE만 사용한다. 이 cross-fade는 논문에
  명시된 별도 schedule이 아니라 training-only SH shortcut을 막기 위한 구현 안정화다.
  checkpoint의 global step에서 결정되므로 exact resume에도 결정적이다.
- loss history에 `sh_proxy_weight`, `explicit_bounce_weight`, `direct_hdr_mean`,
  `indirect_hdr_mean`을 기록해 같은
  실패가 재발하는지 학습 도중 확인할 수 있게 했다.
- 관련 CPU/CUDA/OptiX 회귀 및 전체 suite: `81 passed`.

#### Explicit secondary NEE 교차 전환 보완 (2026-08-18)

- 기존 fade-out은 마지막 5K에서 SH proxy만 0으로 만들고 training explicit bounce를
  계산하지 않아 indirect가 정확히 0이 되는 결함이 있었다.
- `SH * w + explicit_NEE * (1-w)`로 교체하고, IRGS 한 번의 primary trace에서 full SH와
  learned hit material을 동시에 반환하도록 glue contract를 변경했다.
- CPU 전체 suite `90 passed`; 24,000 MiB per-process cap에서 실제 CUDA/OptiX
  first-hit → emitter NEE → backward 회귀 `1 passed`. emitter color/scale/center gradient가
  finite/nonzero이며, 작은 emitter도 무작위 ray-intersection 없이 nonzero contribution을 냄.
- 새 학습 checkpoint는 v7. 기존 v5/v6은 inspect/inference는 가능하지만 목적함수가
  달라졌으므로 exact resume은 거부한다.
- 최초 실학습에서 5K 이후 secondary ray를 256개씩 최대 256회 OptiX 호출하여
  약 0.08초/step에서 2.7--3.2초/step으로 느려지는 launch 병목을 확인했다. 기본
  `bounce_chunk_size`를 65,536으로 올리고, 계산 의미를 바꾸지 않는 실행 전용 옵션으로
  분리해 v7 exact resume에서도 `--bounce-chunk-size 65536` override를 허용한다.
  emitter sample을 chunk loop 밖에서 한 번 생성해 chunk 크기 1과 65,536의 출력도
  bitwise 동일하게 고정했다. conference step-5000 v7 checkpoint의 24 GB 제한 5-step
  resume smoke는 통과했고, warm-up 이후 약 0.13초/step, PyTorch peak reserved 700 MiB;
  전체 suite `90 passed`.

### 4. [x] `2^16 = 65,536` total transport rays/iteration 정합

정정된 해석:

- 논문의 `2^16 rays`는 primary pixel 65,536개가 아니라 iteration당 총 transport ray budget
- primary point마다 BRDF 128개와 emitter 128개, 합계 256개 transport ray를 사용
- 따라서 `256 primary x (128 + 128 samples) = 65,536 transport rays`
- 로컬 `rays_per_step`/`--rays-per-step`은 이름과 달리 primary surface point 수
- explicit NEE가 활성화되는 5K 이후에는 valid first hit마다 기본 1개의 추가
  surface-to-emitter visibility segment가 생기므로 OptiX segment 수는 최대 131,072/step;
  논문의 `2^16` primary sampling budget 자체는 그대로 유지

완료 내용 (2026-08-14):

- 기본 config와 CLI를 256 primary point로 변경
- production queue와 개별 HyperSim/Isaac 실행 스크립트를 256으로 변경
- 기본 128×128 raster에서 16×16 patch 선택
- 단위 테스트에서 `256 x (128 + 128) = 65,536`을 명시적으로 검증
- checkpoint/CLI 호환성을 위해 기존 필드 이름은 유지하되 의미를 코드와 help에 명시

검증 및 대형 설정 비교:

- 과거 1,024-primary production trial은 이미 더 큰
  `1,024 x 256 = 262,144` transport ray 설정에서 GPU peak 3,458 MiB 이하였다.
- 의도적인 65,536-primary stress test는
  `65,536 x 256 = 16,777,216` transport rays/step이며, 24,000 MiB PyTorch cap에서
  약 15.4초/iteration, 20k ETA 약 85.6시간으로 측정됐다. 이는 논문 설정이 아니다.
- exact resume은 checkpoint에 저장된 config가 우선이므로 기존 1,024-primary
  checkpoint를 resume하면 자동으로 256으로 바뀌지 않는다.

### 5. [x] 색공간과 tone mapping 통일

완료 내용 (2026-08-13):

- frame 전체에 HDR이 있으면 PNG가 아니라 non-negative scene-linear HDR을 기본
  photometric target으로 사용하고, symmetric `log1p(HDR)` L1 + matched display SSIM 적용
- HDR이 없는 scene은 `aegir_color.target_png`의 exposure/tone mapper/transfer function을
  train과 eval에서 동일하게 적용
- LDR camera-response metadata가 없으면 scene-global exposure를 학습하고
  `Reinhard + sRGB`를 쓰는 명시적 fallback 적용
- clipped LDR highlight의 L1은 overprediction을 벌하지 않는 one-sided censored loss 적용
- DiffusionRenderer basecolor importer 기본 입력을 sRGB로 명시하고 linear reflectance로
  decode; material prior와 BRDF albedo는 항상 linear reflectance로 유지
- HDR metric display image는 prediction/GT 양쪽에 동일한 Reinhard+sRGB를 적용하고,
  raw float32 linear HDR와 metric-domain 배열은 visualization PNG와 별도 저장
- 현재 변환 데이터 30개 `transforms.json`도 13개 linear-HDR / 17개 LDR fallback으로 주석화
- 새 converter는 HDR encoding과 알려진 PNG mapping을 기록하며, RGB-only 파생본은
  HDR metadata를 그대로 남기지 않고 정확 LDR 또는 fallback policy로 전환

완료 조건:

- 데이터, renderer, prior 각각의 색공간이 config/metadata/code에 명시됨
- sRGB↔linear 변환 unit test 통과
- train/eval 출력 convention이 동일함
- metric용 결과와 visualization용 tone-mapped 결과가 구분됨

검증 기록 (2026-08-13):

- sRGB 기준값 및 sRGB↔linear round-trip
- metadata exact per-frame exposure/gamma 경로와 unknown-LDR fallback
- mixed HDR/LDR scene 거부, HDR symmetric loss, clipped-LDR one-sided loss
- converter metadata 및 DiffusionRenderer sRGB→linear import 회귀 테스트
- 전체 CPU test suite: `68 passed`

### 6. [x] Focus map을 full view에서 계산

구현 내용 (2026-08-13):

- 전체 target view와 `dataset mask & raster valid mask`에서 7x7 local contrast와
  view-global `max(delta)` normalization을 먼저 계산
- target/G-buffer/prior/mask를 선택할 때 동일 좌표의 precomputed focus weight도 함께 crop
- `photometric_loss()`가 precomputed weight를 받아 L1에 적용하도록 확장하되, 독립 호출의
  기존 내부 계산 fallback은 유지
- invalid pixel은 local average와 normalization maximum 및 최종 masked loss에서 제외

검증 기록 (2026-08-13):

- 서로 다른 위치와 크기의 두 crop에서 겹치는 source pixel의 weight가 bit-exact 일치
- invalid 영역 RGB를 `1e4`로 변경해도 모든 valid pixel focus weight가 bit-exact 유지
- precomputed full-view focus crop이 photometric masked L1에 그대로 적용되는지 수식 대조
- 당시 전체 test suite: `59 passed`; 변경 파일 Ruff 검사 통과

### 7. [x] 논문 외 emitter regularizer 기본 비활성화

수정 전 기본값:

- `light_surface=10.0`
- `light_view=1.0`

2026-08-13 수정 완료:

- 논문 재현 기본 config와 CLI 기본값에서 두 값을 모두 `0.0`으로 설정
- surface/view 계산 기능은 삭제하지 않고 ablation 및 안정화 fallback으로 보존
- `--light-surface`, `--light-view`를 명시한 경우에만 활성화
- 논문의 area/white/bounds regularization과 각각의 회귀 테스트 유지
- emitter가 복원 표면에서 과도하게 떠다니는 현상이 관찰되면 우선
  `--light-surface 10.0`을 다시 적용하고, 모든 관측 frustum 밖으로 이탈하는 현상까지
  있을 때만 `--light-view 1.0`도 검토

완료 조건:

- 기본 config에서 surface/view regularizer contribution이 정확히 0
- CLI 또는 config override 시에만 활성화됨
- 기존 area/white/bounds test가 유지됨

검증 기록 (2026-08-13):

- 신규 emitter regularization 회귀 테스트: `6 passed`
- 전체 CPU test suite: `44 passed`
- 변경 파일 Ruff 검사: 통과

### 8. [x] Emitter adaptation에서 Adam state 보존

수정 전 문제:

- split/prune 후 emitter optimizer 전체를 재생성
- 살아남은 emitter의 `exp_avg`, `exp_avg_sq`, `step`도 손실

2026-08-13 수정 완료:

- `EmitterAdaptationResult`가 `new_to_old`, `child_mask`, prune index를 반환
- 모든 unsplit survivor의 raw parameter 행과 Adam `exp_avg`, `exp_avg_sq`를 정확히 복사
- Adam의 scalar `step`과 parameter-group hyperparameter를 보존
- split child 정책: parameter 값은 split 수식으로 생성하고 per-row moment tensor는 0으로 초기화
- standard Adam은 parameter tensor당 scalar `step` 하나만 가지므로 child는 survivor와 tensor-level
  `step`을 공유함. 완전한 per-child step=0은 custom row-wise optimizer 없이는 불가능
- quaternion normalize/re-encode로 momentum 좌표가 바뀌지 않도록 unsplit survivor의
  `raw_quaternions`를 포함한 raw parameter를 재생성 전 값으로 복원

완료 조건:

- prune 후 survivor state가 정확히 보존됨
- split 후 survivor와 child state가 정의된 정책과 일치
- adaptation 전후 optimizer step이 finite
- 회귀 unit test 통과

검증 기록 (2026-08-13):

- prune 후 6개 emitter parameter의 survivor raw 값/moment/step 정확 보존 테스트 통과
- split 후 survivor state 보존 및 두 child moment=0 정책 테스트 통과
- adaptation 이후 backward/Adam step과 모든 parameter/state finite 확인
- 전체 CPU test suite: `53 passed`; 변경 파일 Ruff 검사 통과

### 9. [x] Exact checkpoint resume

구현 내용 (2026-08-13):

- trainer checkpoint schema를 `aegir_shared_gaussian_v5`로 변경
- geometry/material/emitter Adam state와 raw emitter parameter 저장
- Python, NumPy, PyTorch CPU 및 전체 visible CUDA RNG state 저장·복원
- `next_step`, 고정 scene bounds/surface points, frame index/file identity,
  emitter adaptation counter 저장·복원
- CLI `train-gaussian-scene --resume` 구현; `--steps`는 추가 횟수가 아니라 최종
  global-step 목표
- checkpoint config를 기준으로 사용하며 이미 지난 learning-rate/adaptation schedule은
  반복하지 않음
- DN checkpoint, dataset/prior 경로, frame identity, visible CUDA device 수 불일치는
  exactness를 보장할 수 없으므로 hard error
- v1-v4 checkpoint는 inspection/inference만 지원하고 exact resume에서는 명시적으로 거부
- `--emitter-checkpoint`는 emitter-only warm start로 유지하며 `--resume`과 동시 사용 불가

검증 기록 (2026-08-13):

- 실제 Gaussian/material/emitter module과 세 Adam optimizer를 사용한 7-step 연속 실행과
  4-step checkpoint + 3-step resume의 loss, raw parameter, optimizer state, RNG가 정확히 일치
- 위 결정성 테스트에 Python/NumPy/PyTorch 난수와 step 2 emitter split/Adam state migration 포함
- raw emitter quaternion checkpoint round-trip 정확 일치
- CUDA RNG capture/restore round-trip 통과
- legacy v3 load는 유지하면서 exact resume 거부 확인
- CLI `--resume`/`--emitter-checkpoint` 상호배타성 및 final-step 의미 확인
- 전체 test suite: `57 passed`; 변경 파일 Ruff 검사 통과

### 10. [ ] Held-out 평가 파이프라인

현재 상태:

- 2026-08-13: dataset mask를 동일하게 적용하는 PSNR/SSIM/spatial LPIPS 구현
- 기본 평가 sample 수 128, seed 2026 및 frame별 seed 규칙 구현
- float32 raw RGB/linear HDR `.npz`와 visualization PNG 출력 분리
- `metrics.json`에 sample 수, seed, LPIPS network, metric domain, split 검증 결과 저장
- train/evaluation frame overlap은 hard error로 차단
- synthetic/HyperSim/Isaac 11 scene의 기존 train은 그대로 보존하고, 공동 held-out
  pool을 원본 순서에서 교대로 배정하여 독립 val/test로 분할 완료
  - 기본 11개와 RGB-pose-mask 파생본 15개, 총 26개 metadata를 동기화
  - 원래 공동 held-out 목록과 deterministic policy를 `aegir_evaluation_split`에 저장
- FIPT-Real에는 held-out split metadata가 없음

수정 방향:

- PSNR, SSIM, LPIPS 구현
- fixed evaluation seed와 충분한 Monte Carlo sample 수 사용
- raw linear/HDR metric 결과와 visualization PNG 분리
- inference에서 SH fallback 비활성화
- 독립적인 val/test protocol 및 FIPT-Real split 결정

완료 조건:

- train frame이 held-out metric에 포함되지 않음
- invalid mask가 모든 metric에 동일하게 적용됨
- seed와 sampling 설정이 결과 metadata에 저장됨
- 동일 checkpoint 재평가 결과가 재현됨

검증 기록 (2026-08-13, 부분 완료):

- metric mask/raw-output/fixed-generator unit test 통과
- 실제 pretrained AlexNet spatial LPIPS CPU smoke test: finite score 확인
- 전체 CPU test suite: `51 passed`; 이번 변경 파일 Ruff 검사 통과
- synthetic/HyperSim/Isaac 기본·파생 metadata 26개에서 train/val/test 교집합 0,
  전체 frame 완전 분할, 기본본-파생본 frame 목록 일치, 재실행 idempotency 검증
- converter도 동일 deterministic 독립 분할을 생성하도록 변경
- 현재 전체 CPU test suite: `72 passed`
- 남은 항목: FIPT-Real split, inference SH fallback 제거, 새 색공간 규약으로
  실제 checkpoint CUDA 재평가 재현성 검증

### 11. [ ] 나머지 scene 입력 준비

현재 완료된 입력 구성요소:

- FIPT-Synthetic 4 scene의 material prior 부분 완료
  - bedroom 222, bathroom 122, kitchen 217, livingroom 227 frames
  - 총 788개 prior를 직접 로드하여 albedo/roughness/metallic 존재, frame identity,
    finite `[0,1]` 값, GT depth/normal field 부재를 strict validation함
  - 기존 prior는 legacy `basecolor_space=model` metadata이며, loader가 이를 sRGB source로
    해석해 linear reflectance로 decode함. 새 importer는 이를 명시적으로 기록함.
- 새 importer 기준 material prior 추가 완료
  - FIPT-Real classroom 278, HyperSim ai_001_004 100, Isaac warehouse 102 frames
  - `prior_albedo_space=linear_reflectance`, GT depth/normal 미사용 strict validation 통과
- geometry 입력은 총 13 scene에서 완료
  - FIPT-Synthetic 4, FIPT-Real 2, HyperSim 6, Isaac Sim 1
  - fixed-pose COLMAP, RGB-derived raw ZoeDepth, PearsonDepth DN-Splatter,
    SuperSplat PLY가 모두 존재하고 교차 검증됨
- bathroom은 legacy step-12000(2,417,989 Gaussians)을 보존하고, 현재 경로에서
  fresh exact step-9999를 별도 재학습함
  - exact-10k checkpoint/PLY: 2,154,026 Gaussians, finite 검증 통과
  - 10.9% 감소했지만 여전히 과도하게 많아 canonical 결과로 아직 승격하지 않음

남은 작업:

- material prior: FIPT-Real conference 및 나머지 HyperSim 5 scene
- FIPT-Real held-out split
- bathroom Gaussian 수를 그대로 사용할지 densification/culling을 조정해 다시 만들지 결정

완료 조건:

- scene별 RGB, pose, intrinsics, optional mask, sparse PLY 경로 검증
- DN config가 RGB-derived ZoeDepth + Pearson이며 GT depth/normal을 사용하지 않음
- camera optimizer가 off이고 supplied pose가 고정됨
- 모든 frame에 material prior 3종이 존재하고 geometry prior field가 없음

검증 기록 (2026-08-13):

- 13/13 scene의 checkpoint 내부 step, config, tensor finite 여부 검증
- 13/13 PLY header/property 및 checkpoint Gaussian 수와 vertex 수 일치
- 2,137 ZoeDepth frame 전체를 다시 읽어 shape, finite, positive, symlink 정합 검증
- supplied pose의 rotation/translation 보존과 COLMAP points3D 검증
- RGB-pose-mask-Zoe 파생 transforms에 GT depth/HDR/normal/material 경로가 없음을 검증
- FIPT-Synthetic 4 scene material prior 총 788 frame strict validation 통과
- classroom/HyperSim ai_001_004/Isaac prior 총 480 frame strict validation 통과
- 전체 13 scene 중 material prior까지 포함된 scene은 7개

현 코드 기준 20k production trial (2026-08-13 실행, 2026-08-14 검증):

- bedroom: 208 train views, 391,558 Gaussians, 2 emitters
- classroom: 278 train views, 228,879 Gaussians, 6 emitters
- HyperSim ai_001_004: 88 train views, 356,342 Gaussians, 3 emitters
- Isaac warehouse: 90 train views, 531,216 Gaussians, 1 emitter
- 4/4 checkpoint step/next-step 20,000, optimizer/exact-resume state 존재, 모든 tensor finite
- GPU stage peak: AEGIR 2,038--3,458 MiB, DiffusionRenderer 최대 14,058 MiB
- 이 결과는 1,024 primary point, 즉 262,144 direct transport rays와 현
  secondary-bounce proxy를 사용한 trial이며 최종 논문 결과가 아님

### 12. [x] 정리 및 재현성 보강

완료 내용 (2026-08-13):

- `SERVER_HANDOFF.txt`를 현재 `/workspace` 기준 authoritative 문서로 교체
- `docs/SERVER_ENVIRONMENT.md`를 Git-tracked 환경 문서로 만들고 상위
  `ENVIRONMENT_SETUP_SERVER.txt`는 해당 문서로 연결
- 옛 `/home/tako`, `/home/megapose`, `/home/hohyun`, `AEGIR_old` 문서는
  `docs/archive/`로 이동하고 실행 금지 표시
- AEGIR 자체 Git repository를 `main` branch로 초기화하고 source만 추적
- outputs/logs/exports/CUDA cache/packed env는 `.gitignore`로 제외
- 외부 repo upstream/local commit과 patch 목적을 `docs/SOURCE_PROVENANCE.md`에 기록
- DiffusionRenderer와 DN-Splatter patch를 각 repo의
  `aegir-local-20260813` branch에 commit하여 dirty source 제거
- IRGS의 untracked OptiX build/wheel은 repo-local exclude로 분류
- 세 packed env의 executable, `sys.prefix`, scripts path가 현재 prefix임을 확인
- legacy/smoke/DN checkpoint 사용 범위를 `docs/LEGACY_CHECKPOINTS.md`와
  `docs/checkpoint_registry.json`에 기록

검증 기록:

- Git `2.43.0` 설치
- 2D Gaussian Splatting clean: `335ad612f2e783a4e57b9cbc4d1e167bd599fc98`
- IRGS tracked source clean: `1461869c0da84f26aebd8bb6d1caaa8a7bd91326`
- DiffusionRenderer local clean commit: `d088eb095babeeac3cda64542e29548bd61ed568`
- DN-Splatter local clean commit: `91b1c047ed4a3fa2910ac8aff17103baee002fe7`
- packed env에서 active old-prefix 파일 없음; `conda-unpack` 자체와
  `conda-meta/history`의 역사 기록만 남음
- 2026-08-06 AEGIR 결과는 `legacy_visualization_only`, 2026-08-13 smoke
  결과는 `smoke_only_never_resume`, verified Zoe+Pearson DN checkpoint는
  `production_valid_dn_initialization`으로 분류

주의: 컨테이너 교체 시 apt로 설치한 Git/tmux는 다시 설치해야 할 수 있다.

---

## 권장 구현 순서

1. 24 GB 측정이 가능한 최소 CUDA/OptiX smoke harness
2. [완료] Secondary bounce
3. [완료] 65,536 total transport-ray budget 정합
4. [완료] 색공간 및 focus map
5. emitter regularizer와 optimizer state
6. [완료] exact resume
7. evaluation protocol
8. 나머지 scene 입력 준비
9. 최종 20k 학습과 held-out 평가

최종 장시간 학습은 위 P0 성격 작업과 24 GB GPU smoke 검증이 끝난 뒤 사용자에게
검증된 실행 명령을 먼저 제공하고, 명시적인 실행 요청을 받은 경우에만 시작한다.

## 변경 이력

- 2026-08-18: conference DN-style run이 step 5,001의 explicit-secondary 전환에서
  IRGS illegal memory access로 중단된 원인을 동적 scene row 수와 tracer `dummy_sh`
  row 수 불일치로 확정. tracer는 초기 197,483 rows의 dummy SH를 유지한 반면 scene은
  split 후 500,000 rows였고, native OptiX가 새 Gaussian id로 옛 buffer를 bounds check
  없이 읽었다. `refresh_bvh()`가 scene row 수 변경 시 dummy SH를 재생성하도록 수정하고
  split 뒤 새 Gaussian을 실제 trace/backward하는 CUDA 회귀 테스트를 추가. 5K exact
  checkpoint에서 blocking/non-blocking 2-step 및 non-blocking 100-step 진단을 통과했고,
  최종 CPU `92 passed`, IRGS CUDA `10 passed`.
- 2026-08-18: conference v10의 검은 geometry hole을 scene alpha로 확정. 기존
  hard culling이 197,483개 중 57,551개를 삭제했고, 그중 약 50,000개가 누적
  screen-size 조건이었다. 같은 frame 0에서 no-cull alpha `<0.01`은 0%였지만
  hard-cull 결과는 9.3%였다. DN-Splatter의 large-cull predicate만 옮기고 함께
  동작하는 split/duplicate densification을 누락한 것이 원인.
- 2026-08-18: hard world/screen-size deletion을 제거하고 DN-style scene refinement로
  교체. full-view coverage 뒤 screen-space gradient가 높은 큰 2D Gaussian은 tangent
  plane에서 sampling한 2개 child로 교체하고 scale을 1.6으로 축소하며, 작은 고-gradient
  Gaussian은 duplicate한다. 무대체 삭제는 opacity `<0.005`에만 적용한다. survivor의
  geometry/material Adam row는 보존하고 child moment는 0으로 초기화하며, 최대 500,000개
  cap과 step 15,000 stop을 둔다. checkpoint schema는 v8; v1-v7 exact resume은 거부.
  CPU suite `91 passed`, CUDA 2DGS/IRGS `10 passed`, 24 GB 제한 512-Gaussian 4-step
  split smoke에서 8 parents→16 children, final 520 Gaussians, finite v8 checkpoint,
  peak PyTorch reserved 68 MiB를 확인.
- 2026-08-18: conference TSDF 진단에서 확인된 scene floater를 억제하기 위해
  AEGIR geometry 학습에 DN-Splatter `cull_gaussians()`와 동일한 기준을 기본 적용.
  500-step warm-up 이후 100-step마다 opacity `<0.1`, step 3000 이후 tangent scale
  `>0.5`, step 3000--4000에는 누적 screen-size ratio `>0.15`를 제거한다.
  geometry/material/frozen-SH row와 두 Adam state를 함께 이관하고 누적 screen-size 및
  culling counters를 exact-resume v6 checkpoint에 저장한다. 기존 v5 resume은 과거
  동작 보존을 위해 scene culling을 자동 비활성화한다. 이 hard-cull 방식은 위의
  2026-08-18 v8 split/duplicate refinement로 대체되었다.
- 2026-08-15: emitter 형상 과대화 수정. PCA center/scale/axis/color fitting을 같은
  DBSCAN component의 모든 밝은 점이 아니라 multi-view persistent 점으로 제한하고,
  초기 최소 scale 반지름을 scene diagonal의 `0.02`에서 `0.002`로 낮춤. Appendix C의
  adaptive split threshold도 bounds diagonal이 아닌 bbox 최대 축 길이의 20%로 정정.
  shape/aspect-ratio regularizer는 논문 외 항이므로 추가하지 않음. 전체 test `84 passed`.
  실제 conference 191-view dry initialization은 emitter 2개, radius scales
  `[3.646, 0.271, 0.057]`, `[2.463, 0.430, 0.077]`, anisotropy `63.5`, `32.1`로
  얇은 축을 유지했으며 checkpoint나 기존 결과는 변경하지 않음.
- 2026-08-14: 빠른 reconstruction 확인을 위해 inspect에 명시적
  `--training-sh-proxy` 진단 옵션 추가. 이 모드는 expensive explicit secondary
  emitter trace 대신 frozen DN full-SH를 사용하며 metrics metadata에
  `training_sh_proxy_used=true`, `paper_inference_bounce_ready=false`를 기록해 정식
  physical inference 평가와 혼동하지 않도록 함. conference v7 전체 191-frame SH
  inspect를 완료하고 PNG/NPZ/metric row 각 191개를 검증.
- 2026-08-14: conference black-render 원인을 direct-light under-training으로 확인.
  IRGS 공식 5 cm origin offset을 BVH proxy 두께와 분리해 적용하고, frozen DN SH
  training proxy를 5k--15k에 fade-out하여 마지막 5k가 emitter direct transport를
  학습하도록 수정. direct/indirect/proxy weight 진단값을 checkpoint loss history에 추가.
- 2026-08-14: emitter area regularizer를 논문 Eq.19과 정합하도록 수정. source-unit scene의
  각 scale 축을 bounds diagonal로 정규화한 뒤 squared L2 norm을 계산한다. 기존
  first-power norm이 작은 emitter를 과도하게 계속 축소하던 문제를 제거하고, 정확한
  수식값 및 scene 10배 scale 불변성 회귀 테스트를 추가.
- 2026-08-14: 항목 3 완료. frozen DN full SH를 기존 IRGS primary trace에서 조회하고
  Equation 4로 direct/bounce를 결합. inference SH를 차단하고 first hit에서 explicit
  emitter로 one-bounce tracing. v5 checkpoint, CUDA gradient 및 2-step smoke 검증 완료.
- 2026-08-14: 항목 4의 ray 해석을 정정하고 완료 처리. 65,536은 primary 수가 아니라
  총 transport ray budget이며, 기본 config/CLI/run script를
  `256 primary x (128 BRDF + 128 light) = 65,536`으로 통일. 65,536-primary stress
  test의 약 15.4초/iteration 및 85.6시간 ETA도 비교 기록.
- 2026-08-13: 항목 10의 Synthetic/HyperSim/Isaac 독립 val/test metadata 완료.
  기존 train을 보존하고 공동 held-out만 deterministic 50:50으로 분할했으며 기본·파생
  26개 metadata와 세 converter를 동기화하고 누수/완전성/idempotency를 검증.
- 2026-08-13: 항목 5 완료. HDR 우선 log-domain 학습, metadata 기반 LDR camera
  pipeline, unknown-LDR learned exposure fallback, linear albedo prior 및 raw/visualization
  출력 분리를 구현하고 기존 30개 scene color metadata를 주석화.
- 2026-08-13: 항목 6 완료. combined valid mask 기준 full-view focus map을 먼저 계산하고
  patch와 동일 좌표로 crop하도록 수정했으며 crop 불변성과 invalid 경계를 검증.
- 2026-08-13: 항목 9 완료. checkpoint v4 optimizer/RNG/global-step/adaptation 상태와
  strict `--resume`을 구현하고 연속 실행 대 중단/재개의 결정성 및 CUDA RNG를 검증.
- 2026-08-13: 항목 12 완료. 현재 경로 문서, AEGIR Git 기준, 외부 repo local
  commits, packed env prefix 검증, legacy checkpoint registry를 확정.
- 2026-08-13: 항목 2 완료. scene-diagonal 기반 bias/steepness 변환, bias 1회 차감,
  논문 transmittance `0.05`를 적용하고 CPU/CUDA/24 GB smoke 검증 기록.
- 2026-08-13: 항목 1 완료. 독립 CUDA test 3개와 bedroom 10-step 통합 smoke 통과,
  peak VRAM 및 finite checkpoint 검증 결과 기록.
- 2026-08-13: 최초 작성. 경로 relocation 완료 상태와 남은 12개 작업을 기록.
