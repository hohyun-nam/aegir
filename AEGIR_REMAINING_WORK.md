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
65,536 primary rays, ray당 BRDF/light 128 samples 설정의 VRAM을 의미하지 않는다.
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
포함한다. emitter regularization의 area/bounds 항은 이미 scene diagonal로 정규화되어 있다.

### 3. [ ] Secondary bounce를 논문 방식으로 재구현

현재 근사:

- 최대 1,024개 Gaussian center를 sphere proxy처럼 사용
- full SH가 아니라 DC 색만 사용
- secondary cosine sample 4개
- 임의의 `bounce_strength=0.25`
- secondary hit에 IRGS tracer를 사용하지 않음
- inference에서도 SH/DC proxy를 계속 사용

수정 방향:

- 학습 시 secondary ray의 첫 2D Gaussian hit를 trace
- hit Gaussian의 SH/color feature를 training radiance proxy로 query
- inference에서는 SH proxy를 완전히 비활성화
- inference secondary hit에서 recovered explicit emitter까지 다시 trace
- direct/indirect 조합을 논문의 visibility-weighted incident-light 식에 맞춤

완료 조건:

- training SH proxy와 inference emitter transport가 명시적으로 분리됨
- inference 경로에서 `bounce_proxy()`가 호출되지 않음
- first-hit 선택, miss 처리, visibility, gradient 회귀 테스트 통과
- direct/indirect 식과 논문 수식의 대응을 코드에 기록

검증 기록: 미실행

### 4. [ ] `2^16 = 65,536` primary rays/iteration 구현

현재 문제:

- 기본값은 1,024 rays
- 단일 128×128 raster의 pixel 수는 16,384
- 현재 `_factor_patch_shape()`는 ray 수가 image pixel 수보다 크면 오류 발생

수정 방향:

- multi-view/global ray batching 또는 명시적인 replacement sampling 설계
- iteration당 총 primary ray 수 65,536 유지
- BRDF/light transport를 24 GB에 맞는 작은 chunk로 계산
- 필요하면 loss accumulator와 chunk별 backward를 사용해 activation memory 제한
- 총 ray 수와 loss normalization은 chunk 크기와 무관하게 유지

완료 조건:

- CLI `--rays-per-step 65536`이 정상 동작
- 실제 처리한 primary ray 수가 정확히 65,536임을 test에서 확인
- 작은 입력에서 chunked/unchunked forward와 gradient가 허용 오차 내에서 일치
- 128 BRDF + 128 light sample 설정의 peak VRAM이 24 GB 미만
- OOM 시 조용히 설정을 낮추지 않고 명확히 실패함

검증 기록: 미실행

### 5. [ ] 색공간과 tone mapping 통일

현재 문제:

- target PNG를 단순 `[0,1]` 값으로 읽음
- renderer는 linear HDR 뒤 Reinhard tone mapping 적용
- DiffusionRenderer basecolor numeric map의 공간이 명확히 연결되지 않음

수정 방향:

- input RGB가 sRGB인지 linear LDR인지 dataset별로 명시
- 필요한 위치에서 sRGB→linear 변환
- BRDF albedo는 linear reflectance로 유지
- exposure와 tone mapping을 train/eval에서 동일하게 적용
- material prior loss의 색공간 명시

완료 조건:

- 데이터, renderer, prior 각각의 색공간이 config/metadata/code에 명시됨
- sRGB↔linear 변환 unit test 통과
- train/eval 출력 convention이 동일함
- metric용 결과와 visualization용 tone-mapped 결과가 구분됨

검증 기록: 미실행

### 6. [ ] Focus map을 full view에서 계산

현재 문제:

- focus map을 선택된 patch 내부에서 계산
- padding과 patch 경계가 인공 contrast를 만들 수 있음
- patch마다 normalization maximum이 달라짐

수정 방향:

- 전체 target view에서 focus map을 먼저 계산
- 학습 patch 또는 sampled ray와 동일한 위치를 이후 선택

완료 조건:

- 같은 pixel의 focus weight가 patch 위치/크기에 영향을 받지 않음
- invalid mask 경계 처리가 유지됨
- full-view 계산 후 crop 회귀 테스트 통과

검증 기록: 미실행

### 7. [ ] 논문 외 emitter regularizer 기본 비활성화

현재 기본값:

- `light_surface=10.0`
- `light_view=1.0`

수정 방향:

- 논문 재현 기본 config에서는 두 값을 `0.0`으로 설정
- 실험/ablation용 옵션으로 기능 자체는 유지 가능
- 논문의 area/white/bounds regularization은 유지

완료 조건:

- 기본 config에서 surface/view regularizer contribution이 정확히 0
- CLI 또는 config override 시에만 활성화됨
- 기존 area/white/bounds test가 유지됨

검증 기록: 미실행

### 8. [ ] Emitter adaptation에서 Adam state 보존

현재 문제:

- split/prune 후 emitter optimizer 전체를 재생성
- 살아남은 emitter의 `exp_avg`, `exp_avg_sq`, `step`도 손실

수정 방향:

- adaptation 결과가 old emitter index와 new emitter index mapping을 반환
- survivor의 Adam state를 정확히 복사
- 새 child만 새 state로 초기화
- split child state를 0으로 둘지 parent에서 변환할지 정책을 명시

완료 조건:

- prune 후 survivor state가 정확히 보존됨
- split 후 survivor와 child state가 정의된 정책과 일치
- adaptation 전후 optimizer step이 finite
- 회귀 unit test 통과

검증 기록: 미실행

### 9. [ ] Exact checkpoint resume

현재 checkpoint에 없는 상태:

- geometry/material/emitter optimizer state
- CPU/CUDA RNG state
- global step 기반 resume entrypoint
- view/ray sampling state
- emitter adaptation state

수정 방향:

- checkpoint schema를 version-up하고 위 상태를 저장
- CLI `--resume` 또는 동등한 명시적 entrypoint 구현
- legacy checkpoint는 inference만 지원하거나 제한된 warm start임을 명확히 표시

완료 조건:

- 중단 없이 N step 실행한 결과와 K step 저장 후 resume한 결과가 deterministic
  test에서 허용 오차 내 일치
- resume 시 이미 지난 learning-rate/adaptation schedule을 반복하지 않음
- CUDA RNG와 sampling sequence가 복원됨

검증 기록: 미실행

### 10. [ ] Held-out 평가 파이프라인

현재 상태:

- dataset mask를 적용한 PSNR만 계산
- SSIM/LPIPS 없음
- synthetic/HyperSim/Isaac의 val/test filename 목록이 현재 동일
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

검증 기록: 미실행

### 11. [ ] 나머지 scene 입력 준비

현재 완비된 최종 입력:

- FIPT-Synthetic bedroom
  - ZoeDepth + Pearson DN-Splatter checkpoint
  - fixed-pose COLMAP
  - DiffusionRenderer material prior 222 frames

남은 작업:

- DiffusionRenderer prior: bathroom, kitchen, livingroom
- 최종 대상 FIPT-Real/HyperSim/Isaac Sim prior
- DN-Splatter checkpoint: FIPT-Real/HyperSim/Isaac Sim
- 해당 scene fixed-pose COLMAP
- FIPT-Real held-out split

완료 조건:

- scene별 RGB, pose, intrinsics, optional mask, sparse PLY 경로 검증
- DN config가 RGB-derived ZoeDepth + Pearson이며 GT depth/normal을 사용하지 않음
- camera optimizer가 off이고 supplied pose가 고정됨
- 모든 frame에 material prior 3종이 존재하고 geometry prior field가 없음

검증 기록: bedroom만 완료

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
2. Secondary bounce
3. 65,536-ray chunked transport
4. 색공간과 focus map
5. emitter regularizer와 optimizer state
6. exact resume
7. evaluation protocol
8. 나머지 scene 입력 준비
9. 최종 20k 학습과 held-out 평가

최종 장시간 학습은 위 P0 성격 작업과 24 GB GPU smoke 검증이 끝난 뒤 사용자에게
검증된 실행 명령을 먼저 제공하고, 명시적인 실행 요청을 받은 경우에만 시작한다.

## 변경 이력

- 2026-08-13: 항목 12 완료. 현재 경로 문서, AEGIR Git 기준, 외부 repo local
  commits, packed env prefix 검증, legacy checkpoint registry를 확정.
- 2026-08-13: 항목 2 완료. scene-diagonal 기반 bias/steepness 변환, bias 1회 차감,
  논문 transmittance `0.05`를 적용하고 CPU/CUDA/24 GB smoke 검증 기록.
- 2026-08-13: 항목 1 완료. 독립 CUDA test 3개와 bedroom 10-step 통합 smoke 통과,
  peak VRAM 및 finite checkpoint 검증 결과 기록.
- 2026-08-13: 최초 작성. 경로 relocation 완료 상태와 남은 12개 작업을 기록.
