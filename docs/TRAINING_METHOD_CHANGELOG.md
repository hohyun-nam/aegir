# AEGIR training method and change log

This is the persistent source of truth for training semantics in this repository.
Any code change that alters initialization, trainable parameters, losses,
transport, sampling, schedules, or inference must add a dated entry here in the
same commit.

## Reference methods kept for comparison

### Original AEGIR-style baseline

- Initialize the shared Gaussian scene from DN-Splatter/2DGS.
- Initialize area emitters from bright image/G-buffer candidates.
- Initialize per-Gaussian albedo from the DN SH DC appearance and optimize
  albedo, roughness, and metallic together with one `1e-3` material LR.
- Optimize emitter center, rotation, scale, RGB, spread, and falloff jointly.
- Permit Appendix-C emitter energy pruning and scale/energy splitting between
  steps 5K and 15K.
- Steps 0-5K use geometry `1e-5`, material `1e-3`, and emitter `1e-3`.
  Geometry/material decay to zero over 5K-15K while emitter LR decays to
  `1e-4`; steps 15K-20K update emitters only.
- Use the frozen DN full-SH proxy for training-time indirect transport, with
  explicit secondary-bounce transport used for final rendering in the paper
  baseline. See the [historical paper/code audit](https://github.com/hohyun-nam/aegir/blob/61cbf90860df8e7f023e519d090eefe2435dc1c0/docs/PIPELINE_PAPER_AUDIT_2026-09-06.md) for the paper/code
  audit and exact caveats.

To recover the historical adaptive parameter policy with the current CLI, use:

```bash
--albedo-lr 1e-3 \
--optimize-emitter-centers \
--optimize-emitter-rotations \
--unconstrained-emitter-scales \
--enable-emitter-adaptation
```

### Local baseline immediately before this change

Git commit: `b4392c0` (`Add six-axis emitters and corrected AEGIR transport pipeline`).

- Conference runs normally froze DN/2DGS geometry with `geometry_lr=0` and
  disabled Gaussian scene culling.
- Neutral albedo initialization at scene-linear RGB `(0.5, 0.5, 0.5)` was an
  available experiment, but albedo/roughness/metallic still shared `1e-3` LR.
- Emitter center, rotation, scale, common RGB, six signed-axis gains, spread,
  and falloff were all trainable.
- Emitter split/prune was active by default; the latest two-emitter seed split
  once at step 12K and ended with three emitters.
- HDR emitter seeding, persistent-view candidate filtering, finite-segment
  visibility, camera-visible emitter emission, six-axis directional gains,
  origin offset `0.15`, and SH-to-explicit hybrid transport were present.
- Checkpoint format was v9.

## 현재 실전 풀코스 설정 — 2026-09-17

에미터를 2DGS에서 미리 삭제하던 방식을 폐기하고, **한 번 학습한 emitter-inclusive
2DGS 안에서 에미터 Gaussian을 식별해 역할만 분리**한다.

| 단계 | 현재 사용하는 방법 |
| --- | --- |
| Point cloud | RGB 예측 depth, stride 4, voxel 0.10m. 에미터 픽셀도 포함 |
| Native 2DGS | 20K, 예측 depth Pearson 0.2. RGB/depth/초기 seed 모두 에미터 포함 |
| Emitter 초기화 | 전체 2DGS G-buffer의 HDR 상위 0.5%를 3D로 올린 뒤 multi-view persistence → DBSCAN → PCA. GT emitter ID/shape/depth 미사용 |
| Gaussian 연결 | PCA support 내부 source Gaussian에 `emitter_ids=0..E-1`, 일반 장면은 `-1`. tangent 1.05배, 법선 허용 폭은 scene diagonal의 0.1%; 3D emitter는 학습된 세 번째 support scale도 반영 |
| 카메라 G-buffer | tagged Gaussian을 일반 BRDF layer에서 제외하고 analytic emitter의 `L_e`를 합성 |
| Ray tracing | tagged Gaussian을 IRGS BVH에서 제외해 emitter가 자신을 가리거나 secondary surface로 중복 검출되지 않게 함 |
| Emitter OFF | relighting subset에서 켠 emitter ID만 제외. 끈 emitter의 tagged Gaussian은 비발광 BRDF fixture로 다시 raster/BVH에 포함 |
| 저장/호환 | `emitter_ids`를 checkpoint v16에 저장. v1-v15는 전부 `-1`인 배경으로 자동 로드 |
| AEGIR 길이 | 당분간 기본 5K. 5K 이후 품질 정체 원인을 수정하기 전에는 바로 inspect로 진행 |
| Inspect | train 128/1 뒤 128/8, denoiser off 기본 및 denoiser/albedo 비교, tagged source Gaussian을 제외한 PLY export |

`scripts/run_predicted_depth_full_pipeline.sh`가 이 설정의 기준 launcher다. Blender
호환 entrypoint와 Conference entrypoint도 이 스크립트로 연결된다. 새 출력 이름에는
`emitter_inclusive`, `hdrgbuffer`, `5k`를 넣어 과거 emitter-excluded 20K 결과와 섞이지
않게 했다. 일반 CLI의 `--steps`와 `TrainingConfig.steps` 기본값도 5K로 바꿨다.

Association이 켜진 새 학습은 dataset의 GT/segmentation emitter mask를 AEGIR loss
분리에 사용하지 않는다. analytic emitter의 카메라 투영 mask로 photometric/material
영역을 나눈다. emitter mask는 과거 checkpoint/ablation 호환 입력으로만 남는다.

실제 HDR G-buffer 초기화는 풀코스에서 한 번만 수행한다. 학습기가 optimizer와 scene을
만든 직후 `aegir_initialization.pt`를 exact-resume 가능한 step 0 상태로 저장한다.
초기화 PLY를 먼저 만든 뒤 그 checkpoint의 optimizer/RNG까지 복원해 학습을 계속하므로
PLY 확인을 위해 전체 뷰 초기화를 중복 실행하지 않는다. 완료 후
`emitter_initialization/2dgs_emitter_inclusive.ply`에는 원본 inclusive 2DGS가,
`emitter_initialization/map_with_initial_emitters.ply`에는 tagged source row를 뺀 배경과
초기 analytic emitter가 들어간다. `inspect_initialization_vs_5k/`에는 같은 10개 pose의
GT/step0/step5K 비교와 점수가 저장된다.

5K 이후의 배경 geometry 학습 효과를 확인하는 실험은 기본 설정과 분리했다.
`run_conference_inclusive2dgs_mean_only_20k_full_pipeline.sh`는 새로 완료한 model-depth seed와
emitter-inclusive native 2DGS를 재사용하고 AEGIR을 별도 출력에서 20K까지 학습한다.
배경 Gaussian은 means만 LR `1e-6`으로 열며 scale/rotation/opacity, densification,
pruning은 고정한다. 기존 curriculum에 따라 means LR은 5K–15K에 0까지 감소하고 emitter는
기존 설정대로 계속 학습한다. step 0/5K/10K/15K/20K를
같은 10개 pose, BRDF/light 128, secondary 8, denoiser off로 렌더한 뒤 한 폴더로 합친다.
일반 풀코스의 기본 AEGIR 길이는 여전히 5K다.

처음 실행한 scale-only 설정은 사용자 의도와 달라 초기 82 step에서 중지했다. 그 실행의
AEGIR 결과는 비교 대상으로 사용하지 않는다. 교정 실행은 새 mean-only AEGIR 경로를
사용하므로 잘못된 step 0 optimizer 설정을 resume하지 않는다.

Conference mean-only 풀코스가 성공한 경우에만 결과를 자동 분석하고 Blender의 새
predicted-depth/point-cloud/native-2DGS/AEGIR 풀코스를 시작하는 순차 감시 runner를
추가했다. 분석은 checkpoint별 영상 지표뿐 아니라 Gaussian center 이동 분포와
scale/rotation/opacity가 실제로 고정됐는지 수치로 검사한다. Blender도 Conference와
동일하게 ellipsoid emitter, secondary-only train128/1, inspect128/8, means-only LR
`1e-6`, scene culling off, step0/5K/10K/15K/20K 비교를 사용한다.

Native 2DGS PLY exporter는 Nerfstudio `pipeline` checkpoint와 현재
`aegir_2dgs_geometry_v1` checkpoint를 모두 읽는다. 후자의 2축 linear scale에는
시각화용 얇은 세 번째 축을 추가해 log-scale PLY로 기록하고 quaternion에서 normal을
복원한다. 이로써 native 2DGS가 정상 완료된 뒤 PLY export에서 `KeyError: pipeline`으로
중단되던 문제를 수정했다.

Conference의 과거 emitter-inclusive 2DGS 213,856개로 실제 검증했을 때 기존
G-buffer 초기화가 별도 GT emitter 정보 없이 정확히 2개를 복원했고, ellipsoid support에
각각 6,142개와 5,115개를 연결해 202,599개를 배경으로 유지했다. CUDA 2DGS raster와
OptiX BVH 모두 tagged row 제외를 회귀 테스트했다.

## 직전 실전 풀코스 설정 — 2026-09-16 기록

아래는 당시 `scripts/run_predicted_depth_full_pipeline.sh` 및 Blender/Conference
20K checkpoint의 실제 설정이다. 일반 CLI/dataclass의 호환 기본값과 구분한다.
이전 실험은 삭제하지 않고 아래 날짜별 기록에 보존한다.

| 단계 | 현재 사용하는 방법 |
| --- | --- |
| 입력 | 관측 RGB/HDR, 제공된 camera pose/intrinsics, valid/emitter mask |
| Depth | RGB → ZoeD_N, metre/OpenCV axial Z-depth, GT 정합/배율 보정 없음 |
| 초기 point cloud | train views, stride 4, voxel 0.10m, depth 0.1–15m, 통계 outlier 제거, emitter mask dilation 2 제외 |
| Native 2DGS | 20K, 예측 depth Pearson 0.2, invalid-opacity 0.02, protection 7px, distortion 0, densify 10K까지, gradient threshold 0.0005 |
| Emitter 초기화 | HDR 밝은 연결 영역 + calibrated camera ray 다중 뷰 삼각측량; GT 조명/예측 depth 역투영 없이 검증 |
| Emitter 표현 | 풀코스 기본 `ellipsoid_3d`; `planar_ellipse_2d`도 선택 가능. 일반 CLI/dataclass 호환 기본은 2D |
| 배경 Gaussian | AEGIR 시작부터 위치/회전/scale/opacity 고정, scene culling off |
| BRDF | SH-DC albedo 초기화, albedo LR 0.003, roughness/metallic LR 0.001; 5K–15K decay, 15K 이후 고정 |
| Emitter 학습 | center/rotation 고정, scale 초기값의 0.1–1배, RGB/six-axis gains/spread/falloff 학습, split/prune off |
| Transport | secondary-only, 학습 시작부터 SH proxy 0; BRDF/light 각각 128, secondary NEE samples 1 |
| Launch batching | bounce chunk 8192; 샘플 수/추정식과 별개인 실행 배치 크기 |
| AEGIR 로스 | log-domain HDR weighted L1 + display SSIM, 전체 view material prior/TV, area/bounds 각각 0.5; white/surface/view 각각 0 |
| 마지막 5K | BRDF 고정, emitter만 학습; gradient가 없는 material prior/TV 계산 생략 |
| Inspect | 128/8, 기본 denoiser off, denoiser/albedo 비교 및 PLY export; 추가 128/1·128/8·512/8 비교 지원 |
| 입력 검증 | model-depth manifest와 입력/설정 hashes, cache 재사용 및 exact resume provenance 검사 |

**GT depth는 최초 point cloud에도 기본적으로 쓰지 않는다.** 예측 depth는 point cloud
초기화뿐 아니라 native 2DGS 감독에도 사용한다. AEGIR 단계에는 별도 depth/normal
loss를 추가하지 않는다. HDR emitter 위치는 예측 depth와 독립적으로 삼각측량한다.
Conference 평가 분할은 사용자 요청대로 변경하지 않았으며 기존 inspect를 held-out
localization/NVS 평가로 취급하지 않는다.

최근 실제 checkpoint는 Blender
`outputs/blender_practical3d_20260915_051804/aegir_hdr_multiview_v2/aegir_20k.pt`,
Conference `outputs/conference_practical3d_hdrmv_20260915/aegir/aegir_20k.pt`다.
둘 다 step 20K이며 위 loss/secondary-only/freeze 설정을 저장한다.

관련 문서: [실전 풀코스](PRACTICAL_TRAINING_PIPELINE.md),
[HDR 다중 뷰 초기화](HDR_MULTIVIEW_EMITTER_INITIALIZATION.md),
[논문 대조/남은 한계](https://github.com/hohyun-nam/aegir/blob/61cbf90860df8e7f023e519d090eefe2435dc1c0/docs/PIPELINE_PAPER_AUDIT_2026-09-06.md).

## 과거 기본 설정 — 2026-09-14 당시 기록

아래는 당시 GT-depth/planar/hybrid 실험의 기록이다. 2026-09-15 이후 실전
풀코스 기본값으로 읽지 않는다. 현재 설정은 위 표와 마지막 교정 항목을 따른다.

- Each physical emitter is one bounded planar ellipse with center, quaternion,
  two radii, and learned full-sphere directional radiance. The spatial third
  scale remains removed. Common scene-linear RGB, three-axis spread, falloff
  power, and six local signed-axis gains (`+X,-X,+Y,-Y,+Z,-Z`) are trainable.
  Thus the two plane sides learn independently instead of sharing radiance or
  forcing the back side to zero.
- Blender `emitters.json` panel bounding boxes preserve one physical panel to
  one ellipse. Stable multi-view mask components split a legacy compound mesh;
  masks and metric depth remain the fallback when panel metadata is absent.
- GT-depth seed construction and every native-2DGS supervision loss exclude a
  two-pixel-dilated emitter mask. Emitter pixels do not create geometry
  gradients or densification candidates.
- Camera-visible emitter pixels use emission only and skip BRDF/secondary-ray
  shading.
- Checkpoint v14 is exactly resumable. V1-v13 are migrated to planar emitters
  for inference/inspection only.

- Emitter centers and rotations are fixed at their initialized values.
- Emitter scales remain trainable but are clamped independently per axis to
  `0.1x-1.0x` their initialization. They may shrink to fit an over-extended HDR
  cluster but cannot grow into a larger camera-visible proxy.
- Emitter split/prune is disabled. Emitter identity and count therefore remain
  stable unless the historical mode is explicitly enabled.
- Common emitter RGB, signed-axis gains, spread, and falloff remain trainable.
- Albedo initializes from DN SH DC appearance by default, using the historical
  `(0.5 + SH_C0 * DC).clamp(0.02, 0.98)` formula without an additional color
  conversion. Neutral RGB `(0.5, 0.5, 0.5)` remains an explicit option.
- White-light regularization is disabled by default (`light_white=0`,
  `light_white_ramp_end=0`). It remains available as an explicit ablation.
- Albedo uses a dedicated Adam parameter group with initial LR `3e-3`.
  Roughness and metallic retain their shared `1e-3` LR. Both groups follow the
  same 5K-15K linear decay-to-zero curriculum.
- Material prior L1 supervision uses all valid pixels in the rasterized view.
  The expensive BRDF, visibility, and indirect-light transport remains limited
  to the 256-primary-point patch and therefore retains the paper ray budget.
- SH proxy fades independently over 5K-10K. Steps 10K-15K train materials under
  fully explicit secondary transport; steps 15K-20K update emitters only.
- Zero-LR groups disable gradients and do not participate in clipping. Frozen
  geometry skips BVH rebuilds/refinement, and fully frozen scene parameters
  skip the material prior/TV training terms (logged as zero, not measured errors).
- Checkpoint v14 stores the directional emitter state, initialization-relative emitter scale anchors and
  both material optimizer groups, and identifies the full-view material-prior
  objective, independent SH schedule, and active clipping semantics for exact resume.

The Conference wrapper exposes these settings as `ALBEDO_LR`,
`OPTIMIZE_EMITTER_CENTERS`, `OPTIMIZE_EMITTER_ROTATIONS`,
`OPTIMIZE_EMITTER_SCALES`, `CONSTRAIN_EMITTER_SCALES`,
`EMITTER_SCALE_MIN_RATIO`, `EMITTER_SCALE_MAX_RATIO`, and
`EMITTER_ADAPTATION`, plus `LIGHT_WHITE` and `LIGHT_WHITE_RAMP_END`.

## Chronological changes

### 2026-09-15 — Input provenance audit: GT experiments are not the practical pipeline

- The Blender GT-depth runner and the new Conference depthseed runner use
  dataset depth for both point initialization and native-2DGS L1 supervision.
  These are GT-assisted experiments, not RGB-derived depth-model baselines.
- Current emitter initialization prioritizes masks and dataset depth, and may
  use Blender emitter bounding boxes/thickness. It bypasses HDR candidate
  detection when that initializer succeeds. This differs from the user's
  required practical HDR-based initialization.
- Recorded further issues: emitter exclusion versus HDR G-buffer validity,
  raw Zoe millimetre units, missing provenance/cache checks, Conference
  training-view evaluation, and uncompensated initial angular attenuation.
- This entry documents findings only; initialization and runner behavior have
  not yet been corrected. See
  [the historical input audit](https://github.com/hohyun-nam/aegir/blob/61cbf90860df8e7f023e519d090eefe2435dc1c0/docs/PIPELINE_INPUT_PROVENANCE_AUDIT_2026-09-15.md) for evidence and the repair path.

### 2026-09-15 — Conference metric-depth seed and exact emitter exclusion

- Added a conference full-course runner that rebuilds native 2DGS geometry
  before AEGIR instead of reusing the pre-mask 2026-08 checkpoint.
- The FIPT-Real meter-scale camera-Z depth is back-projected at stride 4 and
  voxel-downsampled at 0.10. The current conference seed contains 39,554
  filtered points rather than the multi-million raw back-projections.
- FIPT segmentation ID 93 was verified as the ceiling emission geometry and
  is exported as an emitter mask for all 191 views. It is excluded with a
  two-pixel dilation from the depth seed and native-2DGS RGB/depth losses,
  while class value 64 exempts its valid footprint from invalid-opacity loss.
- Native 2DGS uses metric-depth L1 weight 0.2, invalid-opacity weight 0.02,
  seven-pixel valid-surface protection, zero distortion loss, densification
  through 10K, and gradient threshold 0.0005. Downstream AEGIR freezes this
  rebuilt geometry and uses selectable emitter support; the conference runner
  selects 3D ellipsoids, secondary-only 128/1 training, and 128/8 inspection.

### 2026-09-15 — Selectable pre-planar 3D ellipsoid emitter ablation

- Added `--emitter-representation ellipsoid_3d` while retaining
  `planar_ellipse_2d` as the compatibility default. This changes only the
  analytic light support; the shared scene remains the selected native 2DGS
  checkpoint with emitter-excluded geometry.
- The 3D option keeps the current robust multi-view mask fit for emitter
  center, tangent scales, orientation, HDR color and directional parameters,
  then restores a third normal scale from the Blender emitter bounding-box
  thickness. A compound six-panel mesh supplies the same physical thickness
  to its six disconnected mask components.
- Restored the pre-planar ellipsoid surface sampler, effective-area surrogate
  and analytic ray intersection for this controlled comparison. Center and
  rotation remain fixed, scale constraints and split/prune defaults remain
  unchanged, and RGB/six-axis gains/spread/falloff remain trainable.
- Added checkpoint v15 so exact resume records whether training used planar or
  ellipsoid support. V14 planar checkpoints remain exactly resumable and all
  prior checkpoints remain inspectable.

### 2026-09-14 — Invalid-mask opacity supervision for native 2DGS

- The first binary-mask implementation incorrectly grouped analytic-emitter
  exclusion pixels with true invalid-depth pixels. Its 0.1 opacity penalty
  erased ceiling coverage and thin window blinds; that run is a rejected
  ablation and must not be used as the default geometry.
- Prepared Blender masks now preserve three classes: 255 for normal surface
  supervision, 64 for valid emitter pixels excluded from surface supervision,
  and 0 for true invalid-depth space. RGB, depth, normal, and distortion losses
  still use only the 255 class. Invalid-opacity loss uses only the 0 class, so
  it never asks the 2DGS to open a hole under an analytic emitter.
- A protected band around valid surfaces and exempt emitters allows
  finite Gaussian footprints to represent thin structures such as blinds
  without being punished for slightly overlapping an invalid pane. Blender
  comparison runs showed that weight 0.02 with a 7-pixel band preserves the
  blinds while still removing window fill and avoiding ceiling holes. The full
  course therefore fixes `INVALID_OPACITY_WEIGHT=0.02` and
  `INVALID_OPACITY_PROTECTION_RADIUS=7` as its defaults.
- `LAMBDA_INVALID_OPACITY=0` preserves earlier pipelines. Class-aware training
  uses a fresh `_noemit_classmask` prepared-data path so an old binary mask is
  not silently reused.
- This requires a fresh native-2DGS run and therefore a fresh downstream AEGIR
  run. Existing geometry checkpoints cannot acquire the missing empty-space
  constraint during frozen-geometry AEGIR training.

### 2026-09-14 — Valid-surface masking and Gaussian alpha compositing

- Restricted deferred BRDF and transport evaluation to pixels where the
  rasterized Gaussian G-buffer is valid. Empty geometry pixels can no longer
  inherit normalized material/normal tails and appear as opaque wall-colored
  surfaces.
- Composited shaded Gaussian radiance with its rasterized alpha over the black
  background. Analytic camera-visible emitter emission still replaces a
  Gaussian surface when the ellipse is the nearest hit.
- This is an inference/training renderer correction and does not require
  retraining an existing checkpoint.
- The Blender albedo comparison additionally blacks out the dataset's invalid
  GT-depth mask. Black pixels there do not prove that the rasterized G-buffer
  has zero alpha or a learned black material. The current directional-planar
  checkpoint has substantial Gaussian coverage in parts of the window mask,
  so alpha compositing attenuates but cannot remove all window fill. Removing
  that high-alpha coverage requires explicit invalid-depth opacity supervision
  and a new native-2DGS geometry training run.

### 2026-09-14 — Learned directional emission on planar ellipse support

- Kept one bounded 2D ellipse per physical panel and restored the previous
  anisotropic emission field: common RGB, three local-axis spreads, falloff
  power, and six normalized signed-axis gains.
- Emission is evaluated over the full sphere. `+Z` and `-Z` are separate learned
  gains, so this does not assume equal two-sided emission. Local `+/-X` and
  `+/-Y` gains model off-normal directions, with soft assignment sharpness 12.
- Changed analytic ellipse intersection to accept either side and changed the
  area-to-solid-angle Jacobian to use `abs(n dot omega)`. Directional radiance,
  rather than intersection orientation, determines emitted energy.
- Made all angular parameters part of the emitter Adam group. Center/rotation
  remain fixed and the two radii retain their configured `0.1x-1.0x` bounds.
- Added v14 checkpoints because v13 optimizer state lacks directional tensors.
  V13 remains loadable for inference with its recorded front radiance preserved
  and a neutral directional initialization, but cannot be exactly resumed.

### 2026-09-12 — Configurable secondary-ray launch chunking

- The geometry-configurable training wrapper now accepts
  `BOUNCE_CHUNK_SIZE` and records it in the checkpoint. This execution-only
  setting is also applied when resuming.
- Eight secondary samples with the former 65,536-hit chunk attempted up to
  524,288 secondary rays in one OptiX launch and segfaulted on the Blender
  training scene. The dedicated Blender secondary-only wrapper uses a chunk of
  8,192 hits, keeping each launch at up to 65,536 secondary rays.
- A three-step CUDA smoke test, including exact resume, completed with six
  emitters, fixed geometry, SH-DC albedo, `light_white=0`, explicit transport
  weight 1, SH-proxy weight 0, eight secondary samples, and finite losses.

### 2026-09-12 — Emitter-subset relighting evaluation

- `inspect --active-emitter-indices` can retain selected learned emitters while
  leaving the Gaussian scene and materials fixed. Disabled emitters are removed
  from direct illumination, secondary-bounce NEE, visibility/intersection, and
  camera-visible emission.
- `--evaluation-seed-by-view-order` assigns Monte Carlo seeds from the selected
  camera order. This makes separately indexed datasets comparable when they
  contain the same ordered camera poses. Per-frame metadata records the source
  camera, and the comparison script requires matching camera names, seeds, and
  valid masks.
- The Blender evaluation wrapper compares the original all-six-light GT against
  all six learned emitters and panels-0,5 GT against learned emitters 0 and 5.
  Both use the 40 held-out test poses, 128 BRDF/light samples, one secondary
  sample, seed 2026, and no denoiser or exposure fitting.
- All-six reconstruction measured PSNR 19.2981, SSIM 0.5048, LPIPS 0.5203, and
  log-HDR L1 0.0907. Panels-0,5 relighting measured PSNR 21.0178, SSIM 0.5654,
  LPIPS 0.4604, and log-HDR L1 0.0440. The higher relighting image metrics are
  not a direct improvement over reconstruction because the changed-light GT is
  darker.
- Across valid pixels, Blender panels 0 and 5 retained 43.8% of the all-six GT
  mean scene-linear brightness; the emitter-subset render retained 41.2%. The
  predicted and GT scene-linear lighting deltas had Pearson correlation 0.5695,
  showing the global dimming direction transferred while local light transport
  remained imperfect.

Validation: 122 tests passed, including the CUDA/OptiX tracer tests. Ruff,
Python compilation, shell syntax, and the 40-pair camera/seed/mask protocol
checks passed.

### 2026-09-12 — Optional inspect denoiser and matched comparison artifacts

- Added an opt-in cross-bilateral filter for the indirect linear-HDR component
  during `inspect`. It uses rendered depth and normals for edge stopping, then
  recomposes direct, filtered indirect, and emission before tone mapping.
- The default inspection remains unfiltered. Denoising does not alter training
  or checkpoints, and inspect metadata records the full filter configuration so
  filtered output cannot be mistaken for the baseline renderer.
- Raw metric frames now include the components needed to create reproducible
  no-denoiser/denoiser and white-loss comparisons. The Blender full-course
  scripts generate per-frame panels, contact sheets, GIFs, and metric summaries.

### 2026-09-12 — SH-DC albedo and no white-light prior as the Blender default

Three controlled 20K runs used the same GT-depth voxel-0.10 2DGS geometry,
frozen geometry, hybrid transport, one secondary sample, 128 BRDF/light
samples, and the same 30-view seed-2026 no-denoiser inspection. Only albedo
initialization and the fixed white-light loss differed:

| Albedo initialization | `light_white` | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: | ---: |
| SH DC | 0 | **19.315** | **0.527** | **0.486** |
| SH DC | 0.1 | 17.318 | 0.515 | 0.542 |
| Neutral 0.5 | 0 | 19.049 | 0.492 | 0.520 |

- With SH DC and `light_white=0.1`, the final mean emitter RGB became
  `(15.0965, 15.0963, 15.0965)`. The prior removed the warm illumination in
  the Blender observations and produced a visibly gray rendering.
- With neutral-0.5 initialization and no white prior, the learned albedo stayed
  visibly pale while mean emitter RGB moved to `(36.9712, 21.6126, 9.6826)`.
  Material and illumination color compensated for one another, producing an
  overly yellow light.
- SH DC with no white prior gave the best reconstruction metrics and visual
  balance. The default `LossWeights.light_white`, training CLI
  `--light-white`, and frozen-geometry wrapper `LIGHT_WHITE` are now zero.
  `albedo_initialization=sh_dc`, `light_white_ramp_end=0`, hybrid transport,
  and the existing schedules remain the defaults.
- This result selects the Blender reconstruction baseline. It does not by
  itself prove physically correct material-light separation; changed-light
  Blender ground truth is still required to validate relighting.

Validation: 116 tests passed, including the CUDA/OptiX tracer tests with the
validated NVIDIA 580.173.02 OptiX runtime. Ruff passes on every changed Python
file; the repository-wide Ruff scan still reports pre-existing findings in
unmodified files.

### 2026-09-11 — Optional white-light prior ramp

- Added `training.light_white_ramp_end`, with a default of zero that preserved
  the then-current fixed `lambda_white=0.1` behavior. The target weight later
  changed to zero by default after the 2026-09-12 Blender ablation.
- A positive ramp end applies `smoothstep(clamp(step / ramp_end, 0, 1))` to
  the final white-light weight. The active weight is recorded in every
  loss-history entry so checkpoint comparisons can audit the exact schedule.
- Added `--light-white` and `--light-white-ramp-end` to the training CLI and
  matching environment controls to the frozen-geometry experiment wrapper.
- This supports the Blender ablation that starts the prior at zero, reaches the
  paper value `0.1` at step 15K, and holds it there through step 20K.

### 2026-09-08 — Persisted training transport ablations

Previous secondary-only and SH-only experiments replaced
`training_sh_proxy_weight` outside the saved configuration. Their checkpoints
therefore appeared hybrid even though the actual runs were not.

- Added persisted `training_transport_mode` values `hybrid`,
  `secondary_only`, and `sh_proxy_only`. The default remains `hybrid`.
- `secondary_only` uses explicit emitter NEE from the first training step;
  `sh_proxy_only` uses the frozen DN SH proxy for every step. Hybrid retains
  the independent 5K-10K cross-fade.
- Added a Conference secondary-only pipeline that trains with one bounce
  sample, creates a 30-view GT / DiffusionRenderer albedo / learned albedo
  comparison, rerenders the same 30 poses with one and eight bounce samples,
  and composes the three columns into a comparison folder.
- The secondary inspect helper can skip its otherwise duplicate PLY export via
  `EXPORT_PLY=0`. Once the full pipeline completes, non-20K intermediate
  checkpoints in that new run are removed in accordance with the current
  output-retention policy.

Increasing the inspection bounce sample count reduces Monte Carlo variance;
it does not retrain the checkpoint or remove estimator bias.

### 2026-09-08 — True parameter freezing and independent SH transition

Previous baseline: `6c41fd9`, v11. Geometry/material freezing only set Adam LR
to zero. Their gradients still entered global clipping, so unused gradients
could shrink emitter/material gradients. Prior/TV and BVH rebuilds continued
after freeze. SH fade and material LR decay both ended at 15K.

Implementation:

- `_update_lrs` switches optimizer-owned leaf parameters' `requires_grad` flags
  with their LR and clears stale gradients for inactive groups. Clipping only
  includes parameters in positive-LR groups. Fixed emitter pose remains excluded
  from its optimizer.
- Prior/TV are skipped when no scene parameter can learn; gradients through
  transport to the emitter are preserved. These skipped loss entries are zero
  and must not be interpreted as perfect prior agreement.
- The rasterizer screen-space gradient placeholder is disabled for fixed geometry.
  Scene refinement is also disabled while geometry is frozen.
- BVH refresh requires a preceding geometry update. The first frozen step still
  refreshes after the final active update; a restored trainer builds its own BVH.
- Added persisted `sh_fade_start=5000`, `sh_fade_end=10000`, independent of the
  unchanged material LR decay from 5K to 15K. Exposed CLI flags and Conference
  wrapper environment variables `SH_FADE_START` / `SH_FADE_END`.
- v12 marks the changed optimization semantics. v1-v11 remain inference-loadable,
  but cannot exact-resume as v12. Legacy config loading recovers the old coupled
  fade when the new fields are absent. Old fade timing can be selected with
  `SH_FADE_END=15000`; exact reproduction of old clipping needs the old commit.

Regression checks cover frozen-gradient clipping isolation, actual parameter
freeze and retained emitter derivatives, skipped regularizers, and schedule
serialization/legacy interpretation. No changes to PDF, expected hit, SH radiance
units, origin offset, GGX, or emitter initialization are included in this change.

Validation: 110 tests passed. A four-step CUDA smoke run using the actual
Conference scene crossed the SH transition and material-freeze boundary:
geometry tensors stayed bitwise unchanged, frozen scene gradients were absent,
emitter RGB gradients remained finite/nonzero, BVH refresh count was zero, and
the resulting checkpoint was v12. Peak allocated GPU memory was 0.41 GiB.
This is a functional smoke test, not a new 20K quality/convergence evaluation.

### 2026-09-07 — Bundle patched extension sources for another computer

- Added the currently used IRGS tracer source, including finite-segment
  forward/backward tracing, and the 2DGS rasterizer with its GLM dependency
  under `AEGIR/external`. Preserved upstream license and attribution files.
- Extension installation scripts now default to these bundled sources rather
  than the untracked sibling workspace. A normal Git clone/pull includes the
  modifications; separate patch application is no longer needed.
- Build outputs are ignored and must be rebuilt for the destination environment.
  Training initialization, schedules, losses, and transport math are unchanged.
- Checked source snapshot parity and shell syntax. A destination-machine CUDA
  build must still be performed; this packaging change does not certify that
  machine's compiler, driver, or PyTorch compatibility.

### 2026-09-07 — SH DC initialization with full-view material priors

- Restored `sh_dc` as the default in the training config, CLI, and Conference
  wrapper, to compare appearance-based initialization against the neutral v11
  run now that material priors supervise the full view.
- Preserved albedo LR `3e-3`, the hybrid SH-to-secondary schedule, material
  freeze at 15K, fixed geometry and emitter pose, and HDR emitter initialization.
- The Conference wrapper now also generates 30 GT / DiffusionRenderer albedo /
  learned albedo panels after its 30-view secondary inspection and PLY export.
- Checkpoint format remains v11: initialization is already stored in config,
  and exact resume continues to use the saved state rather than reseeding it.
- This experiment reuses the historical appearance values; it does not establish
  that SH DC is physically calibrated linear reflectance or fix sampling noise.
- Reproduce neutral initialization with `ALBEDO_INITIALIZATION=neutral` and
  `NEUTRAL_ALBEDO=0.5` in a separate output directory.

### 2026-09-07 — Full-view material-prior supervision

Problem observed:

- The renderer rasterized a complete G-buffer but cropped the DiffusionRenderer
  albedo, roughness, and metallic priors to the same 16x16 patch used for
  expensive lighting transport.
- With 191 Conference views and material learning frozen after step 15K, many
  Gaussians received sparse prior gradients while full-view TV continued to
  smooth their materials. Neutral-initialized albedo consequently retained a
  compressed gray dynamic range, especially on dark objects.
- The AEGIR paper specifies `2^16` transport rays per iteration and an L1 loss
  between rendered G-buffer maps and diffusion priors, but does not require
  cropping the material-prior objective to the transport patch.

Implementation:

- Albedo, roughness, and metallic L1 prior losses now use every valid pixel in
  the already-rasterized full view.
- BRDF sampling, explicit-light sampling, visibility tracing, secondary-bounce
  transport, and photometric loss remain on the 256-primary-point patch. The
  expensive transport-ray budget is unchanged.
- Checkpoint format v11 distinguishes the new training objective. V1-v10 remain
  loadable for inference or explicit warm starts but cannot exact-resume under
  the full-view material objective.

### 2026-09-07 — Blender render-frustum intrinsics correction

Problem observed:

- The legacy Blender capture code computed `fx` from `Camera.angle_x` and `fy`
  from `Camera.angle_y`. Those angles describe the physical sensor, while the
  rendered frame also applies Blender's sensor-fit and output aspect ratio.
- A 36x24 mm AUTO-fit camera rendered at 512x512 was therefore recorded as
  `fx=398.22, fy=597.33`, although Blender's actual square render frustum was
  `fx=fy=398.22`.
- The incorrect calibration affected both fixed-pose COLMAP triangulation and
  the ray-distance-to-z conversion used by Blender GT depth.

Implementation:

- `render_dataset.py` now derives `fx`, `fy`, `cx`, and `cy` from
  `Camera.view_frame(scene=...)`, Blender's authoritative fitted render
  frustum.
- Added `correct_capture_intrinsics.py` to create a separate corrected legacy
  dataset without modifying or duplicating RGB/HDR/G-buffers. It reconstructs
  each pixel's ray distance with the legacy calibration and converts it back to
  z-depth using the corrected calibration.
- Corrected datasets deliberately omit old COLMAP, mono-depth, and SFM-depth
  products. Those products must be regenerated from the corrected transforms.
- The Conference AEGIR training method itself is unchanged by this data-camera
  correction.

### 2026-09-07 — Fixed emitter pose and faster neutral-albedo learning

Problem observed:

- Optimized camera-visible emitter ellipsoids expanded/moved away from the
  physical fixtures, and adaptive splitting changed two initialized emitters
  into three learned proxies.
- Neutral `0.5` albedo reduced baked-in shadows but converged too slowly under
  the shared material LR, leaving broad gray/color-compressed results.

Implementation:

- Added separate albedo and non-albedo material Adam groups (`3e-3` and
  `1e-3`).
- Fixed emitter center/rotation by excluding them from optimization and setting
  `requires_grad=False`.
- Added initialization-relative scale anchors and post-step scale clamping.
- Disabled emitter split/prune by default while retaining an explicit baseline
  switch.
- Bumped exact-resume checkpoints from v9 to v10. V1-v9 remain loadable for
  inference or emitter warm start, but not exact optimizer resume.

## Recovering older code with Git

Every pushed commit remains available unless repository history is forcibly
rewritten. Useful commands:

```bash
git log --oneline --decorate

# Inspect the entire repository at the pre-change baseline without changing main.
git switch --detach b4392c0

# Return to current development.
git switch main

# Restore one old file into the current worktree for comparison.
git restore --source b4392c0 -- path/to/file

# Make a new commit that reverses a selected change.
git revert <commit-hash>
```

Do not switch commits in the same worktree while a training process is importing
code from it. Use a second clone or `git worktree add` for side-by-side runs.

## 2026-09-15 — GT 감독 제거 및 실전 입력 파이프라인 교정

사용자 요청: **Conference 평가 분할은 변경하지 않고**, 입력 출처 감사에서 확인한 나머지 문제를 수정한다. 기존 checkpoint의 결과를 새 실전 학습 결과로 재분류하지 않는다.

- Blender/Conference 공통 풀코스는 `scripts/run_predicted_depth_full_pipeline.sh`를 사용한다. RGB → ZoeD_N 추론 → metre/OpenCV axial Z-depth → stride 4, voxel 0.10m seed → native 2DGS 20K → AEGIR 20K → inspect 순서다. GT depth 정합/배율 보정은 하지 않는다. RGB 모델의 metre 출력과 제공된 카메라 좌표 단위가 같은 기준이라는 입력 가정은 필요하며, 추정 depth의 실제 metric 정확도를 보장하지 않는다.
- 2DGS 외부 depth 감독은 **예측 depth + Pearson loss, lambda_depth=0.2**로 변경한다. 기본 경로에서 GT depth L1이나 GT normal/material 감독을 사용하지 않는다. `normal_world_file_path`, `metallic_gt_file_path` 등 잔존 GT metadata도 제거한다.
- 기본 emitter 초기화는 `TrainingConfig.emitter_initialization="hdr_predicted_depth"`다. **관측 HDR 고휘도 후보 → 독립 보존한 예측 depth 역투영 → 다중 뷰 persistence → DBSCAN → PCA**를 사용한다. 배경 G-buffer 유효성이나 emitter Gaussian 존재를 요구하지 않는다. emitter mask는 배경 seed/supervision 제외에 쓰며, 초기 조명 개수·위치를 GT segmentation component/bbox로 강제하지 않는다.
- `emitters.json`의 Blender GT bbox/두께/색/strength를 기본 학습에서 읽지 않는다. 3D support는 GT 두께 없이 예측 후보 및 scene scale에서 정한다. 2D/3D는 `EMITTER_REPRESENTATION=planar_ellipse_2d|ellipsoid_3d`로 계속 선택한다. 새 조명의 사용자 입력/평가용 GT 조명 변환 도구는 별도로 유지한다.
- HDR channel median을 초기 방사 radiance로 맞추기 위해 기본 spread=power=gain=1 커널의 exp(-1) 감쇠를 보상한다. 초기 amplitude `C=e*median(HDR)`이므로 초기 `Le=median(HDR)`다. 이것을 학습 완료 조명이나 Blender radiance 전환에 무조건 e배 곱하는 규칙으로 해석하지 않는다.
- 예측 depth manifest에 모델, weights/generator hash, metre/Z/OpenCV, GT alignment=false, RGB/depth SHA256 및 metadata signature를 기록한다. 기존 raw Zoe의 mm NPY 또는 GT 디렉터리를 `mono_depth`로 이름만 바꾸어 재사용하지 않는다.
- seed, 준비된 class mask/depth/PLY, native checkpoint, AEGIR checkpoint에 입력/설정 출처를 확인한다. 출처 없는 과거 산출물이나 다른 설정의 캐시는 자동 재사용하지 않는다. 중간 AEGIR checkpoint는 dataset/geometry/prior 내용도 기록하며 exact resume 시 비교한다. 결과 경로는 `preddepth...v1`로 분리한다.
- GT 진단은 명시적으로만 허용한다: seed `--allow-gt-depth`, 준비 view `--allow-unverified-depth`, native runner `ALLOW_UNVERIFIED_DEPTH=1`, AEGIR `EMITTER_INITIALIZATION=gt_mask_depth`. `hdr_gbuffer`는 기존 초기화 비교용 선택이며 실전 풀코스 기본값이 아니다. 과거 파일명 `run_blender_gtdepth_full_pipeline.sh`는 호환 entrypoint이며 이제 **모델 depth 풀코스**로 연결한다.
- 유지: SH-DC albedo, white loss=0, 배경 geometry freeze/culling off, secondary-only train128/1·inspect128/8, denoiser 기본 off 및 비교, albedo 비교, invalid loss0.02/protection7, emitter exclusion dilation2, 방향별 방사 학습 및 사용자 지정 pose/scale 제약. Conference split은 추가하지 않는다.
- GPU 검증 환경은 NVIDIA 전체 드라이버 디렉터리 대신 `outputs/optix_runtime_only`의 nvoptix/rtcore만 노출한다. OpenCV/Mesa 라이브러리 충돌을 피한다.

실행 entrypoint: Blender `bash scripts/run_blender_gtdepth_full_pipeline.sh`, Conference `bash scripts/run_conference_depthseed_full_pipeline.sh`. 학습 결과 변경은 **새 풀코스로 재학습한 뒤** 평가해야 한다. 자세한 설정은 [PRACTICAL_TRAINING_PIPELINE.md](PRACTICAL_TRAINING_PIPELINE.md) 참고.

검증: 전체137 tests/Ruff/shell syntax 통과. 실제 Conference RGB2장 ZoeD_N 추론, frame33 seed4,143점/native2DGS1-step/AEGIR1-step,128/8 기본·denoised inspect 및 비교/알베도 비교 실행 확인. 전체 재학습·수렴 검증은 아직 수행하지 않았다. Native2DGS loader의 절대 RGB path 처리 문제를 피하기 위해 prepared view의 RGB 링크와 filename split을 함께 상대 경로로 정규화한다.

## 2026-09-15 — 예측 depth HDR 초기화의 persistence 탈락 교정

Blender practical run `blender_practical3d_20260915_051804`는 native2DGS20K/export까지 완료했지만 AEGIR 초기화에서 중단됐다. 실제120 train views의 유효/in-bounds HDR 후보는37,817개였다. scene diagonal16.0937m에서 기존 voxel fraction0.005는8.0469cm 셀이며, 셀마다 최대5개 뷰만 겹쳐 required_views=6을 충족한 후보가0개였다. 후보 부재가 아니라 voxel 경계 및 mono depth의 뷰별 위치 오차로 인한 잘못된 탈락이었다.

- 기본 `hdr_predicted_depth` 초기화는 `emitter_persistence_radius_fraction=0.01`을 사용한다. 이번 scene의 허용 반경은16.0937cm다. 뷰마다 nearest-neighbor tree를 조회하여 해당 반경 내 고휘도 후보가 있으면1표를 더한다. 한 뷰의 많은 픽셀을 다중 뷰 증거로 중복 집계하지 않는다.
- required_views는 기존 `max(3,ceil(train_views*0.05))`를 유지한다. 이 run은6뷰이며 1뷰 또는 임의 fallback으로 완화하지 않는다. 통과 후보만 DBSCAN/PCA에 사용하고, fitted cluster에도 required_views 이상의 retained view가 있어야 한다.
- 기존 generic HDR/G-buffer initializer는 radius fraction0일 때 이전 voxel 검사를 유지한다. 새 기본 HDR/예측-depth 경로만 radius 검사를 사용한다. 조정은 AEGIR config 또는 runner의 `EMITTER_PERSISTENCE_RADIUS_FRACTION`으로 가능하다.
- RGB/HDR·pose·certified predicted depth만 사용한다. GT emitter bbox/depth/강도와 emitter segmentation component 수를 초기화 근거로 사용하지 않는다. 실제120 train views preflight에서 **6개 emitter**가 검출됐다. 이 숫자를 GT 개수로 강제하지 않았다.
- 완료된2DGS/checkpoint 출처 검증은 보존한다. 코드가 바뀌므로 전체 풀코스를 재실행해서 native cache를 재인증하지 않고, AEGIR만 새 output `.../aegir_hdr_radius01/`에서 시작한다. 기존 실패 pending/output은 삭제하거나 새 provenance로 덮어쓰지 않는다.

초기화 위치는 여전히 모델 depth의 추정치다. 다중 뷰 초기화 통과를 GT 조명 위치·방향 정확도나 학습 수렴의 보장으로 해석하지 않는다.

검증: 전체139 tests/Ruff/shell syntax 통과. 실제120-view 에미터 초기화6개 성공. 기존 tmux session `blender_practical3d_20260915_051804`의 새 window `aegir_radius01`에서 `resume_aegir_radius01.sh`를 실행했고, AEGIR20K의 optimizer step이 진행되는 것을 확인했다. 최종 checkpoint 예정 경로는 `outputs/blender_practical3d_20260915_051804/aegir_hdr_radius01/aegir_20k.pt`이며 같은 runner가 완료 후128/8 기본·denoised inspect 및 albedo/denoiser comparison을 수행한다.


## 2026-09-15: HDR 다중 뷰 삼각측량으로 에미터 초기화 교정

- 원인: 단안 모델 depth의 조명 표면 오차가 에미터 중심에 전파됐다. HDR 상위0.5%만 자르면 균일한 패널이 MC 밝기 차이 때문에 작은 영역으로 나뉘었다. 이전1% 반경 persistence 수정은 후보 부족을 우회했지만 위치 정확도를 검증하지 못했다. `aegir_hdr_radius01`은 잘못된 초기화 실험으로 유지한다.
- 변경: 기본 `emitter_initialization=hdr_multiview`. 관측 HDR의 밝은 연결 영역 전체를 검출하고 카메라 광선을 삼각측량한다. 뷰당 하나의 대응으로 최소max(3, train5%) 뷰, 최대3px 재투영 오차, 최소3도 시차를 검증하고 robust 위치 최적화한다.
- 형상: 사각형 모서리 대응/평면 역투영 또는 타원 silhouette moment fitting으로 normal·면적·장단축을 복원한다. 사각형은 같은 면적의 타원으로 근사한다. ellipsoid_3d/planar_ellipse_2d 선택과 방향별 방사 모델을 유지한다. 관측할 수 없는3D 두께는 scene-diagonal0.002 초기 prior다.
- GT 조명 위치/shape/strength와 emitter mask ID는 대응관계에 사용하지 않는다. 모델 depth로 에미터 위치를 역투영하지 않는다. valid mask와 배경 맵 bounds만 사용하며 GT/depth/임의 emitter fallback을 금지한다. `hdr_predicted_depth`는 명시적 과거 모드로 남긴다.
- CLI/TrainingConfig/공통 풀코스/AEGIR runner 기본값을 전환하고 신규 AEGIR 기본 경로를 `preddepth_hdrmv_...v2`로 구분했다. SH-DC/white0/고정 배경, 사용자가 선택한 train/inspect 샘플 설정은 유지한다.
- `scripts/inspect_hdr_emitter_initialization.py` 추가: train 관측만으로 초기화하여 JSON, 투영 overlay/contact sheet, step0 시각화 checkpoint, 초록 에미터를 넣은 SuperSplat PLY를 생성한다. 학습에서도 초기화 진단 JSON과 checkpoint 내부 진단을 저장하고 resume 시 보존한다.
- 실제 Blender120뷰:6개가 천장 높이2.296–2.303m의2×3 배열로 복원. 조명당28–47뷰, 재투영 중앙값0.29–0.48px. 산출물 `outputs/blender_practical3d_20260915_051804/emitter_init_hdr_multiview_v2/`.
- 검증: 사각형/타원 복원, radiance·면적·normal, 전체 HDR 영역 검출,2뷰/zero-baseline fallback 금지 및 fresh trainer 연결 회귀 테스트 포함 전체145개 통과. 수렴/최종 화질 평가는 별도 재학습이 필요하다.
- 자세한 변경과 명령: [HDR_MULTIVIEW_EMITTER_INITIALIZATION.md](HDR_MULTIVIEW_EMITTER_INITIALIZATION.md). 검증된 실전2DGS는 재사용하고 AEGIR을 새 출력에서 처음부터 학습한다. 잘못 초기화된 checkpoint의 exact resume는 초기화를 다시 실행하지 않는다.
- 실제 실행 추가 검증: Blender 배경132,417 Gaussians/에미터6개, GPU128/1 학습2-step + exact resume1-step 성공. geometry_lr0/white0/SH proxy0, 유한 loss 및 초기화 진단 보존을 확인. `emitter_init_hdr_multiview_v2/verification.json` 저장. 전체20K 재학습은 시작하지 않았다.


## 2026-09-15: Conference HDR 초기화 중복 피팅 병목 수정

- Conference에서 HDR 초기화가10분 이상 진행 정체됐다. 프로세스는 CPU에서 실행 중이었으며 중단 시 stack도 silhouette least-squares의 수치 Jacobian 계산에 있었다. 같은 HDR 관측 대응에 수렴하는 삼각측량 후보를 반복 피팅하고, 최종 중심 재투영 검증 실패 시 하나의 seed만 지우며 전체 후보 점수를 다시 계산하는 구조가 원인이었다.
- 수정: 관측 available mask가 바뀌는 에미터 수락 시에만 재투영 후보 점수를 다시 계산한다. 각 pass에서 같은 refined observation ID 집합의 형상 피팅은 한 번만 수행한다. 최종 중심 재투영/관측 수/크기 검증과 GT/depth fallback 금지는 유지했다. 단순히 허용오차를 늘리거나 실패 후보를 조명으로 채택하지 않는다.
- 검출/score pass/에미터 수락/10초 간격 검사 현황/완료 로그를 추가했다. 진단 JSON에 elapsed_seconds/score_passes/unique_shape_fits/duplicate_hypotheses_skipped를 저장한다.
- 실제 Conference191-view 사전 검사: 약49.5초에13개 에미터 수락. 형상 피팅23회, 중복 seed618개 건너뜀. `outputs/conference_practical3d_hdrmv_20260915/hdr_optimized_preflight.json` 저장.
- 실제 Blender의 기존6개 에미터와 center/normal/scales/radiance/views 동등성을 다시 확인했다. 합성 중복100개 seed의 최종 중심이 검증 실패할 때 score1회/shape fit1회로 끝나고 fallback하지 않는 회귀 테스트를 추가했다. 전체146개 테스트 통과.
- 기존 Conference의 완료된 실전2DGS20K는 유지했다. 학습 checkpoint가 없음을 확인하고 이전 초기화 프로세스만 SIGINT로 종료했다. old pending request/timing은 `aegir/interrupted_initialization_20260915/`에 보관했다. 같은 tmux `conference_hdrmv`와 output `aegir/`에서 AEGIR 단계만 다시 실행한다.
- 재실행 launcher: `outputs/conference_practical3d_hdrmv_20260915/resume_aegir_optimized_hdrmv.sh`. 기존 train128/1, inspect128/8, frozen geometry, ellipsoid_3d, SH-DC, white0, secondary-only 설정과 비교/PLY 후속 작업은 유지한다.
- 실제 재시작 확인: 초기화49.34초, 에미터13개로 완료. 이후 AEGIR iteration 진입과 유한 loss를 확인했다. `aegir/initialization_restart_verification.json` 저장.


## 2026-09-16 — 분산된 수정 이력 통합 및 최신 설정 정리

사용자 지적에 따라 큰 항목 요약만으로 누락된 초기 수정, 운영 수정, 평가 도구,
되돌린 실험을 Git 이력 및 원래 감사 문서와 대조했다. 이 절은 과거 수정의 소급
기록이며, 아래 변경을 모두 9월 16일에 새로 구현했다는 뜻이 아니다. 문서 앞의
2026-09-14 설명은 과거 기록으로 표시하고 최신 실전 풀코스 설정을 별도로 올렸다.
학습 수식이나 기존 checkpoint를 이번 문서 정리로 변경하지 않는다.

### A. 8월 초기 구현에서 수정했지만 이 문서에 모이지 않았던 항목

출처: 초기 baseline `10127b2`, 통합 transport commit `b4392c0`,
[초기 코드 검토](https://github.com/hohyun-nam/aegir/blob/61cbf90860df8e7f023e519d090eefe2435dc1c0/AEGIR_PAPER_CODE_REVIEW_2026-08-12.md),
[IRGS glue 감사](https://github.com/hohyun-nam/aegir/blob/61cbf90860df8e7f023e519d090eefe2435dc1c0/docs/IRGS_GLUE_AUDIT_2026-08-14.md).
원문에는 수정 전 설명과 수정 후 상태가 함께 있으므로 완료 표시와 현재 코드를 구분한다.

1. **회전된 anisotropic emitter의 좌표축 오류**: row-vector의 world-to-local
   변환을 `v @ R`로 맞춰 intersection과 radiance의 local frame을 통일했다.
   회전한 3D emitter의 축별 교차 거리와 quaternion/scale gradient를 검증했다.
2. **학습/평가 split 무시**: loader가 train/val/test/all을 선택하고 split을
   max_views보다 먼저 적용한다. 원본 frame ID를 유지해 material prior가 어긋나지
   않도록 했다. Conference는 이후 사용자 요청대로 기존 분할을 유지한다.
3. **dataset mask 및 Gaussian valid mask 누락**: photo/prior/TV에 유효성 검사를
   연결했고 TV는 양쪽이 유효한 pixel pair만 사용한다. 뒤의 emitter-exclusion 및
   emission 감독 변경으로 photo mask와 재질 mask는 서로 다르게 처리한다.
4. **학습 가능한 global ambient 제거**: explicit emitter 대신 ambient가 장면을
   밝히는 우회 경로를 제거했다. 오래된 ambient checkpoint 필드는 무시한다.
5. **visibility의 scene 단위 의존성**: bias/steepness를 scene scale에 맞추고
   transmittance 종료값을 AEGIR 설정 0.05로 맞췄다. origin offset과 endpoint bias는
   서로 다른 값이며, 최근 wrapper의 origin offset 0.15는 근본 self-hit 해결이 아니다.
6. **임의 secondary proxy 제거 및 transport 경로 구현**: 기존 center/sphere
   proxy와 임의 bounce_strength를 제거하고 native IRGS Gaussian tracing을 연결했다.
   frozen DN full SH를 보존하는 학습 proxy와 explicit emitter 기반 inference를 분리했다.
   이후 secondary-only/NEE 및 독립 fade 실험으로 확장했다.
7. **DiffusionRenderer 입력/출력 glue**: RGB/mask 입력 준비, 공식 inverse inference,
   basecolor/roughness/metallic prior import 경로를 복원했다. basecolor의 sRGB→linear
   decode는 importer에서 한 번만 수행한다. 이는 GT material 감독이 아니다.
8. **ray budget 해석 오류**: `rays_per_step=256`은 primary point 수다.
   `256 × (128 BRDF + 128 light)=65536` primary transport rays로 통일했다.
   secondary 추가 tracing까지 모두 포함한 총 ray 수라는 뜻은 아니다.
9. **HDR/LDR 색공간 정책**: HDR 관측은 symmetric log1p L1, LDR-only는 저장된
   exposure/tone mapper/transfer function을 사용한다. 미지정 LDR에는 명시적인
   learned exposure + Reinhard/sRGB fallback이 있다. 현재 HDR 로스는 논문의 LDR
   photometric과 다른 로컬 선택이며 raw HDR와 표시 PNG를 분리한다.
10. **focus map의 patch 경계 의존성**: 전체 view에서 masked 7×7 contrast와
    global normalization을 계산한 뒤 patch로 crop하여 같은 pixel의 weight를 유지한다.
11. **논문 외 강한 emitter priors**: surface proximity 10, camera-frustum 1의
    기본 가중치를 각각 0으로 내려 opt-in으로 남겼다. 최근 학습에서도 둘 다 0이다.
12. **split/prune 시 Adam state 손실**: survivor의 optimizer moment를 유지하고
    split child moment를 초기화한다. 현재 adaptation은 기본 off지만 역사적 모드는 유지한다.
13. **정확한 resume**: optimizer/global step/Python·NumPy·CPU·CUDA RNG/bounds/
    surface anchors/frame identity/adaptation state를 보존한다. 다른 schema나 입력의
    inference/warm start와 exact resume를 구분한다. 후속 provenance 검사를 추가했다.
14. **평가 reproducibility**: mask·split·seed·sample 수 및 raw float tensor를
    기록하고 PSNR/SSIM/spatial LPIPS 경로를 정리했다. GT 없는 novel path나 학습 view
    재구성을 held-out 정확도로 보고하지 않는다.
15. **IRGS native binding 계약 검사**: CUDA float32/device/shape/quaternion/두 tangent
    scale/SH layout/material channel 한계/trace distance를 native 호출 전에 검사한다.
    raw SH composite와 alpha로 정규화한 hit depth/material/normal을 구분하고,
    ray-origin shift를 depth에 복구해 bias를 중복 적용하지 않도록 했다.

### B. 9월 6일 transport 수정 묶음 — 별도 감사 문서에만 상세 기록되던 부분

출처: `b4392c0`, [논문 대조](https://github.com/hohyun-nam/aegir/blob/61cbf90860df8e7f023e519d090eefe2435dc1c0/docs/PIPELINE_PAPER_AUDIT_2026-09-06.md).

- **에미터 뒤 Gaussian이 shadow에 섞임**: surface–light endpoint를 OptiX
  forward/backward에 전달하여 유한 ray segment 밖 hit를 alpha/depth/material/SH에서 제외했다.
- **persistent 필터 이전의 비지속 HDR 후보 혼입**: 여러 view에서 지지되는 후보만
  DBSCAN/PCA에 사용하도록 바꿨다. 이는 현재 HDR multiview 초기화 이전 단계의 수정이다.
- **HDR 후보에 LDR 색을 넣던 초기화**: HDR channel median으로 radiance를 초기화하고
  기본 angular kernel exp(-1)을 amplitude `e × median(HDR)`로 보상한다.
- **중성 albedo 실험 경로**: neutral 0.5 옵션을 추가했다. 뒤의 Blender 비교에서
  SH-DC/white0를 최종 기본으로 선택했으며 neutral은 실험 옵션으로 유지한다.
- **카메라에 직접 보이는 발광 누락**: camera ray의 emitter hit와 Gaussian surface
  거리로 depth winner를 정하고 emission을 별도 합성한다. emitter에 가려진 BRDF
  shading을 중복 합산하지 않으며 emission_hdr/emitter_mask를 진단 출력으로 보존한다.
- **방향 방사 대칭 한계**: local ±X/±Y/±Z의 mean-one gains를 추가하고 emitter→receiver
  outgoing 방향으로 부호를 통일했다. direct/secondary NEE와 camera emission이 같은
  규약을 사용한다. symmetric-emission으로 원래 대칭 모델도 비교할 수 있다.

### C. 후속 운영·데이터·평가 수정 중 날짜별 기록에서 빠졌던 부분

- **9월 9일 mesh alpha 필터와 revert** (`6eb8abc` → `56dd5f9`):
  TSDF 적분의 낮은 accumulated-alpha depth 제외를 시도한 뒤 되돌렸다. revert 과정에서
  문서 항목까지 삭제됐으므로 이 절에 기록을 복원한다. 현재 활성 수정으로 세지 않는다.
- **9월 12일 Blender panel 분리** (`93b23d6`, `e74344c`): compound emission mesh를
  여섯 개로 분리하고 panels 0,5 조건 및 같은 test camera pose의 GT 수집 경로를 만들었다.
- **LDR-only 새 조명 적응 방향** (`f4125cc`): 배경 geometry/BRDF를 고정하고 새 emitter와
  제한된 camera response를 학습하는 계획을 기록했다. 계획 문서이며 새 조명 적응
  파이프라인이 구현 완료됐다는 뜻은 아니다.
- **9월 13일 train/inspect sample 분리** (`60b1445`): 학습 secondary 1과 inspect 8을
  독립 지정하도록 wrapper를 변경했다. 샘플 증가는 bounce depth 증가가 아니다.
- **9월 13일 login-shell patch check 실패** (`a1f5d32`): 2DGS patch 검사를 로그인 셸에서도
  실행 환경과 일치하도록 수정했다. 이 운영 수정은 학습 로스를 바꾸지 않는다.
- **9월 13일 training OptiX 런타임 선택** (`c89a86e`): 검증된 pinned driver를 노출해
  학습 시작 실패를 교정했다. GPU 전체 사용량과 개별 process VRAM을 구분해야 한다.
- **9월 13일 corrected intrinsics를 실제 풀코스에 연결** (`537a863`): 초기 캡처 수정만으로
  오래된 dataset이 자동 교체되지 않던 경로를 교정하고 camera intrinsics 검사를 추가했다.
- **9월 13일 Blender axial depth 재교정** (`7132df7`): Blender Z pass를 ray distance로
  잘못 해석했던 legacy 변환을 수정했다. 현재 캡처 depth는 axial Z-depth로 보존하고,
  explicit legacy ray-distance 입력만 해당 변환을 한다. intrinsics 오류와 별개 수정이다.
- **9월 14일 렌더 품질 3설정 비교** (`bc6482e`): 같은 pose/seed에서 128/1, 128/8,
  512/8 이미지를 병렬 열로 합성하는 평가 도구를 추가했다.
- **9월 14일 panel eval의 checkpoint 교체** (`b4fe12a`): 비교 wrapper가 실제 최신 Blender
  학습 결과를 참조하도록 바꿨다. 이전 checkpoint metric과 새 모델 metric을 혼용하지 않는다.
- **9월 14일 Blender UI relighting 안내 및 capture runtime** (`4f4a144`, `454cd23`):
  다른 컴퓨터에서 emissive mesh를 설정/저장/전송하는 문서와 pinned OptiX capture를 정리했다.
- **9월 14일 Blender mesh→AEGIR emitter 변환** (`675912c`, 이후 3D 확장):
  bbox/mesh 좌표를 saved→model transform으로 바꾸고 2D/3D support, orientation,
  declared linear RGB radiance를 변환한다. 현재 conversion의 radiance calibration은
  초기 또는 최종 reference emitter state를 명시 선택한다. reference emitter radiance의
  RGB별 중앙값 비율이며 6개 대응의 평균이나 mesh별 exact 물리 변환이 아니다.
  target relighting image로 fitting하지 않으며 새 조명 평가용 GT 변환은 실전 초기화와 별개다.

### D. 문서에 이미 있었지만 짧은 요약에서 빠진 중요한 세부사항

- **full-view material prior**: 이전 16×16 transport patch만 prior 감독하던 것을 전체
  유효 view로 넓혔다. photo/BRDF/visibility transport는 256-primary patch를 유지한다.
- **실제 freeze와 clipping 교정**: LR만 0이던 그룹을 requires_grad off로 바꾸고 stale
  gradients를 지웠다. frozen gradient가 emitter clipping에 섞이지 않으며 frozen geometry의
  BVH rebuild/refinement와 15K 이후 무의미한 prior/TV 계산을 생략한다.
- **hybrid 독립 schedule**: historical hybrid의 SH fade는 5K–10K,
  material decay는 5K–15K다. 최근 secondary-only에서는 SH fade 자체가 적용되지 않는다.
- **transport mode 저장**: monkey patch로만 only-mode를 바꾸던 것을 config/checkpoint에
  저장하여 resume가 다른 indirect mode로 바뀌지 않도록 했다.
- **8-sample segfault**: bounce launch를 8192-hit chunk로 나눠 한 launch의 ray 수를
  제한했다. sample count, RNG 순서 및 transport 수식은 바꾸지 않는 배치 수정이다.
- **단방향 planar → 방향 학습 planar**: 초기 2D 변경의 앞면 한정 방사를 교정하고
  양쪽 교차 및 abs(n·omega) Jacobian을 사용한다. 위아래 radiance는 독립 학습한다.
- **emitter Gaussian 중복 제외**: seed/native 2DGS 감독·densification에서 emitter
  영역을 제외하고, camera-visible emitter pixel은 emission only로 처리한다.
- **invalid 마스크의 과도한 삭제**: binary mask/weight 0.1 실험은 천장·블라인드를
  지워 rejected로 남긴다. 255 surface / 64 emitter-exempt / 0 invalid의 class mask,
  weight 0.02 및 protection 7로 교정했다.
- **window albedo 검정과 실제 alpha 구분**: albedo 비교의 invalid blackout은
  Gaussian 부재를 증명하지 않는다. valid-surface rendering/alpha compositing 수정과
  별도의 native invalid-opacity 재학습을 구분했다.
- **3D support 복원 및 checkpoint migration**: planar v13, directional planar v14,
  selectable 3D v15로 schema를 구분한다. legacy inference migration과 exact optimizer
  resume가 같은 기능은 아니다.
- **실전 provenance**: mm/metre, ray-distance/axial Z, residual GT metadata, 잘못된
  cache 재사용을 검사하고 fresh/resume에 dataset·geometry·prior content를 기록한다.
- **초기화 실패 후 수정 단계**: voxel persistence 탈락 → radius persistence →
  위치 오류 확인 → HDR ray multiview triangulation 순서다. radius01 checkpoint는
  실패한 초기화 실험이고 최신 결과로 재분류하지 않는다.
- **Conference 초기화 병목**: 동일 observation ID 형상 피팅을 pass별 memoize하고
  emitter 수락 시 score를 갱신한다. 검증 허용오차/fallback을 완화하지 않았으며
  진행 로그와 중복/피팅 횟수 진단을 추가했다.
- **denoiser**: inspect의 indirect HDR cross-bilateral filter는 opt-in이다.
  논문 시각화의 OptiX denoiser와 같은 구현이 아니며 기본 렌더와 별도 평가한다.

### E. 최신 완료 상태 및 렌더링 시간 측정

- 9월 16일 실제 Blender/Conference 최신 checkpoint를 읽어 두 결과 모두 step 20K,
  frozen geometry, SH-DC/white0/secondary-only 및 material freeze를 확인했다.
  이전 날짜의 '20K 아직 시작하지 않음'은 당시 상태 기록으로 유지한다.
- `scripts/benchmark_render_samples.py`와
  `scripts/run_blender_render_benchmark_10frames.sh`를 추가했다. Blender의 동일 test
  frames 140–149, 256×256, denoiser/SH off, 설정별 warmup 1장 후 10장을 측정했다.
- 한 장 중앙값: **128/1 8.974241초, 128/8 10.319985초, 512/8 30.776116초**.
  CUDA synchronize로 full-frame render를 측정하며 G-buffer/transport/CPU tensor transfer는
  포함하고 load/warmup/image 저장/metrics는 제외한다. 결과는
  `outputs/blender_practical3d_20260915_051804/render_benchmark_10frames_v1/`에 있다.
- 9월 16일 미반영 practical/HDR 초기화 코드와 새 평가 도구를 검증했다:
  **전체 pytest 146 passed**, 변경 Python Ruff 통과, 변경 shell 7개 syntax 통과.
  이는 회귀 검증이며 relighting/localization 강건성을 입증하는 평가가 아니다.

### F. 2026-09-17 — Conference 배경 scale-only geometry ablation

- 기존 `geometry_lr`는 배경 2D Gaussian의 center, tangent scale, rotation,
  opacity를 전부 같은 optimizer group으로 학습했다. 개별 속성을 독립적으로 고정하는
  `optimize_scene_{means,scales,rotations,opacities}` 설정과 대응 CLI/runner 옵션을
  추가했다. 기존 checkpoint/config에는 네 옵션의 기본값이 모두 true이므로 과거의
  trainable-geometry 의미는 유지된다.
- 첫 Conference 진단은 **scale만 학습**한다. center/rotation/opacity, emitter
  center/rotation, densification/pruning은 고정한다. geometry LR은 `1e-6`이고 배경
  scale은 native 2DGS 초기값의 `0.9–1.1배`로 clamp한다. 이 초기 scale anchor는
  exact-resume state에도 저장한다.
- [당시 scale-only 실험 스크립트](https://github.com/hohyun-nam/aegir/blob/61cbf90860df8e7f023e519d090eefe2435dc1c0/scripts/run_conference_scale_only_geometry_ablation.sh)는 당시 practical Conference
  geometry/data에서 geometry-fixed와 scale-only를 각각 secondary-only 128/1로 새로
  학습하고 128/8 30-view inspect를 수행한다. 두 결과의
  frame/seed/target/mask와 emitter 개수가 같은지 검사하고
  [당시 비교 이미지 스크립트](https://github.com/hohyun-nam/aegir/blob/61cbf90860df8e7f023e519d090eefe2435dc1c0/scripts/compose_geometry_scale_comparison.py)로 GT/fixed/scale-only/difference 비교
  이미지, contact sheet, GIF, metrics JSON을 생성한다.
- 이 실험은 Gaussian 수를 바꾸지 않는다. 따라서 결과 차이는 densification이나
  opacity pruning이 아니라 배경 tangent scale 최적화에서 발생한다.
- 검증: geometry 선택/scale clamp/exact-resume 관련 targeted **59 tests 통과**,
  OptiX tracer를 제외한 전체 **138 tests 통과**, 변경 Python Ruff와 shell syntax 통과,
  실제 기존 Conference 30-view artifact를 사용한 비교 composer smoke test 통과.
  전체 suite의 OptiX tracer 구간은 이 서버의 `surfel_tracer.RayTracer` 생성 중 native
  segfault로 중단되어 별도 GPU runtime 검증 대상으로 남겼다.

### G. 2026-09-17 — Conference 13개 과분할을 실제 두 스트립으로 교정

- Conference의 물리 조명은 두 개의 긴 천장 스트립이다. 기존 `hdr_multiview`가 만든
  13개는 실제13개 조명이 아니라 각 스트립의 불연속 HDR 밝은 영역을 별도 면으로
  피팅한 과분할이었다. 따라서
  `outputs/conference_practical3d_hdrmv_20260915/aegir/aegir_20k.pt`와 이를 기준으로
  시작한 scale-only 실험은 올바른 비교 기준에서 제외한다.
- 원시 후보를 모두 GT/depth 없이 복원한 뒤, 길쭉하고 방향이15도 이내이며 같은
  직선·평면에서 연속되는 support만 병합한다. 알려진 emitter 개수를 입력하거나
  Conference에만 맞춘2개 강제 제한은 사용하지 않는다. 작은 독립 패널6개는 그대로
  유지되는 회귀 테스트를 추가했다.
- 실제 Conference191-view 사전검사에서 원시13개 후보가 두 에미터로 병합됐다.
  결과와 투영 overlay는
  `outputs/conference_practical3d_hdrmv_20260915/emitter_init_hdr_multiview_merged_v3/`에
  저장했다. 초록 support가 두 긴 스트립을 따라가는 것을 확인했다.
- 비교 runner는 같은 교정 초기화에서 geometry-fixed와 scale-only를 모두 새로
  학습한다. 둘 다 emitter center/rotation과 adaptive split/prune은 고정하고 emitter
  scale, shared color, signed-axis gains, spread, falloff는20K까지 학습한다. emitter LR은
  0–5K `1e-3`, 5–15K `1e-4`까지 감쇠, 15–20K `1e-4`다. 배경 BRDF는15K부터
  고정되지만 emitter 학습은 계속된다.
- 실제 Conference preflight는 원시13개를2개로 병합하고 두 에미터가 각각38/40개의
  서로 다른 뷰에서 지지됨을 확인했다. 실제 Blender120-view는 병합 전후 모두6개이고
  중심·크기·재투영 수치도 유지됐다. targeted68 tests, OptiX tracer를 제외한 전체
  140 tests, Ruff와 shell syntax 검사를 통과했다.

### H. 아직 고치지 않은 문제와 계획

- Native IRGS의 opacity-composited expected hit는 strict discrete first surface hit가 아니다.
- Secondary NEE의 PDF surrogate bias는 sample 수를 늘려도 없어지지 않는다.
- origin offset 0.15는 접촉 차폐체를 건너뛸 수 있으며 originating Gaussian exclusion이 아니다.
- SH-DC 초기 albedo는 관측 조명의 그림자/하이라이트를 포함할 수 있다.
- 낮은 roughness에서의 GGX epsilon 문제, 관측되지 않는 조명 탐색, LDR-only 조명
  적응 및 localization feature/pose pipeline은 이 기록만으로 수정 완료라고 하지 않는다.
- 인터랙티브 preview/release full-render 설계와 localization을 논의했지만 구현하지 않았다.

### I. Git 전체 커밋 색인

아래는 이 저장소 baseline 이후 모든 기존 커밋을 빠짐없이 나열한다. 운영/문서/평가
커밋도 포함하며 각 행을 새 학습 수식 변경으로 취급하지 않는다. 9월 15일 practical
입력·HDR multiview·성능 수정과 9월 16일 benchmark/이력 통합은 이 절을 추가하는
커밋에 함께 보존한다. 이번 audit 이전 origin/main은 `e27f9a6`이었다.

| 날짜 | 커밋 | 변경 |
| --- | --- | --- |
| 2026-08-13 | [10127b2](https://github.com/hohyun-nam/aegir/commit/10127b2) | Establish AEGIR source and reproducibility baseline |
| 2026-09-06 | [b4392c0](https://github.com/hohyun-nam/aegir/commit/b4392c0) | Add six-axis emitters and corrected AEGIR transport pipeline |
| 2026-09-07 | [16bce88](https://github.com/hohyun-nam/aegir/commit/16bce88) | Constrain emitters and accelerate albedo learning |
| 2026-09-07 | [b54dccc](https://github.com/hohyun-nam/aegir/commit/b54dccc) | Fix Blender render intrinsics and legacy depth |
| 2026-09-07 | [51e6f4d](https://github.com/hohyun-nam/aegir/commit/51e6f4d) | Supervise material priors over full views |
| 2026-09-07 | [9e999a2](https://github.com/hohyun-nam/aegir/commit/9e999a2) | Restore SH DC albedo initialization with full-view prior comparisons |
| 2026-09-07 | [6c41fd9](https://github.com/hohyun-nam/aegir/commit/6c41fd9) | Bundle patched IRGS tracer and 2DGS rasterizer for portable installs |
| 2026-09-08 | [d4b0675](https://github.com/hohyun-nam/aegir/commit/d4b0675) | Freeze inactive gradients and finish SH transition before material freeze |
| 2026-09-08 | [d0a705f](https://github.com/hohyun-nam/aegir/commit/d0a705f) | Add persisted secondary-only training comparison |
| 2026-09-09 | [6eb8abc](https://github.com/hohyun-nam/aegir/commit/6eb8abc) | Add alpha filtering for 2DGS mesh export |
| 2026-09-09 | [56dd5f9](https://github.com/hohyun-nam/aegir/commit/56dd5f9) | Revert "Add alpha filtering for 2DGS mesh export" |
| 2026-09-12 | [e06f792](https://github.com/hohyun-nam/aegir/commit/e06f792) | Adopt Blender-validated lighting defaults |
| 2026-09-12 | [93b23d6](https://github.com/hohyun-nam/aegir/commit/93b23d6) | Add six-panel Blender relighting scene builder |
| 2026-09-12 | [e74344c](https://github.com/hohyun-nam/aegir/commit/e74344c) | Add diagonal two-panel relighting condition |
| 2026-09-12 | [ae13074](https://github.com/hohyun-nam/aegir/commit/ae13074) | Add emitter-subset relighting evaluation |
| 2026-09-12 | [f4125cc](https://github.com/hohyun-nam/aegir/commit/f4125cc) | Document LDR-only light adaptation plan |
| 2026-09-12 | [3c5adde](https://github.com/hohyun-nam/aegir/commit/3c5adde) | Fix eight-sample secondary training launch size |
| 2026-09-13 | [f8bd564](https://github.com/hohyun-nam/aegir/commit/f8bd564) | Add planar emitters and exclude lights from geometry |
| 2026-09-13 | [60b1445](https://github.com/hohyun-nam/aegir/commit/60b1445) | Separate training and inspect bounce samples |
| 2026-09-13 | [a1f5d32](https://github.com/hohyun-nam/aegir/commit/a1f5d32) | Fix 2DGS patch check in login shells |
| 2026-09-13 | [c89a86e](https://github.com/hohyun-nam/aegir/commit/c89a86e) | Load pinned OptiX driver for AEGIR training |
| 2026-09-13 | [537a863](https://github.com/hohyun-nam/aegir/commit/537a863) | Use corrected Blender intrinsics in full pipeline |
| 2026-09-13 | [7132df7](https://github.com/hohyun-nam/aegir/commit/7132df7) | Fix Blender axial depth handling |
| 2026-09-14 | [1deef25](https://github.com/hohyun-nam/aegir/commit/1deef25) | Restore learned directional planar emitters |
| 2026-09-14 | [6cc422b](https://github.com/hohyun-nam/aegir/commit/6cc422b) | Fix invalid-mask geometry erosion |
| 2026-09-14 | [6348d34](https://github.com/hohyun-nam/aegir/commit/6348d34) | Make invalid-opacity protection radius configurable |
| 2026-09-14 | [9c131ef](https://github.com/hohyun-nam/aegir/commit/9c131ef) | Adopt Blender invalid-mask 0.02 protect-7 defaults |
| 2026-09-14 | [bc6482e](https://github.com/hohyun-nam/aegir/commit/bc6482e) | Add three-setting render quality comparison |
| 2026-09-14 | [b4fe12a](https://github.com/hohyun-nam/aegir/commit/b4fe12a) | Point panel relighting eval at current Blender model |
| 2026-09-14 | [4f4a144](https://github.com/hohyun-nam/aegir/commit/4f4a144) | Document Blender relighting emitter setup |
| 2026-09-14 | [454cd23](https://github.com/hohyun-nam/aegir/commit/454cd23) | Load pinned OptiX driver for Blender capture |
| 2026-09-14 | [675912c](https://github.com/hohyun-nam/aegir/commit/675912c) | Convert Blender meshes to AEGIR emitters |
| 2026-09-15 | [e27f9a6](https://github.com/hohyun-nam/aegir/commit/e27f9a6) | Restore selectable 3D ellipsoid emitters |
