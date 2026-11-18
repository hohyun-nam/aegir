# 서버 파이프라인 병합 및 검증 — 2026-10-06

공통 HEAD `a217f50afc005d3e96f0a54379e4f1934b282c74`와 현재 서버 작업을 ancestor/server/local 3-way로 병합했다. HEAD/index는 그대로이며 변경은 worktree에 남겨 두었다. 전달본은 저장소 밖 `/workspace/incoming/server_migration_20261006/`에 해제했다. 로컬 reference 전체 교체, reset/clean, CUDA binary 복사는 수행하지 않았다.

## 구현과 보존

- LDR SAM3 mask → 학습된 2DGS surface points → distinct-view voxel persistence → DBSCAN/PCA → 3D ellipsoid 초기화를 추가했다. 초기화 진단에 HDR 관측/외부 predicted depth/GT emitter 사용 여부가 모두 false로 기록된다.
- 실제 alpha-compositing contribution의 다중 뷰 투표로 Gaussian emitter ID를 연결하고, tag-only green PLY와 analytic-emitter PLY를 생성한다. retag는 optimizer/RNG resume 상태를 제거한 새 초기화 artifact다.
- Tagged fixture surfels는 camera emission에 포함하고, all-lights-on IRGS visibility/secondary-hit BVH에서 모든 ID의 fixture를 제외한다. Off fixture가 BRDF/occluder로 돌아오는 기존 relighting 정책을 유지한다.
- RGB fixed-pose triangulation seed와 ZoeDepth Pearson weight 0.2의 native 2DGS 30K runner를 추가했다. emitter-inclusive geometry이며 10K/20K/30K를 저장한다.
- 구조와 emitter support를 고정한 AEGIR 5K, train 128 / inspect 256 runner를 추가했다. 재질·radiance/방향성·LDR exposure를 학습한다.
- `adapt-lighting`, disabled emitter definition 처리, 서버의 geometry optimization opt-in/default freeze, lighting adaptation provenance, 기존 localization/Blender relighting/continuation 문서와 스크립트를 보존했다.
- `aegir/irgs_tracer.py`의 pinned 580.173.02 preload와 `aegir/denoising.py`는 병합 전과 hash가 동일하다. 공통 runner도 pinned OptiX를 기본으로 유지하도록 병합했다. 서버 전용 파일 37개의 byte-for-byte 보존 목록은 아래 audit에 있다.
- External `scene/dataset_readers.py`에는 normal `.npy` 호환 patch만 적용했다. image normal만 127.5 decoding하고 `.npy` normal은 재변환하지 않는다. External Git metadata/CUDA source/binary는 변경하지 않았다.

## 서버 경로와 입력 복구

Python은 `/workspace/.conda/envs/dnsplatter/bin/python`, architecture는 `12.0`이다. 새 runner의 기본값에서 `/home/hohyun`, architecture `8.6`, 삭제된 `*_sam3mv_inclusive_2dgs20k_tagged_initply_20260929` 경로를 제거했다.

현재 복구한 입력은 다음과 같다. 아래 경로는 repository root 기준이다.

| 입력 | Blender | Conference |
| --- | --- | --- |
| LDR SAM view | `outputs/server_sam3_ldr_inputs_20261006/blender/sam3_ldr_view` | `outputs/server_sam3_ldr_inputs_20261006/conference/sam3_ldr_view` |
| RGB seed / native view root | `outputs/blender_2dgs_triangulated_seed_30k_20261006` | `outputs/conference_2dgs_triangulated_seed_30k_20261006` |
| Zoe cache source | `datasets/aegir_nerfstudio/blender_my_room_single_room_heightdiverse_20260901_intrinsics_depthfixed_v2_rgb_zoedn_emitter_inclusive_meanonly20k_v1_20260917/predicted_depth` | `datasets/aegir_nerfstudio/fipt_real_conference_rgb_zoedn_emitter_inclusive_scaleonly20k_v1_20260917/predicted_depth` |

Zoe cache source 경로는 workspace 기준이다. `fixed_pose_triangulated_seed_filtered.ply`와 `geometry_triangulated_filtered_view/`를 각 native view root 아래 생성했으며 hash/provenance 검사를 통과했다. Blender seed는 18,633점, Conference는 4,755점이다. seed/SAM/Zoe source의 카메라 포즈와 intrinsics가 일치함을 확인했다. Blender sparse seed는 기존 전체 180뷰 COLMAP이므로 strict train-only benchmark seed가 아니다.

서버 SAM 캐시는 두 장면 모두 40뷰의 비연속 샘플이다. `prepare_sam3_training_view.py --allow-partial-cache`는 source frame array를 유지하고 train selection만 바꾸므로 원본 `frame_XXXX.pt` prior index가 보존된다. RGB가 cached SAM input과 pixel-wise 동일하고 mask가 binary/native-size인지 검증했다. 모든 frame의 GT depth/normal/material/emitter/HDR/EXR 경로를 제거했다. 새 `validate_sam3_ldr_view`는 dataset 로딩 전에 LDR 선언, forbidden path, manifest/mask/RGB hash를 확인한다. LDR prior의 cached HDR observation도 로드하지 않는다.

부분 캐시는 `sam3_provenance.partial_cache=true`로 기록한다. 기본 5K runner는 부분 캐시의 대규모 학습을 거부하며, 의도적인 subset 실험에만 `ALLOW_PARTIAL_SAM=1` / `--allow-partial-sam`을 쓸 수 있다. 전체 120/191뷰 재현에는 full SAM cache를 전송하거나 기존 서버 SAM3 runner로 LDR/`--gt-mask-mode none` inference를 다시 수행해야 한다. 전달 bundle에는 로컬 30K/5K checkpoint와 full SAM masks가 없고 서버에서도 해당 로컬 결과는 확인되지 않았다. 기존 cache에는 필요한 mask가 있으므로 이번 검증에 새 SAM inference/weight 다운로드는 필요하지 않았다.

## 완료한 검증

- 기존/병합/서버 regression suite 180 passed, external `.npy`/PNG normal loader 2 passed: 합계 182 tests. CLI `adapt-lighting` 및 geometry opt-in 호환성을 확인했다. `compileall`, `bash -n`, `git diff --check`도 통과했다.
- `PREPARE_ONLY=1`로 두 장면의 RGB seed/native view를 생성하고 인증했다. HDR/EXR frame paths와 GT metadata를 native geometry view에서도 제거했으며 broken symlink가 없다.
- 검증된 서버 emitter-inclusive **20K geometry**와 **40뷰 SAM cache**로 두 장면의 initialize-only를 실행했다. 두 초기화 checkpoint 모두 step 0 / loss history 0이며 5K fit을 호출하지 않았다.

| 검증 | Blender | Conference |
| --- | --- | --- |
| 초기화 emitters | 6 | 12 |
| visible-contribution tags | 3,249 | 2,244 |
| AEGIR GPU smoke | 2 steps, pass | 2 steps, pass |
| AEGIR peak Torch reserved | 0.483 GiB | 0.274 GiB |
| native 2DGS GPU smoke | 2 steps, RGB seed 18,633, Pearson 0.2, pass | 2 steps, RGB seed 4,755, Pearson 0.2, pass |

AEGIR smoke는 64px / 64 primary rays / BRDF-light 2 samples로 제한했다. OptiX hit visibility 약 0.1, miss 1.0과 nonzero/finite opacity gradient를 확인했다. all-lights-on BVH에 emitter-tagged rows가 없고, all-lights-off 후보 BVH에는 fixture rows가 복귀함을 확인했다. 카메라에는 fixture emission이 나타났고, 2-step 동안 재질은 바뀌지만 scene geometry/emitter support는 정확히 동일했다. 256px camera preview도 finite 값을 확인했다. preview는 direct-only/2 samples이며 정식 128-sample/8-secondary-sample inspect 품질 평가가 아니다.

Native smoke는 원본 해상도에서 `--iterations 2 --depth_supervision --depth_loss_type pearson --lambda_depth 0.2 --densify_until_iter 0`으로 실행했다. checkpoint와 PLY를 생성하고 저장 tensor/optimizer가 finite인지 확인했다. 기존 external loader는 RGB만 축소하고 depth/mask를 축소하지 않아 첫 `--resolution 8` smoke는 shape mismatch로 실패했다. authoritative runner와 같은 `--resolution 1`에서 통과했으며 이 legacy downsampling 문제는 별도로 남겨 두었다. legacy CUDA `simple_knn`은 env에 없지만 서버에 이미 있던 SciPy cKDTree fallback으로 seed scale 초기화가 성공했다. 서버 rasterizer/IRGS extensions를 사용했으며 로컬 바이너리를 가져오지 않았다.

Conference의 12개 후보 중 2개는 accepted Gaussian vote tag가 없다. 초기화 후보 수나 tag 수는 검증 조건의 결과이며 로컬 30K/full-view 결과와 같은 품질을 의미하지 않는다. 원래 로컬 snapshot(6/10 emitters)을 이 smoke 결과로 대체하지 않는다.

## 검증 결과 위치

각 장면의 `outputs/server_migration_validation_20261006/{blender,conference}/` 아래:

- `emitter_initialization/aegir_initialization.pt`, `emitter_initialization.json`
- `sam3_vote_tags_v1/aegir_retagged.pt`, `votes.pt`, `tagging.json`
- `sam3_vote_tags_v1/map_with_emitter_tagged_gaussians_green.ply`
- `sam3_vote_tags_v1/map_with_initial_emitters.ply`
- Blender: `gpu_smoke_v2/{aegir_smoke.pt,smoke_result.json,camera_256.png}`
- Conference: `gpu_smoke/{aegir_smoke.pt,smoke_result.json,camera_256.png}`
- Blender: `native_2dgs_smoke_v2/{chkpnt2.pth,smoke_result.json,point_cloud/iteration_2/point_cloud.ply}`
- Conference: `native_2dgs_smoke/{chkpnt2.pth,smoke_result.json,point_cloud/iteration_2/point_cloud.ply}`

학습 30K/5K는 시작하지 않았다. source data/기존 checkpoints는 덮어쓰지 않았다. 실패한 첫 smoke 결과는 보존하고 새 output으로 재시도했다. 새로 생성한 중간 view/tag diagnostic의 수정 전 버전은 audit 아래로 recoverable move했다.

## 재검증 및 후속 실행

학습 없이 입력을 재검증하는 명령:

```bash
cd /workspace/hohyun_ws/AEGIR
PREPARE_ONLY=1 bash scripts/run_2dgs_triangulated_seed_30k.sh blender
PREPARE_ONLY=1 bash scripts/run_2dgs_triangulated_seed_30k.sh conference
```

검증된 20K geometry로 initialization/tag artifact를 재사용하는 명령:

```bash
INITIALIZE_ONLY=1 \
GEOMETRY=/workspace/hohyun_ws/AEGIR/outputs/2dgs_blender_my_room_single_room_heightdiverse_20260901_intrinsics_depthfixed_v2_emitter_inclusive_meanonly20k_v1_20260917/blender_my_room_single_room_heightdiverse_20260901_intrinsics_depthfixed_v2/nerfstudio_models/step-000020000.pt \
OUTPUT_ROOT=/workspace/hohyun_ws/AEGIR/outputs/server_migration_validation_20261006/blender \
bash scripts/run_triangulated30k_tagged_aegir5k.sh blender

INITIALIZE_ONLY=1 \
GEOMETRY=/workspace/hohyun_ws/AEGIR/outputs/2dgs_fipt_real_conference_emitter_inclusive_scaleonly20k_v1_20260917/fipt_real_conference/nerfstudio_models/step-000020000.pt \
OUTPUT_ROOT=/workspace/hohyun_ws/AEGIR/outputs/server_migration_validation_20261006/conference \
bash scripts/run_triangulated30k_tagged_aegir5k.sh conference
```

초기화/태그 cache 재사용 시 data/geometry/prior content와 source initialization/view hash를 확인한다. 큰 학습은 사용자의 다음 요청 후 full SAM views/selected 30K geometry를 지정해 실행한다. 새 wrappers에서 `SOURCE_DATA`, `PREDICTED_VIEW`, `DEPTH_DIR`, `DATA`, `GEOMETRY`, `SAM_DATA`, `PRIORS`, `RUN_ROOT`, `OUTPUT_ROOT`, `INIT_ROOT`, `TAG_ROOT`, `DNS_ENV`, `TWO_DGS_ROOT`를 override할 수 있다.

백업/audit: `/workspace/incoming/server_migration_20261006_audit/`의 `server_source_before.tar.gz`, `server_changes_before.patch`, `server_hashes_before.json`, `preserved_server_files.json`, `migration_only.patch`, `verification_summary.json`. 초기 서버 작업과 이번 변경을 구분해 복구/검토할 수 있다.
