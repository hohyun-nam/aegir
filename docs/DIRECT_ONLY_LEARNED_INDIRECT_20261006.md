# Direct-only sampling + learned diffuse indirect — 2026-10-06

사용자 요청으로 구현한 속도 실험 모드다. 기존 explicit transport의 기본 동작과
서버 미커밋 변경을 유지하며, 새 모드 선택 시에만 간접광을 학습 가능한 전역
diffuse irradiance로 대체한다. geometry/SAM 초기화 정책은 바꾸지 않는다.

## 계산

`render.lighting_mode = direct_only_learned_indirect`

```text
2DGS camera rasterization → position / normal / BRDF
  direct: emitter surface samples → finite shadow visibility → diffuse + GGX
  indirect: world normal → 9 RGB SH coefficients → diffuse irradiance
  camera-visible fixture emission: existing tagged-surface compositing
HDR = direct + indirect + emission
```

- 직접광은 기존 `sample_surface_total`과 선택한 emitter의 `light_pdf_surrogate`
  를 사용한다. PDF에는 emitter 선택 확률이 포함된다. BRDF/light 혼합 PDF를
  사용하지 않고 light sample 수로만 평균한다. `max(n·wi, 0)`는 유지한다.
- cosine sample 분기 자체를 호출하지 않는다. 에미터 ray가 가려져도 해당 hit의
  재질 조회, DN SH proxy, secondary NEE를 계산하지 않는다. 직접광 전용 경로는
  `IRGSVisibilityTracer.visibility`를 호출하여 material features를 전달하지 않는다.
  native tracer 자체는 기존 color/normal/depth/alpha 인터페이스를 유지한다.
- **기존 ellipsoid PDF surrogate는 유지했다.** 이 모드는 ellipsoid sampling을
  정확한 물리적 area-light estimator로 교정한 작업이 아니다. planar ellipse는
  기존 area-to-solid-angle PDF를 사용한다.
- `E(n) = max(sum_k coefficients[k] Y_k(n), 0)`로 cosine convolution이 이미
  반영된 irradiance를 직접 표현한다. SH basis는 world 좌표의 직교 정규 real SH다.
  `indirect = albedo * (1-metallic) * E(n) / pi`. 색 값은 선형이고 1을 넘을 수 있다.
  HDR 입력이나 GT depth/emitter 정보가 필요하지 않으며 LDR camera loss를 유지한다.
- 전역 SH이므로 같은 normal/material은 위치와 무관하게 같은 간접광을 받는다.
  공간적 occlusion과 간접 specular를 표현하지 않는다. 음수 재구성은 zero clamp하며,
  clamp 전 coefficient energy도 규제하여 숨은 음수 에너지를 억제한다.

## 학습 설정

| 설정 | 기본값 |
| --- | --- |
| light samples | 32 (새 CLI/태깅 runner 모드 선택 시) |
| cosine samples / secondary rays | 0 |
| 초기 간접 irradiance RGB | 0.03, DC만 초기화 |
| SH Adam LR | 0.001 |
| SH 학습 시작 | global step 100 |
| sphere-average SH 제곱 에너지 규제 | 0.1 |
| non-DC 제곱 에너지 추가 규제 | 0.1 |
| geometry | 기존 고정 기본값 |
| tagged training / inspect 해상도 | 128 / 256 |

CLI는 `--indirect-irradiance-init`, `--indirect-lr`, `--indirect-start-step`,
`--indirect-energy`, `--indirect-directional`을 제공한다. 작은 초기값은 warmup 중에도
렌더에 포함되지만 SH optimizer는 시작 step 전까지 고정된다. 재질 prior와 기존
exposure 정책은 유지한다. 이 제약들은 ambient/direct/material 분해의 유일성을
보장하지 않는다. loss history에서 direct/indirect 평균과 indirect prior를 함께 확인한다.

기존 일반 training command에 아래 인자를 추가하면 된다:

```bash
--lighting-mode direct_only_learned_indirect --num-light-samples 32
```

SAM contribution-tagged 초기화에서 fresh fit할 때:

```bash
/workspace/.conda/envs/dnsplatter/bin/python scripts/train_tagged_surface_scene.py \
  --initialization /path/to/sam3_vote_tags_v1/aegir_retagged.pt \
  --data /path/to/sam3_ldr_view --priors /path/to/priors \
  --output /path/to/fresh_run --lighting-mode direct_only_learned_indirect
```

`run_triangulated30k_tagged_aegir5k.sh`도 `LIGHTING_MODE=direct_only_learned_indirect`
와 선택적인 `RENDER_SAMPLES=32`를 지원한다. 새 fit 디렉터리는
`aegir_tagged_surface_5k_direct_sh`여서 기존 explicit 결과와 충돌하지 않는다.
기존 30K geometry/full-SAM 준비 여부와 partial-cache 제한은 그대로 적용된다.
위 명령은 사용 예시이며 이번 구현 과정에서 5K/30K 학습을 실행하지 않았다.

## 저장·inspect·relighting

### 2026-10-07: fresh AEGIR 20K schedule

새 실험은 기존 SAM-tagged 초기화에서 다시 시작한다. 완료된 5K 모델에 학습
step만 추가하는 exact resume이 아니다. 20K는 AEGIR BRDF/light fit의 길이이며
입력은 여전히 서버의 20K geometry와 40-view partial SAM이다.

| 파라미터 | 0–100 | 100–12K | 12K–20K |
| --- | --- | --- | --- |
| Albedo LR | 0.003 | 0.003 | 0.003 → 0 |
| Roughness/metallic LR | 0.001 | 0.001 | 0.001 → 0 |
| Emitter radiance LR | 0.001 | 0.001 | 0.001 → 0.0001 |
| Global diffuse SH LR | 0 | 0.001 | 0.001 → 0.0001 |
| Geometry/emitter support | fixed | fixed | fixed |

후반은 선형 감소다. `indirect_final_lr=None`인 기존 5K checkpoint의 SH는 원래의
상수 LR를 유지하고, exact resume 시 저장된 schedule config를 그대로 복원한다.
`adapt-lighting`은 SH initial/final LR를 모두 0으로 설정해 emitter-only 계약을 유지한다.

태깅 trainer에 `--freeze-start`, `--freeze-end`, `--indirect-final-lr`를 추가했다.
기본 material decay 시작은 steps의 60%, 끝은 final step이다. learned-indirect
5K 초과 fresh fit은 SH final LR 기본값 0.0001을 사용한다. 아래 runner는 모든
schedule 값을 명시하여 초기 제안인 warmup100 / plateau12K / decay20K를 적용한다.

```bash
tmux new-session -d -s aegir_direct_sh_20k \
  'bash /workspace/hohyun_ws/AEGIR/scripts/run_direct_sh_20k_both.sh'
tmux attach -t aegir_direct_sh_20k
```

Blender fitting과 milestone inspect를 끝낸 뒤 Conference를 순차 실행한다.
출력은 `outputs/{blender,conference}_direct_sh_20k_<KST timestamp>/`, 공통 로그는
`outputs/direct_sh_20k_<KST timestamp>.log`다. 이전 5K 결과와 다른 새 폴더를 쓴다.
20K fitting 완료 후 5K/10K/15K/20K checkpoint를 동일한 train 30뷰/256px로 검사한다.
중간 결과는 `inspect_005000`, `inspect_010000`, `inspect_015000`, 최종은 `inspect`다.
`--inspect-steps`를 생략하면 final step만 inspect한다.

이번 변경은 스케줄 경계와 SH decay 중 exact-resume을 CPU 테스트로 검증했다.
전체 `pytest -q tests`는 202 passed이며 compile/shell/diff 문법 검사도 통과했다.
runner는 `bash -n` 및 실제 입력 경로를 확인했으며 20K 학습은 자동 시작하지 않았다.

### Checkpoint/rendering compatibility

- v17 checkpoint의 `indirect_light_state`에 SH를, training optimizer state에 별도
  `indirect` Adam을 저장한다. config, 기존 dataset/input provenance, exposure/RNG/
  optimizer state도 유지한다. 필수 SH가 없는 새 모드 checkpoint는 로딩을 거부한다.
- v1–v16 inference와 v14–v16 exact resume을 지원한다. 기존 3D ellipsoid support가
  v16→v17 전환 때문에 planar로 변환되지 않도록 export/inspection도 갱신했다.
- inspect는 저장된 SH를 복원한다. sample override가 없으면 새 모드는 저장된
  light sample 수, 기존 모드는 historical 128을 사용한다. camera path와 기존
  benchmark script에도 SH 복원을 연결했다. median filter 동작은 유지한다.
- SH는 **captured lighting condition에 고정**된다. 개별 emitter on/off, light scale,
  색 변경 시 직접광/emission만 바뀌고 SH는 그대로다. 전역 ambient가 조명을 끈
  상태에도 남는 근사다. 광원별 간접광 응답을 구현한 것으로 해석하면 안 된다.
- `adapt-lighting`의 emitter-appearance-only 계약을 보존하기 위해 저장된 SH를
  복원하되 LR=0으로 고정한다. 새로운 조명 조건의 간접광 적응은 별도 후속 작업이다.
- `render_full_frame`을 직접 호출하는 코드는 `load_indirect_light(payload, device)`의
  반환값을 `indirect_light=`로 전달해야 한다. 모델 누락 시 새 초기값으로 조용히
  렌더링하지 않고 오류를 낸다.

## 검증

Python `/workspace/.conda/envs/dnsplatter/bin/python`, GPU architecture `12.0` 및
서버 pinned OptiX를 사용했다. CUDA source/binary 및 median filter는 변경하지 않았다.

- 전체 `pytest -q tests`: **192 passed**.
- 새 10 tests: analytic diffuse/DC/방향성, energy/clamp, light-only PDF와 샘플 수 정규화,
  forbidden cosine/secondary 경로, visibility/SH gradient, no-emitter/mask/alpha/metallic,
  fixture emission, 모델 누락, warmup, legacy ellipsoid 호환성을 검증한다.
- CPU 실제 trainer에 작은 raster fixture를 연결한 exact-resume 테스트에서 SH,
  scene/emitter state, Adam moment, 다음 step loss가 uninterrupted 실행과 bitwise 일치했다.
- 두 장면 각각 **2-step** GPU smoke: 실제 tag-inclusive raster, native finite visibility,
  SH gradient/update/save/restore, resume state 복원, geometry/support bitwise 고정,
  BVH all-on exclusion/off-restoration, 256px camera preview finite를 확인했다.
  smoke는 SH update 검증을 위해 warmup=0, light samples=2로 한정했다.

64 primary-point shading만 비교한 synchronized median (warmup 제외, 3회):

| 장면 | 기존 BRDF128+light128, secondary1 | 새 light32, secondary0 |
| --- | --- | --- |
| Blender | 0.691489 s | 0.008211 s |
| Conference | 0.936838 s | 0.013007 s |

이미 rasterized된 동일 patch와 동일 scene/material/emitter를 사용했다. 이 수치는
rasterization, 전체 이미지 chunking/I/O, 학습 backward, 품질 수렴을 포함하지 않는다.
샘플 budget과 간접광 모델을 함께 바꾼 비교이므로 동일 품질/동일 budget 속도 비교가 아니다.
기존 서버 20K geometry와 40뷰 SAM cache 기준이며 로컬 30K/full-view 품질 검증이 아니다.

산출물은 `outputs/direct_sh_validation_20261006/{blender,conference}/`:

- `aegir_smoke.pt`, `smoke_result.json`, `camera_256.png`
- `aegir_smoke.step00002.pt` (중간 저장)

구현 전 source backup: `/workspace/incoming/direct_sh_implementation_20261006.dCqVcc/source_before.tar.gz`.
