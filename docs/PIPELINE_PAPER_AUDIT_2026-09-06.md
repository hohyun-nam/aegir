# 현재 파이프라인 / AEGIR 논문 대조

검토일: 2026-09-06. 코드, 최신 Conference 체크포인트, 평가 metadata 및 작은 CPU 수치 재현을 검토했다. 이후 같은 날 visibility segment, HDR emitter 초기화, 중성 albedo 옵션, 카메라-visible emission, 6축 방향성 emitter를 차례로 구현하고 회귀 테스트를 추가했다.

기준 논문: [AEGIR arXiv v1](https://arxiv.org/html/2606.28635v1). [저자 페이지](https://darkgeekms.github.io/projects/aegir/)는 검토 시점에 Code Coming soon으로 표시된다. 논문에 명시되지 않은 구현 세부사항을 저자 구현의 확정 사실로 취급하지 않는다. 기존 8월 audit 문서는 과거 상태/완료 전후 설명이 혼재하므로 현재 코드와 구분한다.

## 검토 대상과 실제 설정

- `outputs/aegir_conference_secondaryonly_bounce1_raster256_20260904/fipt_real_conference/aegir_20k.pt`
- `outputs/aegir_conference_shonly_raster256_20260904/fipt_real_conference/aegir_20k.pt`
- 공통 geometry source: `outputs/2dgs_depth_pearson_20k_20260823/fipt_real_conference/nerfstudio_models/step-000020000.pt`
- 두 체크포인트 모두 213,856 Gaussian, 최종 emitter 2개, geometry LR 0, scene culling off, raster 256, origin offset 0.15, primary cosine/light 각 128, bounce sample 1.
- 로그 `losses`의 801개 기록에서 secondary-only의 SH weight는 0, SH-only는 1이다. 두 실험의 material LR은 5K부터 감소하여 15K에 0이 된다.
- 최신 실험은 2DGS 20K + ZoeDepth Pearson supervision을 사용한다. GT depth 실험은 별도 실험이며 현재 기본값으로 해석하면 안 된다.
- 일반 `AEGIRConfig()`는 여전히 raster 128 / geometry LR 1e-5 / origin offset 0.05 / culling on이다. Conference wrapper와 체크포인트의 설정이 기본 dataclass와 다르다.

## 논문과의 차이

다음 논문 사실은 [§3.3–3.4, Appendix A–C](https://arxiv.org/html/2606.28635v1)를 기준으로 한다.

| 항목 | 논문 | 현재 파이프라인 |
|---|---|---|
| Geometry 사전학습 | DN 초기화 10K, 2D Gaussian | native 2DGS + Zoe Pearson 20K |
| AEGIR geometry | 초반 최적화, 이후 freeze | 최신 실험 처음부터 freeze |
| Material LR | 5K–15K 감소, 이후 freeze | 동일 |
| 학습 indirect | SH proxy | 기본 코드는 SH→explicit fade; 최신 비교는 각각 only |
| Inference indirect | diffuse 방향 sampling, emitter hit 시 emission | emitter surface NEE + BRDF / light surrogate |
| Photo loss | HDR render를 LDR로 변환 후 비교 | Conference는 log1p HDR L1 + display SSIM |
| Primary light 개수 | emitter별 floor 분배, 남는 수 버림 | 랜덤 시작 cyclic 배분, 정확히 128개 |
| Light PDF | 명시적 surrogate | 같은 surrogate, secondary NEE에도 사용 |
| BRDF | Lambertian + GGX, diffuse에 (1-F) 없음 | 같은 형태 |
| Emitter 방향성 | Eq.1의 제곱 내적으로 앞뒤 대칭 | Eq.1 위에 학습 가능한 local ±X/±Y/±Z scalar gain을 추가; 설정으로 원 논문 대칭 모델 재현 |
| Geometry tracer | IRGS 2D tracer | IRGS + expected-hit 집약, offset 0.15 |
| 초기화 | 밝은 RGB 역투영 + DBSCAN/PCA | HDR 점수 + persistent 후보 + DBSCAN/PCA |

논문의 SH freeze/교체 schedule과 learning-rate schedule을 혼동하면 안 된다. 5K–15K SH fade는 로컬 변경이다. 논문의 secondary inference는 emitter NEE가 아니다. 논문의 DN 학습 depth loss 세부 설정은 공개 본문만으로 현재 Zoe/Pearson과 일치한다고 확인할 수 없다.

## 이전 설명 정정

1. `pL`의 cosine/Jacobian 누락은 로컬만의 재현 실수가 아니다. 논문도 Eq.11–13에서 정규화된 실제 PDF가 아닌 surrogate라고 명시한다. 물리적 정확성의 한계와 논문 불일치를 구분해야 한다.
2. SH에 secondary surface 재질이 포함되어 있다는 이유만으로 double shading이라고 할 수 없다. x'의 outgoing radiance를 x의 incoming radiance로 쓰고 x의 BRDF를 곱하는 것은 올바른 두 표면 transport 구조다. 실제 문제는 SH의 색 공간/단위, 고정된 원래 조명, emitter와의 gradient 연결 부재다.
3. radiance Le에 별도 1/d²를 곱하지 않는 것은 곧 거리 감쇠 누락을 뜻하지 않는다. 현재 estimator에 d²가 들어 있다. 추가 곱셈은 중복 감쇠가 될 수 있다. 유한 면광원의 최종 조도가 모든 거리에서 정확히 1/d²일 필요도 없다.
4. 앞서 적은 ellipsoid Jacobian 식은 선택 emitter의 한 표면점 branch에 대한 밀도 변환이다. closed ellipsoid의 앞/뒤 교차점이나 여러 emitter가 같은 방향에 대응하면 방향의 주변 PDF는 관련 preimage의 기여까지 고려해야 한다. cosine 하나만 덧붙여 정확한 MIS가 완성되는 것은 아니다.
5. 카메라에 보이는 emission 누락은 로컬 코드에서 확인되지만, 논문 저자 코드에도 같은 누락이 있다고 단정할 수 없다.

## 확인된 문제와 영향

### 1. Visibility가 에미터 거리에서 trace를 자르지 않는다 — 수정 완료

위치: `aegir/irgs_tracer.py:274–291,329–347`, 외부 `surfel_tracer/src/optix/gaussiantrace_forward.cu:27–108`.

수정 전에는 scene ray 전체를 trace해 누적 alpha A와 weighted depth D를 얻고 `t=D/A+offset`을 만든 후 `V=1-A*sigmoid(k*(d_light-b-t))`를 평가했다. 에미터 뒤쪽 Gaussian도 A와 t에 섞일 수 있어 서로 떨어진 표면에서는 구간 투과율과 다른 값이 됐다.

실제 `soft_visibility` 함수로 재현한 예:

- 앞 Gaussian: 거리 1, alpha 0.2.
- 에미터: 거리 2.
- 뒤 Gaussian: 거리 3, alpha 0.99.
- 합성 가중치: 0.2, 0.792. A=0.992, expected depth=2.596774.
- bias .001, steepness 80에서 현재 V=1.0. 에미터 앞 구간만 계산하면 V=0.8.

현재 AEGIR glue가 ray별 `d_light-origin_offset-visibility_bias`를 native tracer에 전달하고, OptiX forward/backward가 그 endpoint에서 trace를 중단한다. 따라서 에미터 뒤 Gaussian은 alpha, expected depth, hit material/normal 및 SH proxy에 들어가지 않는다. 기존 soft before-light gate는 경계 완화용으로 유지했다. 앞 alpha 0.2 / 에미터 z=2 / 뒤 alpha 0.99의 CUDA 회귀 테스트에서 visibility 0.8과 앞 hit z=1을 확인했다. 다만 expected-hit 자체의 문제는 아래 2번처럼 남아 있다.

### 2. Expected hit를 실제 secondary surface처럼 사용 — 우선순위 높음

위치: `aegir/irgs_tracer.py:288–303,347–349`, `aegir/render.py:271–338`.

점, normal, material이 여러 hit의 평균이다. 서로 다른 표면이 섞이면 실제 표면이 없는 위치에서 섞인 BRDF로 반사광을 계산할 수 있다. sample 1→8은 이 대표 표면의 부정확성을 없애지 않는다.

수정 후보는 first meaningful hit, opacity 가중 hit 확률 샘플링, 혹은 각 hit transport의 합이다. 단순 nearest hit는 희미한 floater를 선택할 수 있어 자동으로 정답이 되지 않는다. 논문의 first secondary hit 서술만으로 저자의 내부 hit 집약 방식까지 확정할 수 없다.

### 3. 0.15 origin offset은 장면별 편법 — 현 artifact 관련성 높음

위치: `aegir/irgs_tracer.py:264–265`.

ray 방향으로 원점을 0.15 이동한다. 주변 구 전체를 무시하는 기능이 아니다. Conference source transform scale=1이므로 이 장면에서는 15cm 해석이 가능하지만 다른 정규화 장면에 그대로 적용할 수 없다.

주변 self-hit을 줄이는 대신 실제 가까운 차폐체/접촉 그림자를 건너뛸 수 있다. emitter가 offset보다 가까우면 유효 segment 정의도 필요하다. 작은 adaptive bias, originating surface 정보, Gaussian 크기/두께와 연동한 exclusion을 따로 설계할 필요가 있다. offset만 계속 키워서 물리적으로 정확해졌다고 판단하면 안 된다.

### 4. 색 공간과 radiance scale 불일치 — emitter 초기화 및 albedo 초기화 옵션 수정

위치: `aegir/gaussian_scene.py:142–155`, `aegir/irgs_tracer.py:271–287,335–338`, `aegir/trainer.py:255–290`.

- PNG appearance로 학습된 DN SH를 보존하고 IRGS는 evalSH+0.5를 합성한다. inverse transfer/tone mapping/exposure 보정이 없다.
- Conference AEGIR target은 checkpoint observation policy상 scene-linear HDR다. LDR 계열 SH를 그대로 radiance로 섞는다.
- SH-only에서는 간접광의 주된 크기가 emitter intensity로 결정되지 않는다. 다만 emitter endpoint/direction/visibility를 통한 일부 의존성은 있으므로 'emitter gradient가 전부 0'이라는 설명은 부정확하다.
- 초기 albedo도 SH DC appearance를 직접 넣는다. DR prior는 linear reflectance로 decode되지만 초기값은 관측 조명/그림자와 camera encoding을 포함한다.
- 초기 emitter 후보 점수뿐 아니라 cluster 색/세기도 `radiance_hdr`의 robust channel median에서 가져온다. 기본 angular kernel이 `exp(-1)`이므로 emitter amplitude는 관측 radiance의 `e`배로 초기화해 초기 출력 radiance를 관측값과 맞춘다. HDR가 전혀 없는 legacy/LDR 장면만 기존 상대값 초기화를 유지한다.
- `--albedo-initialization neutral --neutral-albedo 0.5`를 추가했다. 이 모드는 모든 Gaussian의 scene-linear RGB albedo를 `(0.5,0.5,0.5)`로 시작해 source SH의 그림자/하이라이트를 초기 물성에 직접 복사하지 않는다. 기존 실험 재현을 위해 기본값 `sh_dc`는 유지했다.

단순 sRGB decoding만으로 unknown tone mapping이나 clipping된 HDR가 복원되지 않는다. linear-HDR SH를 별도 fit하거나 calibrated radiance proxy를 구성하는 접근이 필요하다. 이 문제는 SH 없는 secondary-only의 모든 artifact를 설명하지 않는다.

### 5. 카메라에 직접 보이는 emitter emission 항 없음 — 수정 완료

위치: `aegir/render.py:59–106`, `aegir/inspect.py:48–94`.

수정 전에는 Gaussian G-buffer의 반사광만 shade하고 direct+indirect를 합쳐 조명 패널의 관측 self-emission을 직접 표현하지 못했다. 현재는 모든 pixel의 camera ray를 learned emitter ellipsoid와 교차하고, nearest positive emitter hit가 expected Gaussian surface보다 앞이면 그 pixel을 emitter radiance로 합성한다. Gaussian surface가 앞이면 emitter는 가려지며, Gaussian miss pixel은 surface distance를 infinity로 두어 배경 앞 emitter도 표시한다. 출력은 `emission_hdr`와 `emitter_mask`로도 분리된다.

Conference sampled frames 0,8,64,101,136에서 상위 0.5% HDR 픽셀은 dataset mask상 모두 valid였다. frame64의 RGB maximum은 34.3464다. 모든 픽셀이 실제 emitter라고 분류한 것은 아니며, 최종 raster validity도 별도다. 다만 dataset 단계에서 고휘도 emission을 일괄 제외하는 protocol은 아니다.

이 구현은 emission과 뒤 surface shading을 더하지 않고 depth winner를 선택하며, 최종 관계는 `hdr = masked_direct + masked_indirect + emission`이다. 따라서 관측 발광 pixel이 BRDF에 주던 보상 압력은 줄어든다. 다만 learned ellipsoid가 실제 fixture보다 크거나 위치가 틀리면 self-emission silhouette도 같이 틀리므로 `emitter_mask`를 반드시 진단해야 한다.

### 6. Secondary NEE가 비정규화 surrogate로 나눔

위치: `aegir/emitters.py:238–282`, `aegir/render.py:296–333`.

Primary surrogate는 논문을 따른다. 그러나 현재 추가한 secondary NEE에서도 실제 sampling PDF 대신 이 surrogate로 나눈다. 따라서 정확한 면광원 적분의 unbiased estimator라고 주장할 수 없다. emitter 모양/방향이 바뀌는 relighting, 외부 Blender/Mitsuba로의 조명 전달에서 mismatch가 드러날 수 있다.

sample 수 증가로 variance를 줄일 수 있어도 surrogate bias는 사라지지 않는다. 정확한 변경에는 ellipsoid Jacobian, projected cosine, emitter 선택, 앞/뒤 교차 및 방향별 mixture, emission/occlusion 정의를 함께 정합시켜야 한다.

### 7. 낮은 roughness에서 GGX D를 epsilon이 크게 왜곡 — 수치 재현

위치: `aegir/brdf.py:50–53`.

`a2 / (pi * denom² + 1e-6)`의 epsilon이 날카로운 specular peak의 실제 분모보다 커진다. float64 입력으로 실제 함수와 NdotH=1에서의 analytic GGX `D=1/(pi*r^4)`를 비교했다.

| roughness | 현재 D | analytic D |
|---|---:|---:|
| .02 | .1600 | 1,989,436.79 |
| .05 | 6.2492 | 50,929.58 |
| .10 | 96.9541 | 3,183.10 |
| .20 | 176.9427 | 198.9437 |
| .60 | 2.45605 | 2.45609 |

낮은 r 구간에서 peak의 roughness derivative 부호도 analytic과 반대가 된다. 수치 안정화가 BRDF 형태를 실질적으로 변경한다.

현재 Conference의 Gaussian roughness 최소값은 secondary-only .45686, SH-only .46432이고 둘 다 r<.15가 0개다. 그러므로 현재 바닥 그림자의 주원인으로 지목하지 않는다. glossy 재질 테스트/편집과 이후 학습에는 명확한 결함이다. 단순 epsilon 삭제보다 안정적인 D 식과 roughness 범위에서의 에너지/gradient 검증이 필요하다.

### 8. SH-only / secondary-only 모드가 checkpoint config에 저장되지 않음

위치: `aegir/trainer.py:55–63,736–787`, `aegir/config.py`.

두 checkpoint의 config는 mode 관점에서 같지만 `losses[].sh_proxy_weight`는 각각 전 구간 1/0이다. 저장된 config를 현재 기본 함수에 넣으면 step 0/5K/10K/15K/19999에서 1/1/.5/0/0이 나온다.

즉 모드 override를 재적용하지 않고 일반 CLI exact resume을 쓰면 의도한 schedule을 재현하지 못한다. 최종 inference 로딩 자체에는 SH를 끄므로 같은 방식으로 영향을 주지는 않는다. `indirect_training_mode` 및 version을 명시적으로 저장하고 resume에서 검증해야 한다.

추가로 `inspect.py:275`의 `paper_inference_bounce_ready=True`와 'first hit' 문자열은 현재 NEE/expected-hit 경로의 논문 일치를 과장한다.

### 9. 조명 초기화의 관측 사각지대

위치: `aegir/initialization.py:196–223`.

Persistent threshold는 실제로 해당 위치를 볼 수 있는 view 수가 아니라 전체 view 수의 fraction에 기반한다. 가려진 작은 조명/한 방향에서만 보이는 스탠드는 실제 emitter여도 탈락할 수 있다. 바닥/벽 반사광이 여러 view에서 일관되게 밝으면 persistent만으로 self-emission과 구분되지 않는다.

후보가 없으면 중앙 single emitter fallback을 사용하며, 이후 controller는 기존 emitter를 split/prune할 뿐 missed fixture 위치에 새 후보를 생성하는 단계가 없다. DBSCAN/PCA 개선만으로 모든 광원을 찾는다고 주장할 수 없다.

### 10. 방향성 emitter는 앞뒤 대칭 — 6축 방향 이득으로 수정 완료

위치: `aegir/emitters.py`, `aegir/render.py`, `aegir/adaptive.py`, `aegir/checkpoint.py`.

원 논문 Eq.1은 방향 내적이 모두 제곱되므로 `Le(l)=Le(-l)`이다. 따라서 가림 구조가 없으면 아래로만 비추는 lamp를 emission profile 자체로 표현하지 못한다. 이는 원 논문 모델의 표현 한계이며 실제 housing geometry가 반대쪽 빛을 가리는 현상과 구분한다.

현재 확장 모델은 emitter마다 다음을 학습한다.

- 공통 scene-linear RGB `E`: 전체 광량과 색.
- local `(+X,-X,+Y,-Y,+Z,-Z)` 순서의 양수 scalar gain `g_s`: 방사 방향 분포.
- 기존 ellipsoid pose/scale, 3축 spread, falloff power.

local outgoing direction `u=R^T omega_out`에 대해 signed alignment
`a=(u_x,-u_x,u_y,-u_y,u_z,-u_z)`를 만들고,

`w_s = softmax(k a)_s`, `g(u)=sum_s w_s g_s`,

`Le_new(omega_out)=Le_Eq1(omega_out) g(u)`

로 계산한다. 기본 sharpness는 `k=12`다. 여섯 gain은 항상 산술평균 1로 정규화하므로 공통 RGB와 direction gain 사이의 전역 scale ambiguity를 제거했다. 모두 1이면 `g(u)=1`이므로 기존 대칭 방사식과 정확히 같다. `six_axis_emission=False` 또는 CLI `--symmetric-emission`으로도 원 논문 대칭 모델을 명시적으로 재현할 수 있다.

방향 정의는 모든 호출에서 **emitter에서 receiver로 나가는 방향**으로 통일했다. surface-to-emitter sample direction을 사용하는 direct/secondary NEE에서는 부호를 뒤집고, 카메라 ray에서는 `-ray_direction`을 사용한다. 기존 식은 대칭이라 이 부호 차이가 출력에 드러나지 않았지만 6축 모델에서는 필수다.

Emitter split 시 두 child가 부모의 6축 분포를 복사하고, prune/split 후 Adam state migration에도 새 parameter가 포함된다. inspection JSON은 공통 RGB와 축 label별 gain을 저장한다. v1-v8 checkpoint에는 gain이 없으므로 모두 1로 로드해 기존 렌더를 보존하며, 새 optimizer parameter까지 정확히 resume하는 checkpoint format은 v9로 올렸다.

## 2026-09-06 구현 변경 요약

| 문제 | 적용한 해결 | 현재 상태/의도적으로 남긴 점 |
|---|---|---|
| 에미터 뒤 Gaussian이 visibility/hit에 혼입 | surface-light ray마다 endpoint를 native OptiX tracer에 전달해 alpha, depth, material, SH trace 자체를 유한 구간에서 종료 | 수정 완료; soft endpoint gate는 경계 완화용 유지 |
| self-hit 검은 반점 | 실험에서 `trace_origin_offset=0.15` 사용 | 코드는 유지. originating Gaussian exclusion의 근본 해결은 아직 아님 |
| persistent 후보 이전에 비지속 HDR 점까지 DBSCAN 입력 | 여러 view에서 유지되는 HDR candidate만 DBSCAN/PCA emitter seed에 사용 | 수정 완료; visibility-conditioned persistence와 missed-emitter 재탐색은 남음 |
| emitter 색을 HDR 점수와 다른 LDR 값으로 초기화 | HDR cluster의 robust channel median을 scene-linear radiance로 사용하고 Eq.1 초기 attenuation을 보상 | 수정 완료; HDR가 없는 데이터는 legacy fallback |
| SH DC가 그림자 포함 appearance를 albedo seed로 사용 | 중성 `(0.5,0.5,0.5)` albedo 초기화 옵션과 비교 스크립트 추가 | 옵션 구현 완료; 기본은 과거 재현용 `sh_dc`, SH proxy의 LDR/HDR 문제는 남음 |
| 카메라에 직접 보이는 광원이 반사면으로만 렌더됨 | camera ray와 emitter ellipsoid/Gaussian expected surface의 depth winner를 구해 self-emission 합성 | 수정 완료; `emitter_mask`와 `emission_hdr` 출력 추가 |
| Eq.1 방향성이 앞뒤 대칭 | 공통 RGB + mean-one local 6축 scalar gain, outgoing 방향 통일 | 수정 완료; 기존 대칭 모델 ablation 가능 |

이번 6축 변경은 방사 함수만 확장하며 emitter surface sampler와 PDF surrogate는 바꾸지 않는다. 따라서 6번의 secondary NEE PDF bias를 해결한 것으로 해석하면 안 된다.

## 평가 및 성능

- 두 최신 Conference checkpoint는 191개 전 view를 학습했다. 기존 inspect는 `split=all`, 첫 30장 평가, `held_out_from_train=False`다. 해당 PSNR은 training-view reconstruction이며 held-out NVS가 아니다.
- interpolated/elevated novel path는 유용한 정성 결과지만 대응 GT 없이 정확도 지표로 사용할 수 없다. 새로운 조명 GT, novel camera GT, source geometry 학습 단계부터 공통 split이 필요하다.
- 현재 inspect는 denoiser none. 논문 시각화는 denoise를 사용하므로 이미지의 MC 노이즈를 그대로 비교하면 공정하지 않다. raw metric은 별도 유지해야 한다.
- raster 256에서도 shading은 step당 16x16 patch=256 primary points다. 풀 해상도 256x256 inspect는 65,536 points다. primary ray workload가 256배이고 secondary까지 붙는다. inspect가 오래 걸리는 것은 이 구조로 상당 부분 설명된다.
- 같은 patch를 사용할 때 경계 pixel은 중앙보다 선택될 확률이 낮다. 고해상도에서 pixel coverage가 느려진다. 이는 PDF 오류와 다른 학습 sampling 문제다.
- geometry LR=0이어도 source code는 매 step BVH refresh 및 geometry gradient/optimizer 경로를 유지한다. frozen geometry 전용 cache/gradient 절약의 여지가 있다. 실행 속도 개선 가능성이며 현재 렌더 오류 원인은 아니다.

## 권장 처리 순서

1. 실험 indirect mode/config와 평가 split을 checkpoint에 명시해 비교가 재현되도록 한다.
2. expected secondary hit 정의와 origin-surface exclusion을 개선한다.
3. SH proxy를 linear HDR radiance 단위로 보정하거나 calibrated proxy로 교체한다.
4. 현재 NEE를 유지할 경우 sampler/PDF/emission 정의를 함께 맞춘다.
5. GGX의 낮은 roughness 수치 문제를 수정하고 분석적 BRDF 테스트를 둔다.
6. persistent 후보의 visibility-conditioned 기준, missing-emitter 진단을 추가한다.
7. 6축 emitter와 원 논문 대칭 emitter를 held-out relighting GT로 비교한다.

이번 수정 묶음에는 segment visibility, persistent-only clustering, HDR emitter 색 초기화, 중성 albedo 옵션, 카메라-visible emission, 6축 방향성 emitter가 포함된다. origin offset 0.15와 expected-hit 집약은 의도적으로 그대로 두었고, 중성 albedo는 옵션이며 과거 기본값도 보존했다. 바닥 albedo shadow의 원인을 하나로 확정하지 않았으며, 현재 evidence는 여러 model mismatch와 학습 초기 appearance bias가 공존함을 보여준다.
