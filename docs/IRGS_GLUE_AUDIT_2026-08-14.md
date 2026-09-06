# IRGS glue 정합성 감사 (2026-08-14)

## 결론

AEGIR의 `IRGSVisibilityTracer`는 pinned 공식 IRGS surfel tracer의 핵심
geometry, SH, opacity-compositing 및 autograd 규약과 정합한다. 이번 감사에서
ray/scene tensor를 native binding에 넘기는 순서나 quaternion/scale/SH layout의
blocking mismatch는 발견되지 않았다.

공식 binding은 dtype, device, shape와 compile-time 배열 한계를 검사하지 않고
tensor storage를 `float*`로 해석한다. 잘못된 tensor가 들어오면 Python 예외가 아니라
silent corruption 또는 CUDA failure가 날 수 있으므로 AEGIR glue에 fail-fast contract
검사를 추가했다.

## 감사 기준 source

- IRGS repository: `/workspace/hohyun_ws/external/IRGS`
- revision: `1461869c0da84f26aebd8bb6d1caaa8a7bd91326`
- tracked source state: clean
- installed Python wrapper:
  `/workspace/.conda/envs/dnsplatter/lib/python3.10/site-packages/surfel_tracer/raytracer.py`
- source/installed wrapper SHA-256 (동일):
  `8f066f9df45b48f3ac4b4a468e9c5d6615d684a5f71708790a1311bbe7a5184b`
- installed native extension SHA-256:
  `7b368fa181b4363d17665908498118bb88e102198febe6ae3a891033a7739df0`

대조한 공식 파일:

- `submodules/surfel_tracer/surfel_tracer/raytracer.py`
- `submodules/surfel_tracer/src/optix/gaussiantrace_forward.cu`
- `submodules/surfel_tracer/src/optix/gaussiantrace_backward.cu`
- `submodules/surfel_tracer/src/optix/gaussiantrace_intersection_test.cu`
- `submodules/surfel_tracer/src/optix/auxiliary.h`
- `scene/gaussian_model.py`
- `gaussian_renderer/__init__.py`

## 항목별 대조

| 항목 | 공식 IRGS 규약 | AEGIR glue | 판정 |
|---|---|---|---|
| dtype/device | CUDA float32 pointer | CUDA float32 강제 검사 | 정합 |
| mean | `[G,3]` world coordinate | `[G,3]` | 정합 |
| quaternion | `wxyz`, normalized | `wxyz`, normalized | 정합 |
| tangent scale | two planar scales | DN의 최소 scale axis를 z로 옮긴 후 `[G,2]` | 정합 |
| `ru`,`rv` | rotation column 0/1 divided by scale | 동일 | 정합 |
| normal | rotation/covariance z axis, ray-facing flip | rotation column 2; native kernel이 ray-facing flip | 정합 |
| opacity | sigmoid opacity, `[G,1]` storage | 동일 | 정합 |
| alpha cutoff | `1/255` | `1/255` | 정합 |
| transmittance stop | IRGS repo default `0.03` | AEGIR paper 값 `0.05` | 의도적 차이 |
| SH layout | `[G,K,3]`, degree 0--3 | DN DC+rest `[G,16,3]`, degree 3 | 정합 |
| SH direction | traced `ray_d` | 동일 | 정합 |
| material features | native 최대 12 channels | albedo 3 + roughness 1 + metallic 1 | 정합 |
| back culling | default false | false | 정합 |
| BVH proxy | opacity support ellipsoid, icosahedron x 1.2584 | 동일 vertex/face topology과 radius 식 | 정합 |
| BVH update | geometry update마다 rebuild/update | 매 iteration geometry step 전에 refresh | 정합 |
| backward | rays, mean, opacity, `ru`,`rv`, normal, features, SH | rays와 Gaussian geometry까지 연결 | 정합 |

실제 conference DN checkpoint의 appearance는 `sh_degree=3`이고
`features_dc`는 RGB logit이 아니라 `RGB2SH` 계수다. checkpoint shape은
DC `[197483,3]`, rest `[197483,15,3]`이며 AEGIR에서 `[197483,16,3]`으로
결합하는 것이 맞다.

## 공식 raw output을 AEGIR에서 해석하는 방법

공식 kernel은 각 ray에서 다음 opacity-composited 합을 반환한다.

- `color = sum(T_i * alpha_i * SH_i)`
- `normal = sum(T_i * alpha_i * n_i)`
- `feature = sum(T_i * alpha_i * f_i)`
- `depth = sum(T_i * alpha_i * d_i)`
- `alpha = sum(T_i * alpha_i)`

AEGIR glue는 이를 다음처럼 사용한다.

- training bounce color는 composite 그대로 유지한다. 여기에 before-light gate를
  곱하면 Equation 4의 `(1-V) * L_bounce`가 되므로 alpha를 다시 곱하지 않는다.
- hit depth, normal, material은 `value / alpha`로 conditional hit attribute를 만든다.
- numerical ray-origin shift 이후의 depth에 shift를 다시 더해 원래 shading point
  기준 거리로 변환한다.
- Equation 3 visibility bias는 이 변환 후 정확히 한 번만 뺀다.

이는 공식 IRGS wrapper가 일부 고-opacity ray에서만 normalization하는 동작과
의도적으로 다르다. AEGIR inference에는 secondary hit point/material이 필요하므로
conditional attribute가 필요하고, training SH color에는 opacity composite가 필요하다.

## numerical ray origin과 visibility bias

공식 IRGS renderer는 surface point에서 `light_t_min=0.05`만큼 ray origin을 옮긴다.
AEGIR glue도 meter 단위로 변환된 dataset에서 같은 규약을 사용한다.

- self-hit 회피용 origin offset: `0.05 m`
- visibility bias: `scene_scale * 1e-3`
- steepness: `80 / scene_scale`

origin offset은 순수 ray-start shift이고 endpoint shadow bias로 재사용하지 않는다. BVH의
얇은 2D Gaussian proxy 두께도 별도의 `scene_scale * 1e-6` 값으로 유지한다. 단일 surfel
중심에서 바깥쪽으로 시작하는 ray가 자기 자신을 occluder로 검출하지 않는 회귀 테스트를
추가했다. light보다 뒤에 있는 surfel도 visibility를 낮추지 않는 것을 함께 검증했다.

## native compile-time 한계

pinned IRGS extension에는 다음 값이 compile-time으로 고정되어 있다.

- `MAX_BUFFER_SIZE=16`
- `T_SCENE_MAX=100.0`
- `MAX_FEATURE_SIZE=12`

AEGIR는 feature channel 수와 scene scale을 native call 전에 검사한다. 현재 conference
step-5000 checkpoint의 bounds diagonal은 `19.8674`이므로 100-unit trace limit 안이다.
향후 scene 또는 emitter-to-surface 거리가 100 coordinate units에 접근하면 scene을
normalize하거나 IRGS PTX/native extension을 새 한계로 rebuild해야 한다.

## 추가된 회귀 검증

`tests/test_irgs_tracer.py`는 다음 8개 CUDA/OptiX 검증을 수행한다.

1. hit/miss visibility와 opacity gradient
2. inference mode 실행
3. coordinate scale 1/10에서 visibility와 opacity-gradient 동등성
4. full-SH training proxy와 inference SH disable
5. analytic single-surfel alpha, SH color, depth, normal, material 값
6. self-hit 회피와 behind-light geometry 무시
7. ray origin/direction 및 mean/scale/quaternion/opacity 전체 backward
8. float32가 아닌 scene을 native binding 전에 거부

검증 결과: `8 passed` (`1.76 s`). 테스트는 단일 synthetic surfel만 사용했고 실행 중인
conference 학습을 중단하거나 checkpoint를 변경하지 않았다.

## 남는 근사와 한계

- IRGS tracer의 depth/normal/material은 strict discrete first-hit가 아니라
  opacity-composited conditional expectation이다. AEGIR paper의 "first secondary hit"를
  native 수준에서 엄밀한 discrete hit로 바꾸려면 glue가 아니라 tracer output/API 변경이
  필요하다.
- BVH candidate selection은 미분되지 않지만, 선택된 Gaussian의 analytic intersection과
  compositing backward는 공식 tracer를 통해 미분된다. 이는 IRGS 자체의 경계다.
- native `.so`가 기록된 local source에서 reproducibly rebuild되었는지는 binary hash만으로
  증명할 수 없다. 다만 설치 wrapper는 pinned source와 byte-for-byte 동일하고, analytic
  forward/backward OptiX 검증이 통과했다.
- 현재 실행 중인 conference 학습 process는 시작 시 import한 수정 전 Python module을
  계속 사용한다. 이번 fail-fast 검사와 새 테스트는 다음 process부터 적용되며 transport
  수식이나 학습 결과를 바꾸는 수정은 아니다.
