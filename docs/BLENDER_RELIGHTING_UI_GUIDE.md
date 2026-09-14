# Blender relighting emitter setup guide

이 문서는 데이터 제작 컴퓨터에서 Blender 화면을 보며 새 relighting 조건을
만들기 위한 절차다. 첫 oracle relighting 실험은 기존 학습 장면의 geometry와
카메라를 그대로 유지하고, 기존 여섯 패널의 발광을 끈 뒤 새로운 타원형
emissive mesh 하나만 켜는 것을 기준으로 한다.

현재 AEGIR emitter의 공간 support는 무한한 Gaussian이 아니라 **경계가 있는
평면 ellipse**다. ellipse 내부를 균일하게 면적 샘플링한다. 따라서 첫 실험은
Blender `AREA` Light보다 타원형 emissive mesh를 사용해야 현재 데이터 캡처,
emitter mask, depth, bbox 및 scene-linear HDR 파이프라인과 바로 연결된다.

## 작업 파일

원본을 직접 수정하지 말고 다음 relighting 장면을 열어 새 이름으로 저장한다.

```text
/workspace/hohyun_ws/datasets/raw/blender/my_room/my_room_relight_6panel.blend
```

함께 복사해서 수정할 lighting 설정은 다음 파일이다.

```text
/workspace/hohyun_ws/datasets/raw/blender/my_room/my_room_relight_6panel_lighting.json
```

권장 새 파일명:

```text
my_room_relight_new_ellipse.blend
my_room_relight_new_ellipse_lighting.json
```

현재 서버 캡처 기준 Blender 버전은 5.2.0 LTS이며 장면 단위는
`1 Blender unit = 1 meter`다.

## 1. 원본 장면 보호

1. Blender에서 `my_room_relight_6panel.blend`를 연다.
2. `File > Save As`를 선택한다.
3. `my_room_relight_new_ellipse.blend`로 저장한다.
4. `AEGIR_TrainCameras`, `AEGIR_ValCameras`, `AEGIR_TestCameras` 컬렉션과
   카메라 transform을 수정하지 않는다.
5. Render Resolution, pixel aspect, camera focal length, lens shift, exposure와
   Color Management를 수정하지 않는다. 서버 캡처 스크립트가 최종 출력
   해상도와 Cycles 설정을 지정하지만 카메라 자체 설정은 장면에서 읽는다.

## 2. 기존 조명 확인

Outliner에서 `AEGIR_Emitters` 컬렉션을 펼치면 다음 여섯 개가 있어야 한다.

```text
CeilingPanel_0
CeilingPanel_1
CeilingPanel_2
CeilingPanel_3
CeilingPanel_4
CeilingPanel_5
```

기존 패널은 삭제하거나 숨기지 않는다. 패널 geometry는 그대로 두고 발광만
끈다. Blender Material Properties에서 각 패널의 Principled BSDF
`Emission Strength`를 0으로 설정하면 Rendered View에서 새 조건을 미리 볼
수 있다. 최종 서버 렌더에서는 lighting JSON이 이 값을 다시 설정하므로 JSON의
새 condition에서도 여섯 패널을 모두 `enabled: false`로 지정해야 한다.

Outliner에서 다른 `LIGHT` 타입 object를 검색한다. 학습 데이터와 동일하게
모든 비등록 Light의 Render 아이콘을 끄거나 Energy를 0으로 둔다. 또한
`World Properties > Surface > Background > Strength`가 0인지 확인한다.
남은 World/HDRI 또는 Light는 `emitters.json`에 기록되지 않는 조명이 되어
공정한 relighting 평가를 깨뜨린다.

## 3. 타원형 emissive mesh 만들기

1. 3D Viewport에서 `Shift+A > Mesh > Circle`을 선택한다.
2. 좌측 아래 Add Circle 패널 또는 `F9`에서 다음처럼 설정한다.
   - Vertices: `64`
   - Radius: `1 m`
   - Fill Type: `N-Gon`
3. object 이름을 `RelightEllipse_0`으로 바꾼다.
4. `N` 패널의 Item 탭에서 Location과 Rotation을 입력한다.
5. X/Y Scale로 원하는 타원 크기를 만든다. 예를 들어 전체 크기
   `0.60 m x 0.30 m`는 반지름 `0.30 m x 0.15 m`에 해당한다.
6. `Ctrl+A > Scale`을 실행한다. 적용 후 Dimensions가 의도한 실제 미터
   크기인지 확인한다.
7. local X/Y가 타원 평면이고 local Z가 평면 normal이 되도록 회전한다.
8. `M > AEGIR_Emitters`로 object를 emitter 컬렉션으로 이동한다.

직사각형 mesh를 사용하면 AEGIR ellipse가 bbox에 내접하면서 사각형 모서리를
표현하지 못한다. 첫 비교에서는 Circle을 X/Y로 늘린 타원 mesh를 사용한다.

## 4. emitter ID 지정

1. `RelightEllipse_0`을 선택한다.
2. `Object Properties > Custom Properties > New`를 누른다.
3. property 이름을 `aegir_emitter_id`로 바꾼다.
4. 정수 값 `7`을 입력한다.

기존 패널은 ID 1부터 6을 사용하므로 새 emitter는 7을 사용한다. 모든 ID는
양의 정수이며 중복되면 서버 캡처가 즉시 실패한다. Object Pass Index는
캡처 스크립트가 자동으로 배정하므로 Blender 화면에서 직접 맞추지 않는다.

## 5. emissive material 만들기

1. `Material Properties > New`로 새 material을 만든다.
2. 이름을 `AEGIR_RelightEllipse_0_Material`로 지정한다.
3. `Use Nodes`가 켜져 있고 `Principled BSDF`가 하나 이상 있는지 확인한다.
4. Principled BSDF의 값을 설정한다.
   - Base Color: 검정 또는 조명 기구의 비발광 표면색
   - Emission Color: 원하는 조명색
   - Emission Strength: 원하는 세기
5. 첫 실험은 기존 패널과 변수를 줄이기 위해 다음 값을 권장한다.
   - Emission Color: `(1.0, 0.78, 0.52)`
   - Emission Strength: `50.0`

Blender color picker의 표시색과 저장되는 scene-linear RGB를 혼동하지 않는다.
최종 수치는 lighting JSON의 `rgb` 배열이 authoritative하다. 서버 렌더
스크립트는 JSON의 RGB와 Strength를 Principled BSDF에 적용한 뒤 HDR을
저장한다.

## 6. 위치를 화면에서 정하기

처음에는 조명 기구가 test camera에 직접 보이지 않거나 화면 가장자리에만
보이는 위치가 좋다. 그러면 새 조명 geometry 자체의 오차보다 Gaussian map의
조명 반응을 중심으로 평가할 수 있다.

다음 항목을 기록한다.

| 항목 | 기록값 |
| --- | --- |
| Object name | `RelightEllipse_0` |
| Location XYZ (m) |  |
| Rotation XYZ |  |
| Dimensions XY (m) |  |
| Emission RGB | `(1.0, 0.78, 0.52)` |
| Emission Strength | `50.0` |
| `aegir_emitter_id` | `7` |

평면과 거의 같은 높이의 벽을 강하게 비추려는 실험에서는 타원을 벽 쪽으로
기울이거나 벽보다 위/아래에 둔다. 두께가 0인 평면 emitter는 평면과 정확히
평행한 방향에서 투영 면적이 0에 가까워지므로, 같은 높이의 옆 방향 직접광이
본질적으로 약하다.

## 7. lighting JSON 추가

`my_room_relight_6panel_lighting.json`을
`my_room_relight_new_ellipse_lighting.json`으로 복사한다. top-level `scene`을
새 blend 파일명으로 바꾸고 `emitters` 배열에 새 object를 추가한다.

```json
{
  "name": "RelightEllipse_0",
  "aegir_emitter_id": 7,
  "material": "AEGIR_RelightEllipse_0_Material",
  "default_rgb": [1.0, 0.78, 0.52],
  "default_strength": 50.0
}
```

`conditions`에 `new_ellipse_only`를 추가한다. `AEGIR_Emitters` 컬렉션에 있는
모든 object 이름이 이 condition에 반드시 들어 있어야 한다.

```json
"new_ellipse_only": {
  "CeilingPanel_0": {"enabled": false, "rgb": [1.0, 0.78, 0.52], "strength": 50.0},
  "CeilingPanel_1": {"enabled": false, "rgb": [1.0, 0.78, 0.52], "strength": 50.0},
  "CeilingPanel_2": {"enabled": false, "rgb": [1.0, 0.78, 0.52], "strength": 50.0},
  "CeilingPanel_3": {"enabled": false, "rgb": [1.0, 0.78, 0.52], "strength": 50.0},
  "CeilingPanel_4": {"enabled": false, "rgb": [1.0, 0.78, 0.52], "strength": 50.0},
  "CeilingPanel_5": {"enabled": false, "rgb": [1.0, 0.78, 0.52], "strength": 50.0},
  "RelightEllipse_0": {"enabled": true, "rgb": [1.0, 0.78, 0.52], "strength": 50.0}
}
```

`enabled: false`이면 실제 적용 Strength는 0이다. 위 항목에 남아 있는
`strength: 50.0`은 다시 켤 때 사용할 설정값일 뿐 비활성 조명에 적용되지
않는다.

JSON 문법상 바로 앞 condition 뒤에는 쉼표가 필요하고 마지막 condition
뒤에는 쉼표를 넣지 않는다. object 이름은 Blender Outliner 이름과 대소문자까지
정확히 일치해야 한다.

## 8. Blender 화면에서 최종 확인

1. Viewport Shading을 Rendered로 바꾼다.
2. 기존 여섯 패널의 emission이 꺼져 있는지 확인한다.
3. 새 타원만 발광하는지 확인한다.
4. `AEGIR_TestCameras` 중 가까운 카메라 여러 개를 활성 camera로 바꿔가며
   새 emitter가 의도하지 않게 화면을 가리지 않는지 확인한다.
5. 타원 mesh의 normal, 위치, Dimensions를 다시 확인한다.
6. `File > External Data > Pack Resources`를 실행한다.
7. blend 파일을 저장한다.

UI preview는 배치 렌더 검증을 대신하지 않는다. lighting JSON이 최종 material
값을 덮어쓰므로 반드시 다음 2프레임 preflight를 수행한다.

## 9. 서버에서 2프레임 preflight

새 blend와 JSON을 서버의 같은 디렉터리에 복사한 뒤 빈 출력 경로로 실행한다.
서버 wrapper는 저장소에 검증된 OptiX 580.173.02 라이브러리가 있으면 이를
자동으로 사용하고, 다른 컴퓨터에서는 시스템 NVIDIA 드라이버로 fallback한다.

```bash
cd /workspace/hohyun_ws/AEGIR

SPLITS=test MAX_FRAMES=2 \
bash scripts/blender_aegir/run_server_render.sh \
  /workspace/hohyun_ws/datasets/raw/blender/my_room/my_room_relight_new_ellipse.blend \
  /workspace/hohyun_ws/datasets/raw/blender/my_room/my_room_relight_new_ellipse_lighting.json \
  /workspace/hohyun_ws/datasets/aegir_nerfstudio/blender_my_room_relight_new_ellipse_preflight \
  new_ellipse_only 32 256
```

확인할 파일:

```text
images/frame_*.png
radiance_hdr/frame_*.npy
depths/frame_*.npy
emitter_masks/frame_*.png
emitters.json
transforms.json
```

`emitters.json`에서 `RelightEllipse_0`의 다음 값이 있어야 한다.

- `type: "MESH"`
- `emitter_id: 7`
- 비어 있지 않은 `world_bbox_corners`
- `condition.enabled: true`
- 의도한 color와 strength

기존 여섯 패널은 `condition.enabled: false`, 새 emitter는 true여야 한다.

## 10. held-out test 40장 렌더

preflight가 맞으면 기존과 다른 빈 출력 경로에 40개 test camera를 렌더한다.

```bash
cd /workspace/hohyun_ws/AEGIR

SPLITS=test \
bash scripts/blender_aegir/run_server_render.sh \
  /workspace/hohyun_ws/datasets/raw/blender/my_room/my_room_relight_new_ellipse.blend \
  /workspace/hohyun_ws/datasets/raw/blender/my_room/my_room_relight_new_ellipse_lighting.json \
  /workspace/hohyun_ws/datasets/aegir_nerfstudio/blender_my_room_relight_new_ellipse_test \
  new_ellipse_only 256 512
```

test-only 캡처는 train frame이 없으므로 다음처럼 검증한다.

```bash
/workspace/.conda/envs/dnsplatter/bin/python \
  scripts/blender_aegir/validate_dataset.py \
  --data /workspace/hohyun_ws/datasets/aegir_nerfstudio/blender_my_room_relight_new_ellipse_test \
  --min-train-frames 0 \
  --min-test-frames 40
```

새 emitter를 의도적으로 모든 카메라 밖에 둔 경우 emitter-visible 검증만 0으로
낮춘다.

```bash
--min-emitter-visible-frames 0
```

## 11. AEGIR 변환 시 사용할 값

캡처가 생성한 `emitters.json`의 `world_bbox_corners`를 2DGS checkpoint의
dataparser transform과 scale로 model 좌표에 옮긴다. 현재 AEGIR 데이터 로더는
다음 변환을 이미 구현한다.

```text
model_point = scale * (rotation * blender_world_point + translation)
```

변환된 bbox에서 중심, 평면 PCA 축, 두 반지름과 `wxyz` quaternion을 구한다.
양면 균일 emissive mesh의 초기 방향 파라미터는 다음을 사용한다.

```text
axis_gains     = [1, 1, 1, 1, 1, 1]
spreads        = [32, 32, 32]
falloff_power  = [1]
```

초기 radiance는 scene-linear `rgb * strength`에 모든 조건에서 공통인 하나의
보정 상수를 곱한다. test condition마다 GT 이미지를 보고 별도 보정하면 oracle
relighting 수치와 이미지 기반 light fitting 수치가 섞이므로 금지한다.

캡처가 끝난 뒤 다음 명령은 enabled mesh만 골라 AEGIR model 좌표로 변환하고,
JSON과 바로 로드할 수 있는 tensor state를 데이터셋 폴더에 저장한다. radiance
보정은 새 test 이미지가 아니라 기존 학습 checkpoint의 초기 emitter radiance를
사용한다.

```bash
/workspace/.conda/envs/dnsplatter/bin/python \
  scripts/convert_blender_emitters_to_aegir.py \
  --dataset /workspace/hohyun_ws/datasets/aegir_nerfstudio/blender_my_room_relight_new_ellipse_test \
  --dnsplatter-checkpoint /path/to/2dgs/nerfstudio_models/step-000020000.pt \
  --aegir-checkpoint /path/to/aegir_20k.pt
```

생성 파일은 `aegir_emitter_override.json`과 `aegir_emitter_override.pt`다.
JSON에는 중심, 두 반지름, `wxyz` quaternion, local 축, radiance, six-axis gain,
spread와 falloff가 기록된다.

현재 저장소에는 기존 checkpoint의 emitter 일부를 선택하는 기능은 있지만,
새 `emitters.json`을 runtime emitter로 교체하는 CLI는 아직 없다. 다음 구현은
`inspect --emitter-json ... --replace-checkpoint-emitters` 형태로 추가하며,
Gaussian geometry와 material은 고정한 채 새 emitter만 렌더에 사용해야 한다.

## 최종 체크리스트

- [ ] 원본 blend를 수정하지 않고 새 이름으로 저장했다.
- [ ] 카메라와 scene geometry를 수정하지 않았다.
- [ ] 새 object는 타원형 mesh이고 local Z가 normal이다.
- [ ] 새 object가 `AEGIR_Emitters` 컬렉션에 있다.
- [ ] `aegir_emitter_id=7`이며 다른 ID와 겹치지 않는다.
- [ ] 새 material에 Principled BSDF가 있다.
- [ ] 기존 여섯 패널과 모든 기타 Light, World 조명을 껐다.
- [ ] lighting JSON에 일곱 emitter가 전부 들어 있다.
- [ ] `new_ellipse_only`에서 새 emitter 하나만 enabled다.
- [ ] 외부 texture를 Pack Resources한 뒤 저장했다.
- [ ] 2프레임 preflight의 RGB, HDR, depth, mask와 `emitters.json`을 확인했다.
- [ ] 같은 40개 held-out test camera로 최종 GT를 렌더했다.
