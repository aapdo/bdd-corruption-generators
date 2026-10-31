# BDD Corruption Generators

Rain · Fog · LowLight · MotionBlur · Glare · Snow를 만드는 CPU 연구용 코드입니다.
처리된 RGB와 추정 깊이·흐름을 사용하며, 각 결과 옆 JSON에 실제 조건과 가정을 기록합니다.
Shadow는 포함하지 않습니다. API 키나 클러스터 스케줄러 없이 생성 커널을 사용할 수 있습니다.

2026-10-31 개선판은 동일한 BDD100K MOT 8개 영상의 대표5프레임에 7강도를 비교하는
정적 보정 단계입니다. 최종 약·중·강 선정, 연속 영상의 시간적 일관성, 학습·검출 성능은
아직 검증하지 않았습니다. 코드 공개와 학습 데이터 품질 통과를 구분합니다.

## 설치와 실행

Python3.10 이상의 별도 환경에서 다음 명령을 실행합니다.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python generate.py --help
```

입력은 BDD 1280×720 또는 640×360 RGB입니다. 출력은640×360 JPEG95, 4:4:4입니다.
Depth NPY는 RGB에 정렬된 양수 camera-z 깊이, flow NPZ는640×360×2 픽셀 흐름과
640×360 boolean `confidence`를 담습니다. 이 코드가 깊이 모델이나 flow를 자동 다운로드하지 않습니다.
`recipes/baseline_render.py:estimate_dis_pair`로 인접 RGB 쌍의 CPU DIS 흐름을 계산할 수 있습니다.
예제 내부 파라미터는640×360에서 fx=fy=450,cx=320,cy=180의 BDD 가정값입니다.
실제 카메라 보정이나 깊이의 미터 정확도를 검증한 값이 아닙니다.

```bash
python generate.py --type Fog --image frame.jpg --depth depth.npy --level .015 --output outputs/fog.jpg
python generate.py --type LowLight --image frame.jpg --level -1.5 --video clip01 --frame 10 --output outputs/dark.jpg
python generate.py --type MotionBlur --image frame.jpg --flow flow.npz --level 6 --output outputs/blur.jpg
python generate.py --type Glare --image frame.jpg --level .5 --controlled --output outputs/glare.jpg
python generate.py --type Snow --image frame.jpg --depth depth.npy --level .5 --controlled --output outputs/snow.jpg
```

출력 이미지나 JSON이 이미 있으면 덮어쓰지 않습니다. 정적 CLI의 Snow는 정지 카메라를
명시적으로 가정합니다. 실제 연속 입력에는 아래 상태 객체를 유지해야 합니다.
Glare의 `--bbox x0 y0 x1 y1`는 사용자가 확인한 발광등의 **원본 좌표**입니다.
RGB 광도 조건을 통과하지 못하거나 추가 후보 광원이 있으면 `--controlled`가 필요합니다.
밝거나 붉다는 이유만으로 실제 광원임을 검증했다고 표시하지 않습니다.

## 7단계 비교 설정

| 종류 | 강도 | 강도 외 조건 |
|---|---|---|
| Rain |5,15,25,50,75,100,200mm/h|노출3ms·입자 시드7301·고정 광학 설정|
| Fog |β=.00375,.005,.0075,.01,.015,.03,.06|같은 추정 깊이·영상별 고정 대기광|
| LowLight |−.5,−.75,−1,−1.5,−2,−3,−4EV|영상별 잡음 계수·프레임별 난수 공유|
| MotionBlur |p90=1,2,3,4,6,8,12px|같은 flow·20px 상한·적응형 적분|
| Glare |gain=.1,.25,.4,.5,.75,1,1.5|같은 광원 위치·색·PSF|
| Snow |활성 입자율10,25,40,50,65,80,100%|입자ID·운동·활성 임계값 공유|

이전 MotionBlur의2–30ms 목록을 실제 흐림 길이 목표로 바꿨습니다. px와ms를 같은 단위로
비교하지 않습니다. 극심한 Rain, β=.03/.06, −4EV, Glare1.5는 상한 진단 후보입니다.
`SEVERITY.json`은 최종 학습용3단계가 아닙니다.

## Rain

[astra-vision/rain-rendering](https://github.com/astra-vision/rain-rendering)의 공식 입자 시뮬레이터와
PBR 렌더러를 외부 런타임에서 사용합니다. renderer revision
`0a9d8ca739d71d4b09f61567c6f6cea94a79868e`, particle binary revision
`6a130aa7294dd980940416e8e0b21fef7112efaa`를 사용한 연결입니다.
다운로드 URL과 archive SHA는 [EXTERNAL_ASSETS.json](EXTERNAL_ASSETS.json)에 있습니다.
레거시 바이너리·런타임 라이브러리·별도 streak DB를 준비해야 하며 자동 설치하지 않습니다.

`rain_runtime/rain_backend.py`는 upstream의 `common` 패키지와 프로젝트의 `recipes/common.py`가
충돌하지 않도록 별도 디렉터리에 있습니다. 카메라 위치[1.5,1.5,.3],look-at[1.5,1.5,-1],
초점4mm·픽셀4.444μm·5FPS·gain20·f/6·노출3ms를 고정합니다.

```text
output/RAIN_INPUTS.json  # {"video":"clip01","samples":[{"image_id":1}],"speeds_kmh":[20.0]}
output/input/clip01/rgb/0000.png    # 1280x720
output/input/clip01/depth/0000.npy  # camera-z, 같은 크기
output/rainstreakdb/env_light_database/...  # 외부 DB
```

```bash
export BDD_RAIN_RUNTIME=/path/to/qualified_runtime
export PYTHONPATH="$BDD_RAIN_RUNTIME/upstream"
python rain_runtime/rain_backend.py simulate /path/to/output 25
# 원본 upstream을 보존하고 레이어 내보내기만 추가한 사본을 만듭니다.
python rain_tools.py export-patch --upstream "$BDD_RAIN_RUNTIME/upstream" --output /path/to/patched_upstream
# 아래 correct 결과의 경로/SHA를 output/simulations/25/SIMULATED.json에 지정합니다.
python rain_tools.py correct --input official_camera0.xml --motion motion.json --depths depths.npy --output corrected_camera0.xml
export PYTHONPATH=/path/to/patched_upstream
python rain_runtime/rain_backend.py render /path/to/output 25
python rain_tools.py compose --image frame.jpg --depth depth.npy --layers /path/to/0000.png.layers.npz --rate 25 --output rain.jpg
```

`motion.json`은 `{"motions":[{"twist":[vx,vy,vz,wx,wy,wz]}],"base_speeds_kmh":[20.0]}`
형식입니다. [v,w]는 이전 카메라 점을 현재 카메라로 변환하는 **left twist/초**입니다.
선택적 `depths.npy`는 XML 노출 순서의 N×H×W 추정 깊이입니다.
`SIMULATED.json`은 backend가 만든 기존 JSON의 `path`,`sha256`을 새 XML의 실제 값으로
변경한 **새 작업 사본**을 사용합니다. 닫힌 이전 실행의 원본을 덮어쓰지 않습니다.

`recipes/rain_adapter.py`는 공식 XML의 전진 이동 기여분을 제거하고 추정 SE3의3ms 노출
끝점으로 대체합니다. XML은x-right,y-up,z-back 및 아래쪽 이미지 원점입니다. 깊이 가림은
선분5표본 중 가장 긴 연속 가시 구간을 사용합니다. 전체6DoF 시뮬레이터 재현이나 실제
카메라 운동 보정 결과로 해석하지 않습니다. 강우량별 기존 입자 실현값을 재사용할 수 있으며
수정된 끝점으로 PBR을 다시 렌더링해야 합니다.

원래 PBR의 전역 평균 밝기 감산 전 rainy/fogged/mask 레이어를 내보냅니다. 선형광에서
입자 잔차·장면 조명·별도 대기 투과율을 합성합니다. 유효하지 않은 깊이는 원본을 보존합니다.
젖은 도로·물방울·진짜 강우량 보정은 포함하지 않습니다.

## Fog

`recipes/appearance.py:fog_scene`은 선형광에서 `T=exp(-βD)`, `I=TJ+(1−T)A`를 사용합니다.
`lighting_profile`은 같은 영상의 여러 대표 프레임으로 한 번 계산해 모든 강도에 고정합니다.
하늘은 RGB·깊이·상단 연결성으로 만든 휴리스틱이며 semantic GT가 아닙니다.
하늘 후보 깊이는 최소90 추정m로 처리합니다. 실제 시정거리나 거리 정확도를 주장하지 않습니다.
정적 CLI는 한 이미지로 조명 프로파일을 계산하는 축약 예제입니다.

## LowLight

`recipes/appearance.py:lowlight_linear`은 선형 RGB를 `2**EV`로 줄이고
`variance=aL+b`의 shot/read 잡음을 추가합니다. Bayer mosaic와 demosaic가 없습니다.
채널 상관 .85, 영상별a/b 계수와 프레임별 표준 정규 난수를 강도 간 공유합니다.
`noise=False`로 노출만 변경한 A, 기본 잡음의 B를 비교할 수 있습니다.
처리 RGB의 주간 저노출 합성이며 실제 RAW 센서·야간 촬영·자동 노출을 재현하지 않습니다.

## MotionBlur

`recipes/motion.py:target_blur`는 신뢰 가능한 흐름의p90 길이에 맞는 노출을 먼저 계산합니다.
60ms 안에 목표에 도달하면 관측 flow를 사용합니다. 관측 움직임이 부족하거나 신뢰 표본이
50픽셀 미만이면 **추가한 회전 카메라 흔들림**을 `controlled_camera_shake`로 표시합니다.
그 흔들림을 관측값이나 실제 촬영 노출로 해석하지 않습니다. 적분은 선형광·홀수 표본·
모든 강도20px 상한을 사용합니다. 실제p90/max/노출/가정 흔들림/신뢰도 면적을 기록합니다.
희소5FPS 흐름의 등속 근사이며 폐색을 완전히 복원하지 않습니다.

`estimate_scene_motion`은 RGB·깊이의PnP 제안에서 하늘/후드 휴리스틱과 비강체 잔차를
제외하고, 좌우 독립 적합의 일치도를 검사합니다. `validated_motion`은 이 **명시된 새 프로파일**의
조건부 통과입니다. 센서 기준의 실제 운동 검증이나 이전 실패 검사 통과를 뜻하지 않습니다.

## Glare

`recipes/glare.py`는 이전에 확인한 차량 발광등의 현재 광도를 측정하고 작은 source-colored
PSF를 합성합니다. 상대 램프의RGB 후보는 자동으로 검증된 광원으로 표시하지 않습니다.
유효한 램프가 없으면 장면별 하늘 방향을 참고한 **프레임 밖 태양 가정**으로 연속적인
veiling을 만듭니다. 이 경로는CONTROLLED이며, 붉은 도장/후드를 임의 광원으로 승격하지 않습니다.
광원색은 선형 RGB, 합성광은 양수입니다. 검은 중심을 만드는 감산과 거대한 붉은 원판을
사용하지 않습니다. 현재 개선판은Flare7K 이미지·제거망을 사용하지 않는 자체 합성기입니다.
특정 논문/렌즈 전체 재현으로 표시하지 않습니다.

## Snow

`recipes/snow_scene.py:SceneSnowRenderer`는 지속적인3D 입자·낙하/바람/펄럭임·깊이 가림을
사용합니다. 근거리 defocus와 부드러운4σ PSF, 장면광을 사용하고 입자율과 연무β를 따로
기록합니다. 기본5만 입자·10ms 노출이며 활성 임계값으로 같은 입자의 부분집합을 선택합니다.
강설량mm/h·적설·실제 카메라 초점 보정은 아닙니다.

연속 처리에서는 동일renderer 객체에 시간순 `render`를 호출합니다. 중간 프레임을 렌더링하지
않을 때 `render_ops.advance_snow`로 상태를 진행시킵니다. 같은 프레임의7강도는
`render_ops.snow_levels`로 같은 사전 상태를 복제하고 fingerprint 일치를 확인합니다.
스냅샷마다 새 객체를 만들면 시간적 연속성이 없습니다. 불안정한 자세로 재초기화한 구간은
별도 segment로 기록해야 합니다.

## 검증과 재현 범위

```bash
pip install pytest
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -B -m pytest -q -c /dev/null -p no:cacheprovider tests
```

테스트는 노출/색 경로, 저신뢰flow, 실제 흐림 목표, 광원 가정, XML 좌표/가림,
잘못된 깊이 처리, Snow 상태 공유를 검사합니다. CPU 단위 검증을 실제 환경 현실성이나
학습 성능으로 확대하지 않습니다. 코드 독립 검토에서 지적된3항목은 회귀 테스트로 보완했습니다.
새 Sol 이미지 검토는 수행하지 않습니다.

`KERNELS_SHA256.json`은 이 공개 사본의 코드 SHA입니다. 데이터·라벨·깊이 모델 가중치·
API 키·개인 서버 설정·생성 이미지는 포함하지 않습니다. BDD100K는 사용자가 별도로
권한을 확보해야 합니다. 저장소 공개는 제3자 데이터/자산의 재배포 허가가 아닙니다.

외부 소스와 의존성의 사용 조건은 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)를 확인하세요.
