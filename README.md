# BDD Corruption Generators

2026-11-01 추가: [Flare7K* PGS 생성 후 선별](FLARE7KSTAR.md)은 별도 경로입니다.
기존 광원 적합성 필터를 사전 적용하지 않고, 1~3개의 광원·산란 패턴을 합성합니다.
전체 MOT 2,000영상·397,688프레임의 약/중/강 생성을 시작하는 새 실험에 사용합니다.
`generate_flare7kstar.py`는 오프라인 DPT 추론과 영상용 시간 처리를 제공합니다.
공식 코드가 공개되지 않아 논문 수식의 재구현이며, 자동 지표와 사람의 시각 승인을 구분합니다.
OpenAI 또는 외부 모델 API를 호출하지 않습니다.

2026-10-31 추가: **Glare Eligibility Filter**는 생성 전에 원본 영상의 광원 조건을 별도로
검사합니다. 전체 BDD100K MOT2,000영상은1fps후보 조사→후보 구간5fps추적 경로를 사용합니다.
VERIFIED/CONTROLLED/UNCERTAIN/INELIGIBLE와 실제 검사한 구간, 원본SHA, 광원ID·종류·광도·
기하·추적 근거를 기록합니다. VERIFIED는 **보정되지 않은 시작 규칙의 자동 통과**이며
확률·실제 방사휘도·수작업 승인·렌더 품질·학습 승인이 아닙니다. 음성1fps사이 구간을 전수
검사로 표시하지 않습니다. 차량 주석 없는test는 차량등을 자동 승인하지 않습니다.

`recipes/glare_eligibility.py`가 독립 필터입니다. 밝은 하늘이나 주간 속성만으로 가정 태양을
허용하지 않습니다. 가정 태양에는 외부 확인 하늘·방향 신뢰 근거·명시적 승인이 모두 필요합니다.
신호등/가로등용 semantic detector는 초기 경로에 포함하지 않아 미지 후보는UNCERTAIN입니다.
임계값은 정책에 고정된 초기값이며 일부 수작업 표본과 대조하는 보정은 남아 있습니다.

```bash
python screen_glare.py --manifest clip_frames.json --output clip_eligibility.json
python generate.py --type Glare --image frame.jpg --frame 100 --level .5 \
  --eligibility clip_eligibility.json --output outputs/glare.jpg
```

`clip_frames.json`은 `video_id`, 선택적 공식 `attributes`, 시간순 `frames`를 담습니다.
각 frame은 `path`,0기반5fps `frame_index`,`timestamp_s`,선택적 `objects` 목록입니다.
객체는 `track_id`,`category`,`bbox`(원본xyxy),boolean `occluded`/`truncated`를 사용합니다.
주석이 없으면 빈 목록을 사용합니다. CLI는 연속5fps프레임을 검사하는 예제입니다.
대규모1fps→5fps작업·사이트에는 원본 영상 분할을 유지하고 모델 평가/학습을 추가하지 않습니다.
새 Glare CLI에는 SHA가 맞는 적합성 결과가 필수이며, 부적격/불확실 프레임에 가짜 태양이나
원본 대체 이미지를 생성하지 않습니다. CONTROLLED 생성은 추가로 `--controlled`가 필요합니다.
현재 사이트 판정 작업은 생성 전 검사이며 Glare 이미지 렌더링은 수행하지 않습니다.

Rain · Fog · LowLight · MotionBlur · Glare · Snow의 CPU 연구용 생성 코드입니다.
Shadow는 제외합니다. 2026-10-31 R2는 소스 감사에서 확인된 설정·광원·운동·저장 오류를
수정하고 **생성 완료와 학습 승인**을 분리합니다. API 키·개인 서버·학습 가중치 없이 커널을
사용할 수 있습니다. 공식 Rain 런타임과 streak DB는 별도로 준비해야 합니다.

현재 범위는 BDD100K MOT 8개 train/dev 영상의 대표5프레임, 6종·7강도 비교입니다.
실제 거리/카메라 보정, 실제 BDD 연속 영상의 시간적 품질, 검출기·Adapter 정확도는
미검증입니다. 정적 사진과 합성100프레임 CPU 테스트를 학습 데이터 승인으로 해석하지 않습니다.

## 실행

Python3.10 환경에서 NumPy·OpenCV·Pillow를 준비합니다.

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python generate.py --help
python generate.py --type Fog --image frame.jpg --depth depth.npy --level .015 --output outputs/fog.jpg
python generate.py --type LowLight --image frame.jpg --level -1.5 --noise-model independent --output outputs/dark.jpg
python generate.py --type MotionBlur --image frame.jpg --flow flow.npz --level 6 --output outputs/blur.jpg
python generate.py --type Snow --image frame.jpg --depth depth.npy --level .5 --scene-haze-beta-max .004 --controlled --output outputs/snow.jpg
```

RGB는1280×720 또는640×360입니다. 출력은640×360 JPEG95,4:4:4입니다.
Depth NPY는 원본 RGB 또는 출력 크기와 일치하는 camera-z 배열이어야 합니다. 음수는
거부하고0/nonfinite는 미지 기하로 처리합니다. 축소에는 nearest를 사용해 객체 경계에서
앞·뒤 깊이의 중간값을 새로 만들지 않습니다. 크기 일치는 실제 정합의 증명이 아닙니다.
Flow NPZ는640×360×2 유한 실수 `flow`와640×360 boolean `confidence`를 담습니다.
내부 파라미터는 출력에서fx=fy450,cx320,cy180의 BDD 가정값이며 실제 보정값이 아닙니다.

`outputs/name.jpg`, `name.json`, `name.READY.json`이 생깁니다. 이미지·JSON의 SHA를
검사한 뒤 READY를 마지막으로 공개합니다. 기존 파일은 덮어쓰지 않습니다.
완료 마커 없는 파일은 샘플로 소비하지 않습니다.

## 강도

| 종류 | 비교7강도 | 고정·별도 기록 |
|---|---|---|
| Rain |5,15,25,50,75,100,200mm/h|노출3ms·시드7301·같은 공식 입자 설정|
| Fog |β=.00375,.005,.0075,.01,.015,.03,.06|영상별 고정 대기광·같은 추정 깊이|
| LowLight |−.5,−.75,−1,−1.5,−2,−3,−4EV|같은 영상의 잡음 계수·강도 간 난수 공유|
| MotionBlur |p90=1,2,3,4,6,8,12px|정적 비교 모드·전 강도20px 상한|
| Glare |gain=.1,.25,.4,.5,.75,1,1.5|같은 광원·PSF 반경·독립 bloom/veiling gain|
| Snow |활성 입자율10,25,40,50,65,80,100%|같은 입자ID·궤적·활성 임계값|

`SEVERITY.json`은 최종 약·중·강 선정값이 아닙니다. 극심한 강우, 높은β,−4EV와gain1.5는
상한/실패 진단 후보입니다. 이전 MotionBlur의ms와 현재px는 같은 단위가 아닙니다.

## Snow

`recipes/snow_scene.py:SceneSnowRenderer`는 지속적인3D 입자·바람·깊이 가림·부드러운
거리별PSF·장면광을 사용합니다. 기본5만 입자,10ms·5노출 표본입니다.
`scene_haze_beta_max=.004`를 실제 적용하며0으로 끄거나 상한을 바꿀 수 있습니다.
기존 `SnowRenderer.config['haze_beta']`를 SceneSnowRenderer에 명시하면 오류를 반환합니다.
다른 설정을 조용히 무시하지 않습니다.

미지 깊이를 하늘로 추정하지 않습니다. `--verified-sky sky.npy` 또는 렌더러의 `sky_mask`
인자로 **별도로 확인한 boolean 마스크**만 전달합니다. 하늘은 회전만 반영해 노출 중 투영하고,
유한 깊이의 앞 물체가 가립니다. 미지 객체 영역은 원본을 보존합니다. 현재40프레임에는
확인된 하늘 마스크가 없으므로 자동 승인하지 않습니다. 하늘 연무의90m는 명시적 proxy입니다.

연속 입력은 같은 객체에0.2초 간격으로 시간순 `render()`를 호출합니다.
`render_ops.snow_levels()`는 같은 사전 상태를 복제해 강도를 비교하고 원본 상태를 한 번 진행합니다.
복사·렌더 시간을 각각 기록합니다. 최적화 전에 실제 병목을 측정하며 입자 상태 공유를 유지합니다.
현재40프레임에서는 입자율 고정1.0에서 연무 상한0/.004/.008 비교도 생성합니다.

## Glare

`recipes/emitter.py`는 `tail_light`, `headlight`, `streetlight`별 광도·주변 대비·연결 면적을
검사합니다. `identity_verified`, `photometric_pass`, `tracking_pass`를 따로 기록하고 모두
통과해야 `SOURCE_VERIFIED`입니다. bbox 지정만으로 광원을 승인하지 않습니다.
`LampTracker`의 인접 프레임 LK 전후방 검사 결과는 RGB·bbox·발광 core·track ID·광원 종류에
묶습니다. 다른 램프 위치로 추적 근거를 옮겨 사용할 수 없습니다. 초기화는 추적 통과가 아닙니다.

```bash
python screen_glare.py --manifest clip_frames.json --output clip_eligibility.json
python generate.py --type Glare --image frame.jpg --frame 100 --level .5 --eligibility clip_eligibility.json --output glare.jpg
```

이전R2의 `--bbox`/`--source-record` 직접 합성 경로는 새 CLI에서 적합성 결과로 대체했습니다.
기존 낮은 수준의 신원·추적 커널은 이전 재현을 위해 남아 있습니다. 새 결과는 원본 파일SHA와
640×360처리RGB의 `rgb.tobytes()` SHA를 모두 확인하고 적격 광원만 합성합니다.

새 경로의 `--controlled`는 필터가 외부 하늘·태양 방향·명시 승인 근거로 CONTROLLED를
반환한 경우에만 합성을 허용합니다. 밝은 하늘·주간 속성만으로 허용하지 않습니다. 야간/터널의 무광원 장면은
`INELIGIBLE`로 거부하며 가짜 태양이나 원본 대체 이미지를 저장하지 않습니다.
태양 방향은 같은 영상의 고정 프로파일을 따릅니다. 현재 분류는 추정값이며 실제 태양 관측이 아닙니다.

Bloom과 넓은 약한 veiling 성분은 독립 gain으로 제어합니다. severity가 반경을 계속 키우지
않습니다. 선형광에 양의 성분을 더하며 검은 중심 감산·거대한 붉은 원판을 사용하지 않습니다.
이 모델은 보정된 렌즈 재현이나 Flare7K 망이 아닙니다.

## Rain

[astra-vision/rain-rendering](https://github.com/astra-vision/rain-rendering)의 공식 입자/PBR
런타임을 외부에서 사용합니다. renderer revision `0a9d8ca739d71d4b09f61567c6f6cea94a79868e`,
particle binary revision `6a130aa7294dd980940416e8e0b21fef7112efaa`입니다.
URL·SHA는[EXTERNAL_ASSETS.json](EXTERNAL_ASSETS.json)에 있습니다. 원본 upstream은 보존합니다.
실행 호환성·레거시 라이브러리·streak DB 준비는 별도이며 자동 설치하지 않습니다.

`rain_runtime/`는 프로젝트의 `recipes/common.py`와 upstream `common` 패키지의 충돌을 피합니다.
카메라 위치[1.5,1.5,.3],look-at[1.5,1.5,-1],초점4mm·픽셀4.444μm·5FPS·gain20·f/6·노출3ms입니다.
출력 작업 폴더에는 `RAIN_INPUTS.json`, `input/<video>/rgb/0000.png`(1280×720),
동일 크기 `input/<video>/depth/0000.npy`, 외부 `rainstreakdb/env_light_database/`를 준비합니다.
RAIN_INPUTS는 `video`, 시간순 `samples:[{"image_id":1}]`, `speeds_kmh:[20.0]` 형식입니다.

```bash
export BDD_RAIN_RUNTIME=/path/to/qualified_runtime
export PYTHONPATH="$BDD_RAIN_RUNTIME/upstream"
python rain_runtime/rain_backend.py simulate /path/to/output 25
python rain_tools.py export-patch --upstream "$BDD_RAIN_RUNTIME/upstream" --output /path/to/patched_upstream
python rain_tools.py correct --input official_camera0.xml --motion motion.json --depths depths.npy --output corrected_camera0.xml
export PYTHONPATH=/path/to/patched_upstream
# 새 작업 사본의 SIMULATED.json에 corrected XML path/SHA와 corrected_camera0.json을 SE3_correction으로 지정
python rain_runtime/rain_backend.py render /path/to/output 25
python rain_tools.py compose --image frame.jpg --depth depth.npy --layers /path/to/0000.png.layers.npz --rate 25 --output rain.jpg
```

`motion.json`의 예시 구조는 다음과 같습니다. 생략/중복/null 프레임·시간 대응을 거부합니다.

```json
{"render_exposure_indices":[0],"base_speeds_kmh":[20.0],"motions":[{
  "xml_index":0,"frame_id":100,"timestamp_s":20.0,"twist":[0,0,-5.5,0,0,0],
  "qa":{"validated_motion":true,"fallback_assumption":null,"twist_capped":false,
        "static_support_pixels":1200,"median_reprojection_pixels":0.8}
}]}
```

기본 경로는 수치적 운동 검사를 요구합니다. `--diagnostic`을 명시하면 실패/가정 운동을
사용할 수 있지만 승인 상태를 부여하지 않습니다. `validated_motion`은 실제 카메라/깊이
보정과 분리됩니다. `render_exposure_indices`는 XML의`i`요소0기반 순서이며 실제 렌더링에
사용할 모든 노출을 포함해야 합니다. 렌더러에서도 XML SHA·프레임ID·시간 대응을 다시 확인합니다.
기본 renderer는 앞쪽N노출을 사용하므로 비연속 인덱스는 XML을 별도로 추출/재인덱싱해야 합니다.
마지막의 추가 미사용 노출은 대응 목록에서 제외할 수
있지만 보정/미보정 인덱스를 모두 기록합니다. `[v,w]`는 left twist/초입니다.
`depths.npy`는 motion 목록 순서의N×H×W camera-z 깊이입니다.

공식 XML의 전진 기여분을 제거하고 SE3로3ms 끝점을 보정합니다. XML은x-right,y-up,z-back,
이미지 아래쪽 원점입니다. 깊이 가림은 선분5표본의 최장 가시 구간입니다. 전체6DoF 입자
시뮬레이터 재현이 아닙니다. rainy/fogged 레이어를 전역 밝기 감산 전에 내보내고 선형광
잔차를 합성합니다. 마스크는 강제로 곱하지 않으며 mask 안/밖 잔차·좌우 밀도·깊이 상관을 기록합니다.
추정 조명·강우량별 대기감쇠의 실제 악천후 보정은 미완료입니다.

`recipes/wetness.py`는 독립적인 선택적 도로막 근사입니다. 시간에 따라 누적되고 확인된
도로 마스크가 필요합니다. 표면 광학은 미보정이며 현재 생성 sweep에서는 **비활성**입니다.

## MotionBlur

`STATIC_TARGET_P90`는 사진마다 목표px를 맞추는 보정 모드입니다. 관측 flow와60ms 이하
노출로 불가능하면 명시적 `controlled_camera_shake`를 추가합니다. 실제 이동과 가상 흔들림을
별도로 기록합니다. edge sharpness loss와 차량 bbox 내부 손실은 원본의 신뢰 경계에서 측정합니다.
차량 라벨은 렌더링 제어가 아니라 별도 오프라인 품질 지표에만 사용합니다.

`TemporalExposureRenderer`는 구간 내 노출시간을 고정합니다. 움직임이 작아져도 매 프레임
p90을 강제하거나 shake로 자동 전환하지 않습니다. 선택적 가상 회전은 시간에 연속적인
고정 seed 궤적입니다. `exposure_plan()`은 구간 flow 통계로 노출 후보와 도달 가능성을 기록합니다.
낮은 flow 신뢰 경계는 보수적으로 마스킹하며 폐색을 완전히 복원하지 않습니다.

```bash
python generate.py --type MotionBlur --image frame.jpg --flow flow.npz --level 6 \
  --temporal-exposure-ms 20 --timestamp-s 1.2 --output temporal_blur.jpg
```

실제 영상에서는 같은 객체와 고정 exposure를 유지해 시간순으로 호출합니다.
이 CLI는 한 프레임의 예제이며 실제 연속 영상 검증을 대신하지 않습니다.

## LowLight

선형 RGB의`2**EV`감광은 유지하고 A노출만/B독립shot/read/C상관 .85/D고정패턴을 비교합니다.
`--noise-model exposure_only|independent|correlated|fixed_pattern`으로 선택합니다.
Shot은 처리 RGB에서의 Gaussian 근사입니다. 센서·RAW 보정 결과로 주장하지 않습니다.
고정패턴은 영상/카메라별 같은 seed, shot/read는 프레임별 seed입니다. EV 간 표준 난수는 공유합니다.
Bayer/demosaic가 없고 주기적 색 격자를 추가하지 않습니다. 실제 밝기·이론/양자화 후 SNR을 기록합니다.

## Fog

`T=exp(-βD), I=TJ+(1−T)A`를 선형광에서 적용합니다. 영상별 조명 프로파일은 모든 강도에
고정합니다. 어두운 장면에sRGB .48 하한을 강제하지 않습니다. 추정 하늘 휴리스틱으로 깊이를
90m로 승격하지 않으며 확인된 별도 sky mask에만 명시적 proxy를 허용합니다.
미지 깊이는 원본을 보존합니다. 거리대별 평균 투과율·유효 깊이 비율·깊이 분포를 기록합니다.
β의 실제 시정거리 해석, 영상 간 실제 미터 척도, 야간 국소 산란 모델은 미검증입니다.

## 완료와 품질 승인

`recipes/sample_io.py:read_completed()`는 READY·이미지·JSON의 크기/SHA를 확인합니다.
`recipes/quality.py`의 상태는 `GENERATED → STATIC_PASS → TEMPORAL_PASS → RELEASED`이며
부적합 결과는`REJECTED`입니다. 생성/CPU 테스트만으로 다음 단계로 승격하지 않습니다.
정적 물리/시각 검토, 시간 일관성, 라벨·분할 무결성, 검출 성능 변화, Adapter 회복 성능,
실험 프로토콜의 **해당 이미지 SHA에 묶인 검증 보고서**가 필요합니다.
`read_released()`는 완료/승인 근거가 없으면 학습·평가 소비를 거부합니다.
현재40프레임은 생성/비교 후보이며 RELEASED 데이터가 아닙니다.

```bash
pip install pytest
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -B -m pytest -q -c /dev/null -p no:cacheprovider tests
```

CPU 검사는 합성100프레임 상태, 연무 설정, 하늘/미지 분리, 광원/추적 결합, XML 완전성,
mask 정합, 이미지 선명도, 입력 정합, 중단된 저장, 승인 우회를 검사합니다.
독립 Sol **코드** 검토의4항목도 회귀 테스트로 보완했습니다. 새 Sol 이미지 검토는 수행하지 않습니다.

외부 데이터·모델·streak DB·개인 서버 설정·키·생성 사진·이전 비공개 Git 이력은 포함하지 않습니다.
사용 조건은[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), 소스 SHA는
[KERNELS_SHA256.json](KERNELS_SHA256.json)을 확인하세요.
