# Wu 주간·역광 Glare 비교

태양이 직접 보이는 원본 BDD MOT train/dev 영상 3개에서 각 10초·50프레임을 비교합니다.
기존 야간·차량등 결과는 보존하며, 이 주간 경로는 별도 탭입니다.

- [주간 결과 갤러리](http://210.94.179.18:30092/eligibility/wu_day_glare_20261101/index.html)
- [기존 야간·차량등 갤러리](http://210.94.179.18:30092/eligibility/vehicle_glare_20261101/index.html)
- [Wu et al., How to Train Neural Networks for Flare Removal, ICCV 2021](https://yichengwu.github.io/flare-removal/)
- [공식 코드와 데이터, 고정 revision](https://github.com/google-research/google-research/tree/83f1e7b59e762648c953de751d31f75720683622/flare_removal)

## 공식 방법에서 가져온 부분

`recipes/flare_wu_day.py`는 공식 `python/synthesis.py`와 `python/utils.py`의 수식을
NumPy/OpenCV로 이식했습니다. 공식 TensorFlow 파이프라인의 비트 단위 재현은 아닙니다.
공식 실측·시뮬레이션 PNG를 읽으며 16비트 PNG의 하위 비트도 보존합니다.

RGB를 gamma=2.0으로 선형화하고, 플레어의 채널별 배경을 제거하고, 공간 변환·클리핑·
화이트밸런스·RGB gain·21픽셀 Gaussian blur를 거쳐 장면 선형광에 더합니다.
DC 제거는 `(F-min(F))*max(F)/(max(F)-min(F)+epsilon)`이며 채널별로 계산합니다.
화이트밸런스는 `max(channel_mean)*F/(channel_mean+epsilon)`입니다.

약·중·강의 추가 광량은 **0.5 / 1.5 / 4.5**입니다. 동일 패턴·원본·위치·blur·gamma에서
gain만 바꿉니다. 원본의 노출 gain=1, 추가 scene noise=0, DC offset=0,
RGB 상대 gain=(1,1,1), Gaussian sigma=1을 고정합니다. 공식 학습 코드의 무작위
장면 crop·flip·noise·exposure 증강은 원본 주석과 강도 비교를 보존하려고 끕니다.

## BDD 영상용 추가 처리

주 에이전트가 원본을 직접 보고 실제 태양 영역을 확인한 위치를 기준으로 합니다.
주변 배경 optical flow의 앞뒤 일관성을 확인하고 현재 프레임의 밝은 태양 core로 위치를
보정합니다. 현재 태양이 확인되지 않으면 그 프레임은 효과를 중지하고 기록합니다.
광원이 없는 하늘, 도로, 차량 문에 임의의 새 광점을 배치하지 않습니다.

실측 패턴은 템플릿 광원→관측 태양, 템플릿 중심→장면 중심의 두 위치를 맞추는 유사변환을
사용합니다. 시뮬레이션 산란 패턴은 원래의 source core를 관측 태양에 정렬합니다.
이 정렬·추적은 BDD용 확장이며 Wu 논문 자체의 영상 추적 알고리즘이 아닙니다.
렌즈 중심을 영상 중심으로 근사하므로 실제 카메라에 보정된 광학 시뮬레이션은 아닙니다.
원본에도 자연스러운 플레어가 있어서 추가 합성에 의한 변화로 해석해야 합니다.

## 실행

NumPy, OpenCV, Pillow만 사용하며 GPU·모델·API 키가 필요 없습니다.
공식 패턴을 별도로 내려받고 source core를 직접 확인합니다. 패턴의 가장 밝은 ghost를
광원으로 오인하지 마세요. 주 광원이 화면 밖인 패턴은 이 경로에서 제외합니다.

```json
{
  "pattern": {
    "path": "/data/official_pattern.png",
    "sha256": "다운로드한 파일의 SHA256",
    "kind": "captured",
    "origin": [784.2108974358974, 167.8852564102564]
  },
  "frames": ["/data/frame000.jpg", "/data/frame001.jpg"],
  "anchor_index": 0,
  "reviewed_sun_xy": [960.0, 100.0]
}
```

위 origin은 공식 `captured/frame_0571.png`에만 대응하는 확인 위치입니다.
임의 패턴에 같은 위치를 사용하면 안 됩니다.

```bash
python generate_wu_day.py --manifest clip.json --output outputs/wu_day_attempt1
python -m unittest discover -s tests -p test_flare_wu_day.py
```

출력 PNG와 프레임별 원본 SHA·광원·변환·출력 SHA를 기록하며 기존 결과를 덮어쓰지 않습니다.
전체 영상을 한 번에 메모리에 읽으므로 이 CLI는 짧은 표본 구간용입니다.
약·중·강은 탐색 후보이며 최종 severity나 학습·검증 데이터 승인이 아닙니다.

## 출처와 라이선스

Wu et al. 공식 코드: Apache License 2.0. 수식 이식 출처는 위 고정 revision의
`python/synthesis.py`, `python/utils.py`이며 포트·정렬·추적 변경을 명시했습니다.
공식 flare PNG: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), Wu et al.
데이터와 라이선스 안내는 공식 저장소 README를 따릅니다. 패턴·BDD 사진은 이 Git 저장소에
포함하지 않습니다. 사진의 BDD 이용 조건은 별도입니다.
