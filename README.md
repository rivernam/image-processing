# OpenCV SearchMax

OpenCV 다중 스케일 템플릿 매칭을 실험하는 Windows 데스크톱 도구입니다. 이미지의 ROI 또는 이미지 파일 전체를 템플릿으로 학습하고, 생성 샘플이나 외부 이미지에서 후보를 검색하며, 생성 샘플은 정답 위치와 자동 평가할 수 있습니다.

## Windows 11 설치와 실행

Python 3.11 이상이 필요합니다. 저장소 루트에서 Windows `cmd` 또는 PowerShell을 열고 다음을 실행합니다.

```text
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
opencv-searchmax
```

PowerShell에서 실행 정책 때문에 활성화가 차단되면 활성화 없이 아래처럼 실행할 수 있습니다.

```text
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\opencv-searchmax.exe
```

개발 테스트 의존성까지 설치하려면 `python -m pip install -e ".[test]"`를 사용하고, `python -m pytest -v`로 전체 테스트를 실행합니다.

## 기본 작업 흐름

1. **Train**
   - `Load Image for ROI`로 이미지를 연 뒤 `Select ROI`를 켜고 마우스로 영역을 드래그한 다음 `Train from ROI`를 누릅니다.
   - 이미지 전체를 템플릿으로 쓰려면 `Train from File`을 누릅니다.
   - 학습이 완료되면 왼쪽 `ROI Template` 미리보기에 실제 검색 템플릿이 계속 표시됩니다. 미리보기를 클릭하면 중앙 이미지 영역에서 크게 볼 수 있습니다.
2. **1. Choose Test Source**
   - 실제 외부 이미지를 검색하려면 `Existing Images`를 선택합니다.
   - 재현 가능한 합성 샘플로 평가하려면 `Synthetic Test Images`를 선택합니다.
3. **2. Prepare Test Images**
   - `Existing Images`에서는 `Open Images to Search`로 검색할 이미지를 엽니다. 외부 이미지에는 정답 상자가 없으므로 검색 결과는 표시되지만 성공률/IoU 자동 평가는 제공되지 않습니다.
   - `Synthetic Test Images`에서는 Count와 Seed, Hue와 Saturation 범위를 정하고 필요하면 `Add Backgrounds for Generation`으로 배경을 추가한 뒤 `Generate Test Images`를 누릅니다. 생성 완료 후 샘플이 자동으로 테스트 목록이 됩니다.
4. **3. Run Search**
   - 검색 설정을 조정하고 공통 `Run Search`를 누릅니다. 현재 준비된 외부 또는 생성 이미지가 검색 대상이 됩니다. 진행 중에는 `Cancel`로 남은 작업을 취소할 수 있습니다.
5. **Results**
   - 결과 영역은 이미지 공간을 넓게 유지하기 위해 기본적으로 접혀 있습니다. `Show Results`를 눌러 펼치고, `Final Results` 행을 선택하면 해당 상자가 이미지에 강조됩니다.
   - 생성 샘플 검색 뒤에는 요약 성공률과 평균 IoU/시간이 표시되며, `File > Export Evaluation CSV`로 평가 기록을 저장할 수 있습니다.
   - `File > Save Project`와 `Load Project`는 학습 이미지/ROI, 검색·생성 설정, 최근 테스트/배경 경로를 저장하고 복원합니다. `Load Sample Metadata`는 재시작 후 생성 샘플과 정답을 복원합니다.

## 설정 기본값과 의미

| 설정 | 기본값 | 의미 |
| --- | ---: | --- |
| Count / Seed | 20 / 1234 | 생성 샘플 수와 재현용 난수 시드 |
| 생성 크기 | 1280×720 | 생성 이미지 크기 |
| 생성 Scale | 0.8–1.5 | 템플릿 삽입 배율 |
| Brightness | -12–12 | 생성 샘플 밝기 변화 |
| Contrast | 0.9–1.1 | 생성 샘플 대비 변화 |
| Blur / Noise sigma | 0 또는 3 / 0–3 | 생성 샘플 블러 커널과 가우시안 노이즈 |
| Hue min / max | -60° / 60° | ROI 물체에만 적용하는 색조 회전 범위 |
| Saturation min / max | 0.0 / 1.0 | ROI 물체의 채도 배율 범위. 0은 완전 회색, 1은 원래 채도 |
| Min / Max scale | 80% / 150% | 검색할 템플릿 배율 범위 |
| Step | 2% | 배율 탐색 간격 |
| Threshold | 0.80 | 후보로 인정할 정규화 매칭 점수 하한 |
| Maximum results | 1 | 이미지마다 반환할 최종 결과 수(1–100) |
| NMS IoU | 0.30 | 겹치는 후보를 억제하는 IoU 기준 |
| Mode | Color | 컬러 또는 그레이스케일 매칭 |

Hue와 Saturation 변형은 ROI 템플릿에만 적용되며 배경 색상은 바뀌지 않습니다. 같은 Seed와 설정을 사용하면 위치, 크기, 색상 변형을 포함한 생성 결과를 재현할 수 있습니다. Saturation은 0.0–2.0 범위를 지원하므로 1보다 큰 값을 사용하면 원본보다 채도를 높일 수 있습니다.

## 이미지 매칭 상세 로직

검색은 입력 이미지마다 아래 순서로 실행됩니다.

1. **검색 모드 선택**
   - `Color` 모드는 BGR 테스트 이미지와 컬러 ROI 템플릿을 그대로 사용합니다.
   - `Grayscale` 모드는 테스트 이미지를 그레이스케일로 변환하고 학습 시 만들어 둔 그레이스케일 템플릿을 사용합니다.
2. **Scale 목록 생성**
   - `Min scale`부터 `Step`씩 증가시키며 `Max scale` 이하의 값만 검색합니다.
   - 예를 들어 Min 80%, Max 85%, Step 2%이면 80%, 82%, 84%를 검색하며 Step에 맞지 않는 85%는 별도로 추가하지 않습니다.
   - 각 Scale의 템플릿 폭과 높이는 `round(학습 템플릿 크기 × Scale)`로 계산합니다. 결과 크기가 1픽셀 미만이거나 테스트 이미지보다 크면 해당 Scale을 건너뜁니다.
3. **템플릿 리사이즈**
   - 축소할 때는 `cv2.INTER_AREA`, 원본 크기 이상으로 확대할 때는 `cv2.INTER_LINEAR` 보간법을 사용합니다.
4. **점수맵 계산**
   - OpenCV `cv2.matchTemplate(..., cv2.TM_CCOEFF_NORMED)`로 템플릿의 각 배치 위치에 대한 정규화 상관 점수맵을 계산합니다.
   - UI에서는 후보 점수를 0–1 범위로 제한해 표시합니다. 값이 1에 가까울수록 템플릿과 더 유사합니다.
5. **후보점 추출**
   - 점수가 `Threshold` 이상이면서 주변 3×3 영역의 최대값인 위치만 후보가 됩니다.
   - 같은 점수의 최대 영역이 서로 이어진 plateau라면 연결된 영역마다 대표 위치 하나만 사용합니다.
   - Scale마다 점수 내림차순으로 최대 1,000개 후보를 유지합니다. 동점은 위쪽(Y가 작은 위치), 왼쪽(X가 작은 위치) 순으로 결정됩니다.
6. **전체 후보 정렬**
   - 모든 Scale의 후보를 합쳐 점수 내림차순으로 정렬합니다. 동점은 Y, X, Scale 오름차순으로 정렬해 같은 입력과 설정에서 항상 같은 결과 순서를 만듭니다.
7. **NMS와 결과 제한**
   - 가장 높은 점수 후보부터 선택하고, 이미 선택한 상자와의 IoU가 `NMS IoU`보다 큰 후보는 겹치는 중복으로 제거합니다.
   - IoU가 임계값과 정확히 같으면 유지됩니다.
   - 선택 결과가 `Maximum results`에 도달하면 종료하며 이것이 `Final Results`입니다.

두 상자 A와 B의 IoU는 다음과 같이 계산합니다.

```text
IoU = 교집합 면적 / (A 면적 + B 면적 - 교집합 면적)
```

`Advanced Search Settings`의 `Show pre-filter candidates`를 켜면 `Diagnostics` 표에 7단계 NMS와 결과 개수 제한을 적용하기 전의 점수순 상위 100개 후보가 표시됩니다. 이 목록은 원시 score-map 전체가 아니라 5단계에서 추출된 로컬 최대 후보입니다. 진단을 끄면 일반 검색은 진단 목록을 만들거나 전달하지 않습니다.

`Elapsed ms`는 한 이미지에 대해 Scale 탐색을 시작한 시점부터 후보 수집, NMS, 최종 선택이 끝날 때까지의 시간입니다. 이미지 읽기와 UI 표시는 포함하지 않으며, 같은 이미지에서 나온 모든 결과 행에는 같은 시간이 기록됩니다. 검출 결과가 없어도 실제 검색 시간은 유지됩니다.

### 생성 샘플 자동 평가

생성 샘플은 저장된 `truth_box`와 `Final Results`를 비교합니다. 최종 결과가 여러 개이면 점수가 가장 높은 결과가 아니라 **정답 상자와 IoU가 가장 큰 결과**를 평가 대상으로 선택합니다.

- `success`: 선택 결과와 정답 상자의 IoU가 0.5 이상
- `score`: 평가 대상으로 선택된 결과의 템플릿 매칭 점수
- `iou`: 선택 결과와 정답 상자의 상자 IoU
- `center_error`: 두 상자 중심점 사이의 유클리드 픽셀 거리
- `truth_scale`: 정답 상자의 폭 배율과 높이 배율의 산술 평균
- `scale_error_percent`: `(검출 Scale / truth_scale - 1) × 100`

최종 검출이 없으면 `success`는 false, IoU는 0이며 score, 중심 오차, Scale 오차는 비어 있습니다. 요약 값은 전체 샘플의 성공률과 평균 IoU/중심 오차/Scale 오차/검색 시간을 계산하며, 검출이 없는 샘플은 값이 존재하지 않는 중심·Scale 오차 평균에서 제외됩니다.

결과 표의 X와 Y는 검출 상자 왼쪽 위 위치의 이미지 픽셀 좌표이고, Width와 Height는 검출 경계 상자의 픽셀 크기입니다. Scale은 학습 템플릿 크기에 대한 검출 배율입니다.

평가 CSV의 `score`, `iou`, `center_error`, `scale_error_percent`, `elapsed_ms`는 위 자동 평가에서 계산한 값입니다.

## 저장 파일

- 프로젝트는 UTF-8 JSON이며 학습 이미지 경로, ROI, 검색 설정, 생성 설정, 최근 테스트/배경 경로를 저장합니다. 프로젝트 폴더 안의 경로는 가능한 경우 프로젝트 파일 기준 상대 경로로 기록되고, 외부 파일은 절대 경로가 될 수 있습니다. 프로젝트를 옮길 때 상대 경로 이미지도 함께 옮기고, 절대 경로 파일은 같은 위치에 유지해야 합니다.
- 생성 샘플 메타데이터 JSON은 프로젝트 JSON과 별도인 스키마 버전 1 파일입니다. 루트에는 `schema_version: 1`과 `samples` 목록이 있고, 각 샘플은 `image_path`, `truth_box`, `transform`, `seed`를 가집니다. `truth_box`는 생성 이미지 안의 정답 상자이고, `transform`에는 `scale`, `brightness`, `contrast`, `blur_kernel`, `noise_sigma`, `hue_shift_degrees`, `saturation_scale`이 들어갑니다. 상대 `image_path`는 메타데이터 JSON 파일의 폴더를 기준으로 해석됩니다. 이전 메타데이터에 색상 필드가 없으면 Hue 0°, Saturation 1.0으로 복원됩니다.
- 평가 CSV는 Excel에서 한글 경로를 읽을 수 있도록 UTF-8 BOM으로 저장되며 열은 `image, success, score, iou, center_error, scale_error_percent, elapsed_ms`입니다. CSV 내 이미지 경로는 실행 시 샘플 경로 문자열입니다.
- 생성 이미지는 `Generate Test Images`에서 선택한 폴더에 `sample_0001.png` 형식으로 저장됩니다. 같은 폴더/이름이 있으면 덮어쓸 수 있으므로 필요한 결과는 별도 폴더를 사용하십시오.

## 제한 사항

- 배율 변화만 탐색하며 회전과 원근(perspective) 변화는 검색하지 않습니다.
- Cognex VisionPro의 VPP 프로젝트를 읽거나 쓸 수 없으며 VPP 호환성을 제공하지 않습니다.
- 점수는 OpenCV 알고리즘의 값이므로 Cognex 도구의 점수와 수치적으로 동등하지 않습니다.
- 큰 이미지, 넓고 촘촘한 배율 범위, 많은 테스트 이미지는 처리 시간이 늘어날 수 있습니다.

Windows 11에서 한글 경로와 기본 작업 흐름을 지원하도록 구현되어 있습니다. 배포 전에는 실제 Windows 환경에서 디스플레이 배율 100%, 125%, 150% 각각으로 컨트롤 잘림 여부를 수동 확인하는 것을 권장합니다.
