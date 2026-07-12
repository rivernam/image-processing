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

점수는 OpenCV의 정규화 상관 템플릿 매칭 값이며 UI에서는 0–1 범위로 표시됩니다. `Final Results`는 Threshold를 통과한 후보에 NMS(겹침 억제)를 적용한 뒤 점수순으로 `Maximum results`개까지 제한한 결과입니다. `Advanced Search Settings`의 `Show pre-filter candidates`를 켜면 `Diagnostics` 표에 결정적 점수순 **NMS 이전 상위 100개 후보**가 표시됩니다. 원시 score-map 전체는 아닙니다.

결과 표의 X와 Y는 검출 상자 왼쪽 위 위치의 이미지 픽셀 좌표이고, Width와 Height는 검출 경계 상자의 픽셀 크기입니다. Scale은 학습 템플릿 크기에 대한 검출 배율입니다. Elapsed ms는 해당 이미지의 전체 배율 탐색, 후보 선택, NMS까지 걸린 검색 시간이며 결과 행별 독립 측정 시간이 아닙니다. 같은 이미지의 결과 행에는 같은 검색 시간이 표시됩니다.

생성 샘플의 `success`는 최종 검출 상자와 생성 정답 상자의 IoU가 0.5 이상이라는 뜻입니다. CSV의 `score`, `iou`, `center_error`, `scale_error_percent`, `elapsed_ms`는 각각 선택된 검출의 점수, 상자 IoU, 중심점 픽셀 오차, 정답 대비 배율 오차(%), 이미지 검색 시간을 뜻합니다. 검출이 없으면 score와 일부 오차 값은 비어 있을 수 있습니다.

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
