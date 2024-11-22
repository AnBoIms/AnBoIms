# mmdetection을 통한 신분증, 간판 탐지
---

## 신분증 탐지 방법
- 모델: mask-rcnn 활용
- 모델 결과물: Idcard_Detection_output 폴더 확인

### 1. 가상환경 설치
- mmdetection 홈페이지에서 안내한 대로 가상환경 설치
- anboims project에서는 anaconda 가상환경에서 다음과같이 환경을 구축함.
pytorch: 2.4.1
mmcv: 2.1.0
mmengine: 0.10.5
cpu환경
openmim: 0.3.9
python: 3.8.20

### 2. mmdetection폴더 경로에서 다음과 같이 코드실행
python output_send.py

#### 모델 학습시키는 방법
1) 데이터셋 구축
- roboflow플랫폼 추천
- 원하는 format으로 라벨링 데이터셋 출력가능
2) 데이터셋 경로를 mmdetection 폴더 내에 적절한 위치에 저장.
- 정확한 경로 설정은 mmdetection에서 Train with customized datasets페이지에서 확인.
3) training 코드 예시
```
python tools\train.py configs\anboims_dataset\mask-rcnn_r50-caffe_fpn_ms-poly-1x_anboimsdataset.py
```
#### 학습된 모델 테스트하는 코드예시
```
python tools\test.py configs\anboims_dataset\mask-rcnn_r50-caffe_fpn_ms-poly-1x_anboimsdataset.py work_dirs\mask-rcnn_r50-caffe_fpn_ms-poly-1x_anboimsdataset\epoch_12.pth
```
#### 사진 출력물 테스트 코드예시
```
test_maskrcnn.py 
```