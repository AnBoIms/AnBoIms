from mmdet.apis import init_detector, inference_detector
import mmcv
import matplotlib.pyplot as plt
import cv2
import numpy as np

# 1. Mask R-CNN config 파일과 학습된 모델 파일 경로 설정
config_file = 'configs/anboims_dataset/mask-rcnn_r50-caffe_fpn_ms-poly-1x_anboimsdataset.py'
checkpoint_file = 'work_dirs/mask-rcnn_r50-caffe_fpn_ms-poly-1x_anboimsdataset/epoch_12.pth'

# 2. 모델 초기화
model = init_detector(config_file, checkpoint_file, device='cpu')  # GPU 사용 시 'cuda:0', CPU 사용 시 'cpu'

# 3. 테스트할 이미지 파일 경로 설정
img = 'test_images/KakaoTalk_20241109_072449401_08.jpg'
image = mmcv.imread(img)

# 4. 이미지에서 객체 탐지
result = inference_detector(model, img)

# 5. 마스크 시각화
# 마스크 정보 추출
masks = result.pred_instances.masks.cpu().numpy()  # 마스크 정보
scores = result.pred_instances.scores.cpu().numpy()  # 신뢰도 점수
labels = result.pred_instances.labels.cpu().numpy()  # 클래스 라벨

score_thr = 0.3  # 신뢰도 임계값 설정
output_image = image.copy()

for i in range(len(masks)):
    if scores[i] < score_thr:
        continue

    # 마스크 시각화 (반투명 효과)
    mask = masks[i]
    color = np.random.randint(0, 255, size=(3,), dtype=np.uint8)  # 무작위 색상
    output_image[mask] = output_image[mask] * 0.5 + color * 0.5

# 6. 결과 이미지 저장
output_path = 'output/result_mask_segmentation.jpg'
mmcv.imwrite(output_image, output_path)

# 7. 결과 확인 (선택적)
plt.imshow(output_image[:, :, ::-1])  # BGR -> RGB 변환하여 표시
plt.axis('off')
plt.show()

print(f"Processed {img} -> Saved to {output_path}")
