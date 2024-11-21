from mmdet.apis import init_detector, inference_detector
import mmcv
import matplotlib.pyplot as plt
import torch
from torchvision.ops import nms
import cv2
import numpy as np

config_file = 'configs/anboims_dataset/cascade-mask-rcnn_r50_fpn_1x_coco_anboimsdataset.py'
checkpoint_file = 'work_dirs/cascade-mask-rcnn_r50_fpn_1x_coco_anboimsdataset/epoch_12.pth'

# 1. 모델 초기화
model = init_detector(config_file, checkpoint_file, device='cpu')  # GPU 사용 시 'cuda:0', CPU 사용 시 'cpu'로 설정

# 2. 테스트할 이미지 파일 경로 설정
img = 'test_images/KakaoTalk_20241109_072449401_04.jpg'
image = mmcv.imread(img)

# 3. 이미지에서 객체 탐지
result = inference_detector(model, img)

# 4. 결과 필터링 및 NMS 적용
score_thr = 0.3
iou_threshold = 0.3

bboxes = result.pred_instances.bboxes.cpu()
scores = result.pred_instances.scores.cpu()
labels = result.pred_instances.labels.cpu()
masks = result.pred_instances.masks.cpu()

keep_scores_idx = scores > score_thr
bboxes = bboxes[keep_scores_idx]
scores = scores[keep_scores_idx]
labels = labels[keep_scores_idx]
masks = masks[keep_scores_idx]

keep_nms_idx = nms(bboxes, scores, iou_threshold)
bboxes = bboxes[keep_nms_idx]
scores = scores[keep_nms_idx]
labels = labels[keep_nms_idx]
masks = masks[keep_nms_idx]

# 5. 폴리곤 출력 및 시각화
for bbox, score, label, mask in zip(bboxes, scores, labels, masks):
    # 클래스와 점수 정보
    class_name = model.dataset_meta["classes"][label]
    x1, y1, x2, y2 = map(int, bbox)
    cv2.putText(image, f'{class_name}: {score:.2f}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 마스크를 폴리곤으로 변환
mask = mask.cpu().numpy().astype(np.uint8)  # Tensor -> NumPy 변환 및 dtype 설정
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 윤곽선 그리기
cv2.drawContours(image, contours, -1, (0, 255, 0), thickness=2)

# 반투명 마스크 오버레이
overlay = image.copy()
mask_color = (0, 255, 0)  # 폴리곤 내부 채우기 색상 (튜플로 정의)
for contour in contours:
    cv2.fillPoly(overlay, [contour], mask_color)
image = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)


# 6. 결과 이미지 저장 및 출력
output_path = 'output/result_polygon.jpg'
mmcv.imwrite(image, output_path)

plt.imshow(image[:, :, ::-1])  # BGR을 RGB로 변환하여 표시
plt.axis('off')
plt.show()
