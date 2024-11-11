from mmdet.apis import init_detector, inference_detector
import mmcv
import matplotlib.pyplot as plt
import torch
from torchvision.ops import nms
import cv2

# 1. Faster R-CNN config 파일과 학습된 모델 파일 경로 설정
config_file = 'configs/anboims_dataset/faster-rcnn_r50_fpn_1x_coco_anboimsdataset.py'  # Faster R-CNN 설정 파일 경로
checkpoint_file = 'work_dirs/faster-rcnn_r50_fpn_1x_coco_anboimsdataset/epoch_12.pth'  # 학습된 Faster R-CNN 모델 체크포인트 경로

# 2. 모델 초기화
model = init_detector(config_file, checkpoint_file, device='cpu')  # GPU 사용 시 'cuda:0', CPU 사용 시 'cpu'로 설정

# 3. 테스트할 이미지 파일 경로 설정
img = 'test_images/KakaoTalk_20241109_072449401_08.jpg'  # 테스트할 이미지 파일 경로
image = mmcv.imread(img)

# 4. 이미지에서 객체 탐지
result = inference_detector(model, img)

# 5. 결과 필터링 및 NMS 적용
score_thr = 0.3
iou_threshold = 0.3  # NMS IoU 임계값 설정

# 바운딩 박스, 점수, 라벨 추출
bboxes = result.pred_instances.bboxes.cpu()  # 바운딩 박스 좌표
scores = result.pred_instances.scores.cpu()  # 신뢰도 점수
labels = result.pred_instances.labels.cpu()  # 클래스 라벨

# 신뢰도 필터링 적용
keep_scores_idx = scores > score_thr
bboxes = bboxes[keep_scores_idx]
scores = scores[keep_scores_idx]
labels = labels[keep_scores_idx]

# NMS 적용
keep_nms_idx = nms(bboxes, scores, iou_threshold)
bboxes = bboxes[keep_nms_idx]
scores = scores[keep_nms_idx]
labels = labels[keep_nms_idx]

# 6. 중복 바운딩 박스에서 작은 바운딩 박스 선택
# IoU가 일정 이상으로 겹치는 바운딩 박스 중 면적이 작은 것만 남깁니다.
selected_bboxes = []
selected_scores = []
selected_labels = []

for i in range(len(bboxes)):
    x1, y1, x2, y2 = bboxes[i]
    area_i = (x2 - x1) * (y2 - y1)
    is_duplicate = False

    for j in range(len(selected_bboxes)):
        x1_sel, y1_sel, x2_sel, y2_sel = selected_bboxes[j]
        area_sel = (x2_sel - x1_sel) * (y2_sel - y1_sel)
        
        # IoU 계산
        inter_x1 = max(x1, x1_sel)
        inter_y1 = max(y1, y1_sel)
        inter_x2 = min(x2, x2_sel)
        inter_y2 = min(y2, y2_sel)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        union_area = area_i + area_sel - inter_area
        iou = inter_area / union_area if union_area > 0 else 0

        # IoU가 높으면 중복으로 간주, 작은 바운딩 박스만 남기기
        if iou > iou_threshold:
            is_duplicate = True
            if area_i < area_sel:
                selected_bboxes[j] = bboxes[i]
                selected_scores[j] = scores[i]
                selected_labels[j] = labels[i]
            break

    if not is_duplicate:
        selected_bboxes.append(bboxes[i])
        selected_scores.append(scores[i])
        selected_labels.append(labels[i])

# 최종 선택된 바운딩 박스를 numpy 배열로 변환
selected_bboxes = torch.stack(selected_bboxes).numpy()
selected_scores = torch.tensor(selected_scores).numpy()
selected_labels = torch.tensor(selected_labels).numpy()

# 7. 필터링된 결과를 시각화
for bbox, score, label in zip(selected_bboxes, selected_scores, selected_labels):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 바운딩 박스 그리기
    cv2.putText(image, f'{model.dataset_meta["classes"][label]}: {score:.2f}', (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 클래스와 점수 표시

# 8. 결과 이미지 저장
output_path = 'output/result.jpg'
mmcv.imwrite(image, output_path)

# 9. 결과 확인 (선택적)
plt.imshow(image[:, :, ::-1])  # BGR을 RGB로 변환하여 표시
plt.axis('off')
plt.show()
