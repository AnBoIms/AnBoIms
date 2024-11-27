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
img = 'test_images/KakaoTalk_20241126_140949348_02.jpg'  # 테스트할 이미지 파일 경로
image = mmcv.imread(img)

# 4. 이미지에서 객체 탐지
result = inference_detector(model, img)

# 결과가 없는 경우 대비
if not hasattr(result, 'pred_instances') or result.pred_instances is None:
    print("No predictions detected.")
    selected_bboxes = torch.empty((0, 4))
    selected_scores = torch.empty(0)
    selected_labels = torch.empty(0)
else:
    # 바운딩 박스, 점수, 라벨 추출
    bboxes = result.pred_instances.bboxes.cpu() if result.pred_instances.bboxes.numel() > 0 else torch.empty((0, 4))
    scores = result.pred_instances.scores.cpu() if result.pred_instances.scores.numel() > 0 else torch.empty(0)
    labels = result.pred_instances.labels.cpu() if result.pred_instances.labels.numel() > 0 else torch.empty(0)

    # 바운딩 박스가 비어 있는 경우 대비
    if bboxes.numel() == 0:
        print("No bounding boxes detected after filtering.")
        selected_bboxes = torch.empty((0, 4))
        selected_scores = torch.empty(0)
        selected_labels = torch.empty(0)
    else:
        # 신뢰도 필터링 적용
        score_thr = 0.1
        keep_scores_idx = scores > score_thr
        bboxes = bboxes[keep_scores_idx]
        scores = scores[keep_scores_idx]
        labels = labels[keep_scores_idx]

        # NMS 적용
        iou_threshold = 0.5
        keep_nms_idx = nms(bboxes, scores, iou_threshold)
        bboxes = bboxes[keep_nms_idx]
        scores = scores[keep_nms_idx]
        labels = labels[keep_nms_idx]

        # 최종 결과 저장
        selected_bboxes = bboxes
        selected_scores = scores
        selected_labels = labels

# 7. 필터링된 결과를 시각화
if selected_bboxes.numel() > 0:  # .size 대신 .numel() 사용
    for bbox, score, label in zip(selected_bboxes, selected_scores, selected_labels):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 바운딩 박스 그리기
        cv2.putText(image, f'{model.dataset_meta["classes"][label]}: {score:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 클래스와 점수 표시
else:
    print("No objects detected to visualize.")

# 8. 결과 이미지 저장
output_path = 'output/result.jpg'
mmcv.imwrite(image, output_path)

# 9. 결과 확인 (선택적)
plt.imshow(image[:, :, ::-1])  # BGR을 RGB로 변환하여 표시
plt.axis('off')
plt.show()
