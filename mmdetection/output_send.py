from mmdet.apis import init_detector, inference_detector
import cv2
import numpy as np
import os

# 1. 모델 설정 파일과 체크포인트 파일 경로
config_file = 'configs/anboims_dataset/mask-rcnn_r50-caffe_fpn_ms-poly-1x_anboimsdataset.py'  # Mask R-CNN 설정 파일
checkpoint_file = 'work_dirs/mask-rcnn_r50-caffe_fpn_ms-poly-1x_anboimsdataset/epoch_12.pth'  # 학습된 모델 체크포인트 파일

# 2. 모델 초기화
model = init_detector(config_file, checkpoint_file, device='cpu')  # GPU 사용 (CUDA)

# 3. 테스트 이미지 및 출력 경로 설정
test_dir = 'tools/data/anboims_dataset_1000/test'  # 테스트 이미지 폴더
output_dir = 'output_things/images'  # 결과 이미지 저장 폴더
coords_dir = 'output_things/coordinate'  # 좌표 저장 폴더
rect_dir = 'output_things/rectangles'  # 추출된 사각형 이미지 저장 폴더
os.makedirs(output_dir, exist_ok=True)
os.makedirs(coords_dir, exist_ok=True)
os.makedirs(rect_dir, exist_ok=True)

# 4. Convex Hull 좌표 추출 함수
def get_coordinate(segmentation_points):
    """마스크에서 Convex Hull을 계산하고 외곽 좌표 4개를 반환"""
    points = np.array(segmentation_points, dtype=np.float32)
    hull = cv2.convexHull(points)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)
    return np.int0(box).tolist()


# 5. 모델 추론 및 결과 저장
for img_name in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_name)
    output_img_path = os.path.join(output_dir, img_name.replace('.jpg', '.png'))  # 출력 이미지 경로
    output_txt_path = os.path.join(coords_dir, img_name.replace('.jpg', '.txt'))  # 좌표 저장 경로
    rect_img_path = os.path.join(rect_dir, img_name.replace('.jpg', '_rect.png'))  # 사각형 이미지 경로

    # 6. 모델 추론
    result = inference_detector(model, img_path)
    img = cv2.imread(img_path)
    
    # 7. 예측 결과에서 마스크 정보 추출
    masks = result.pred_instances.masks.cpu().numpy()  # 마스크 정보 추출
    coordinates_list = []  # 좌표 저장 리스트

    for mask in masks:
        if mask.sum() > 0:  # 유효한 마스크만 처리
            segmentation_points = np.argwhere(mask > 0)
            segmentation_points = [(x, y) for y, x in segmentation_points]

            hull_points = get_coordinate(segmentation_points)  # Convex Hull 계산
            for point in hull_points:
                cv2.circle(img, tuple(point), 5, (0, 255, 0), -1)  # 초록색 점 그리기
            
            # 사각형 내부 추출
            rect_mask = np.zeros_like(img, dtype=np.uint8)
            cv2.drawContours(rect_mask, [np.array(hull_points)], -1, (255, 255, 255), -1)
            rect_img = cv2.bitwise_and(img, rect_mask)

            # 추출된 사각형 이미지 저장
            cv2.imwrite(rect_img_path, rect_img)
            print(f"Rectangle extracted -> Saved to {rect_img_path}")

            coordinates_list.append(hull_points)  # 좌표 저장

    # 8. 이미지 저장
    cv2.imwrite(output_img_path, img)
    print(f"Processed {img_name} -> Saved to {output_img_path}")

    # 9. 좌표를 txt 파일로 저장
    with open(output_txt_path, 'w') as f:
        for coords in coordinates_list:
            f.write(','.join([f"{x[0]} {x[1]}" for x in coords]) + '\n')  # 좌표를 텍스트 형식으로 저장
    print(f"Coordinates saved to {output_txt_path}")
