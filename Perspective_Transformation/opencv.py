import numpy as np
import cv2
import matplotlib.pyplot as plt

# 이미지 읽기
img = cv2.imread('./img/test4.jpg')
h, w = img.shape[:2]

if img is None:
    print("이미지를 찾을 수 없습니다. 경로를 확인하세요.")
    exit()


# 원본 좌표와 변환 좌표 정의
pts1 = np.float32( [
                [
                    1954.8125861205585,
                    2089.9253168028067
                ],
                [
                    2527.879000943951,
                    3005.507606036368
                ],
                [
                    3024,
                    2706.891766575731
                ],
                [
                    3024,
                    1456.1676149746957
                ]
            ]
)
pts2 = np.float32([[0, 0], [0, 540],[860, 540], [860, 0]])  # 변환 후 원하는 좌표

print(f"이미지 크기: {w} x {h}")
for point in pts1:
    if not (0 <= point[0] < w and 0 <= point[1] < h):
        print(f"좌표 {point}가 이미지 범위를 벗어났습니다.")

# # 원본 좌표에 원 그리기 (시각적 확인용)
# cv2.circle(img, (1384, 2000), 10, (255, 0, 0), -1)  # 파란색 원
# cv2.circle(img, (1575, 2410), 10, (0, 255, 0), -1)  # 녹색 원
# cv2.circle(img, (1977, 1948), 10, (0, 0, 255), -1)  # 빨간색 원
# cv2.circle(img, (1845, 1559), 10, (0, 255, 255), -1)  # 노란색 원

# 투시 변환 매트릭스 계산 및 변환 적용
M = cv2.getPerspectiveTransform(pts1, pts2)

output_width = int(max(pts2[:, 0]) - min(pts2[:, 0]))
output_height = int(max(pts2[:, 1]) - min(pts2[:, 1]))
dst = cv2.warpPerspective(img, M, (output_width, output_height))


# 원본 및 변환 결과 표시
plt.figure(figsize=(12, 6))  # 그래프 크기 조정
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # OpenCV 이미지를 RGB로 변환
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))  # 변환된 이미지 출력
plt.title('Perspective Transformed')

plt.show()