import numpy as np
import cv2 
import matplotlib.pyplot as plt
import os

## 파일 경로를 상대경로를 활용해서 더 쉽게 접근이 가능하게 수정하였습니다. - 한영욱
# 현재 파일이 실행되는 디렉토리 경로를 가져오기
current_dir = os.path.dirname(os.path.abspath(__file__))

# 이미지 파일 경로 설정 (현재 디렉토리 기준)
image_path = os.path.join(current_dir, 'test1.png')

img = cv2.imread(image_path,0)

ret, thresh1 = cv2.threshold(img,127,255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img,127,255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img,127,255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img,127,255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img,127,255, cv2.THRESH_TOZERO_INV)

# thresh6 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
# thresh7 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

titles =['Original','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img,thresh1,thresh2,thresh3,thresh4,thresh5]

for i in range(6):
  # plt.figure(figsize=(10,8))
  plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
  plt.title(titles[i])
  plt.xticks([]),plt.yticks([])
plt.show()