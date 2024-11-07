import os
import numpy as np
import cv2
# 실행하고 파일을 저장하도록 설정
# 현재 파일이 실행되는 디렉토리 경로를 가져오기
current_dir = os.path.dirname(os.path.abspath(__file__))
img_name = 'test2'
# 이미지 파일 경로 설정 (현재 디렉토리 기준)
image_path = os.path.join(current_dir, img_name+'.png')
original_img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# 이미지 읽기
img = cv2.imread(image_path, 0)

# 이미지가 정상적으로 읽혔는지 확인
if img is None:
    print("이미지를 찾을 수 없습니다. 경로를 확인해 주세요:", image_path)
else:
    # 다양한 Threshold 적용
    ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    thresh6 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh7 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 제목과 이미지 리스트
    titles = ['BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV', 'ADAPTIVE_MEAN', 'ADAPTIVE_GAUSSIAN']
    images = [thresh1, thresh2, thresh3, thresh4, thresh5, thresh6, thresh7]

    # 이미지 저장 경로 설정 및 저장
    output_dir = os.path.join(current_dir, 'output_images/'+img_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(len(images)):
        output_path = os.path.join(output_dir, f'{titles[i]}'+img_name+'.jpg')
        cv2.imwrite(output_path, images[i])
        print(f'{titles[i]} 이미지가 저장되었습니다: {output_path}')
        
        

# 원본 이미지의 복사본 생성
edited_img = original_img.copy()

# 빨간색으로 변경할 부분만 마스크로 선택
mask = thresh7 != 255

# 마스크된 흰색 영역을 빨간색으로 변경
edited_img[mask] = [0, 0, 255]

# 결과 이미지 보기
cv2.imshow('Red Text Areas', edited_img)
cv2.waitKey(0)
cv2.destroyAllWindows()