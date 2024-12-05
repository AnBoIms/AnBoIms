import cv2
import numpy as np
import json

# JSON 파일과 이미지 경로
with open("Image_Inpainting/idcard_category.json", "r") as f:
    data = json.load(f)

image_path = 'Image_Inpainting/test_image/test1.png'
textures = {
    "name": 'Image_Inpainting/inpainting_image/name.png',
    "resident_id": 'Image_Inpainting/inpainting_image/number.png',
    "address": 'Image_Inpainting/inpainting_image/address.png'
}

# 이미지 로드
image = cv2.imread(image_path)

# 텍스처 합성과 인페인팅 함수
def process_bounding_box(image, texture_path, bounding_box):
    texture = cv2.imread(texture_path, cv2.IMREAD_UNCHANGED)
    h, w, _ = texture.shape

    # 바운딩 박스를 NumPy 배열로 변환
    bounding_box = np.array(bounding_box, dtype=np.float32)

    # 텍스처 좌표 설정
    texture_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # 투영 변환 행렬 계산
    M = cv2.getPerspectiveTransform(texture_points, bounding_box)

    # 텍스처 변환
    warped_texture = cv2.warpPerspective(texture, M, (image.shape[1], image.shape[0]))

    # 마스크 생성
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [np.int32(bounding_box)], 255)

    # Inpainting
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # 투명 배경 텍스처 처리
    if warped_texture.shape[2] == 4:  # RGBA
        alpha_channel = warped_texture[:, :, 3]
        bgr_texture = warped_texture[:, :, :3]
    else:
        alpha_channel = np.ones_like(warped_texture[:, :, 0]) * 255
        bgr_texture = warped_texture

    # 합성
    alpha_channel = alpha_channel / 255.0  # Normalize alpha channel to [0, 1]
    for c in range(0, 3):
        inpainted_image[:, :, c] = (
            inpainted_image[:, :, c] * (1 - alpha_channel) +
            bgr_texture[:, :, c] * alpha_channel
        )

    return inpainted_image

# 각 텍스트 타입별로 처리
for annotation in data["annotations"]:
    text_type = annotation["type"]
    bounding_box = annotation["bounding_box"]
    texture_path = textures.get(text_type)
    
    if texture_path:
        image = process_bounding_box(image, texture_path, bounding_box)

# 결과 저장
cv2.imwrite("Image_Inpainting/output_image/synthesized.jpg", image)
