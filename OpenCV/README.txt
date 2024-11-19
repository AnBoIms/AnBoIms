# opencv.py
input directory에 image 넣고 opencv.py 수정해서 돌리시면 됩니다

1. box (사용자 지정)
2. polygon (사용자 지정)
3. auto (EasyOCR 돌린 빨간 박스 인식해서 inpainting)
4. file (원본 파일과 마스크 파일 넣으면 inpainting)

-----------------------

수정가능한 parameter

1. inpaintRadius: 값이 클수록 인페인팅 범위 넓어짐 (기본 3)

2. flags: 
* cv2.INPAINT_NS (Navier-Stokes 방식) : 더 부드럽게 작업
* cv2.INPAINT_TELEA INPAINT_NS : 상대적으로 더 뚜렷한 결과

===========================================================

#synthesis.py
input image path랑 outputname 수정해서 run