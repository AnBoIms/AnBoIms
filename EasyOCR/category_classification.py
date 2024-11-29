import easyocr
import re

# EasyOCR Reader 초기화
reader = easyocr.Reader(['ko', 'en'])

def extract_text_with_positions(image_path):
    # 이미지에서 텍스트와 위치 정보 추출
    result = reader.readtext(image_path)
    return result

def clean_name_text(text):
    """
    한글만 남기고 텍스트를 정제하는 함수
    """
    cleaned_text = re.sub(r'[^가-힣\s]', '', text)  # 한글과 공백만 남김
    return cleaned_text.strip()


def combine_text_by_position(result):
    name = ""
    id_number = ""
    address = ""

    # Y 좌표를 기준으로 정렬 (위에서 아래로)
    result = sorted(result, key=lambda x: (x[0][0][1], x[0][0][0]))  # Y, X 순서로 정렬

    # 주민등록번호 관련 텍스트 병합
    id_card_bottom_y = None
    resident_id_pattern = re.compile(r'\d{6}-?\d{7}')



    for item in result:
        box, text, _ = item
        top_left, _, bottom_right, _ = box
        y_position = (top_left[1] + bottom_right[1]) / 2  # 중심 Y

        # 주민등록증 영역 감지
        if re.search(r'[주민등록증]+', text):
            if id_card_bottom_y is None:
                id_card_bottom_y = bottom_right[1]  # 주민등록증 하단 Y 좌표
            else:
                id_card_bottom_y = max(id_card_bottom_y, bottom_right[1])
            continue

        # 이름 추출: 주민등록증 아래, 주민등록번호 위
        if id_card_bottom_y and y_position > id_card_bottom_y and not id_number:
            cleaned_text = clean_name_text(text)
            if re.match(r'^[가-힣\s]{2,}$', cleaned_text):  # 한글만 포함
                name = text.strip()

        # 주민등록번호 추출
        if resident_id_pattern.search(text):
            id_number = resident_id_pattern.search(text).group()
            id_number_y = y_position

        # 주소 병합
        if id_number and y_position > id_number_y:
            address += text.strip() + " "

    # 주소에서 불필요한 공백 제거
    address = address.strip()

    return name, id_number, address

def main():
    # 분석할 이미지 경로
    image_path = '/home/hyejin/test_ocr/test_image/test101.png'

    # 텍스트와 위치 정보 추출
    result = extract_text_with_positions(image_path)
    print("OCR Result with Positions:", result)

    # 이름, 주민등록번호, 주소 추출 및 병합
    name, id_number, address = combine_text_by_position(result)

    print(f"이름: {name}")
    print(f"주민등록번호: {id_number}")
    print(f"주소: {address}")

if __name__ == "__main__":
    main()
