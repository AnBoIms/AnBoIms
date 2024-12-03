import easyocr
import re

# EasyOCR Reader 초기화
reader = easyocr.Reader(['ko', 'en'])

def extract_text_with_positions(image_path):
    result = reader.readtext(image_path)
    return result

def merge_close_boxes(boxes, max_y_diff=15):
    """
    Y 좌표가 가까운 텍스트 박스들을 병합
    """
    merged_text = []
    temp_box = []
    boxes = sorted(boxes, key=lambda x: x[0][1])  # Y 좌표 기준으로 정렬

    for box in boxes:
        coords, text, confidence = box
        y_center = (coords[0][1] + coords[2][1]) / 2

        if not temp_box:
            temp_box.append((coords, text))
            continue

        prev_coords = temp_box[-1][0]
        prev_y_center = (prev_coords[0][1] + prev_coords[2][1]) / 2

        # Y 좌표 차이가 max_y_diff 이하인 경우 병합
        if abs(y_center - prev_y_center) <= max_y_diff:
            temp_box.append((coords, text))
        else:
            # 병합된 박스 처리
            merged_text.append(temp_box)
            temp_box = [(coords, text)]

    # 마지막 그룹 추가
    if temp_box:
        merged_text.append(temp_box)

    # 병합 결과
    result = []
    for group in merged_text:
        all_text = " ".join([t[1] for t in group])
        all_coords = [
            min([g[0][0][0] for g in group]),  # 좌상단 X
            min([g[0][0][1] for g in group]),  # 좌상단 Y
            max([g[0][2][0] for g in group]),  # 우하단 X
            max([g[0][2][1] for g in group])   # 우하단 Y
        ]
        result.append((all_coords, all_text))
    return result

def process_ocr_results(ocr_results):
    """
    OCR 결과를 처리하여 이름, 주민등록번호, 주소를 분류
    """
    resident_id_top_y = None
    address_start_y = None
    first_resident_id_box = None
    name_box = None
    resident_id_boxes = []
    address_boxes = []
    excluded_boxes = []

    # 주민등록증 찾기 (가장 위쪽 텍스트)
    ocr_results = sorted(ocr_results, key=lambda x: x[0][0][1])  # Y 좌표로 정렬
    closest_distance = float('inf')
    for box in ocr_results:
        coords, text, confidence = box
        text_cleaned = re.sub(r'\s+', '', text)  # 공백 제거
        if "주민등록증" in text_cleaned:
            resident_id_top_y = coords[2][1]  # 주민등록증 하단 Y 좌표
            break

    # 이름, 주민등록번호, 주소 분류
    for box in ocr_results:
        coords, text, confidence = box
        text_cleaned = text.strip()

        # 주민등록증 위 텍스트는 제외
        if resident_id_top_y and coords[0][1] <= resident_id_top_y:
            excluded_boxes.append(box)
            continue
        
        # 이름 추출
        if resident_id_top_y and coords[0][1] > resident_id_top_y:
            distance = coords[0][1] - resident_id_top_y
            if distance < closest_distance:
                name_box = (coords, text_cleaned)
                closest_distance = distance
             
        
        # 주민등록번호 추출 (숫자와 '-' 포함)
        if re.match(r'^[0-9\-]+$', text_cleaned):
            if not first_resident_id_box:
                # 첫 번째 주민등록번호 박스 저장
                first_resident_id_box = box
                resident_id_boxes.append(box)
            else:
                # 주민등록번호 추가
                resident_id_boxes.append(box)
            address_start_y = max(address_start_y or 0, coords[2][1])
            continue

        # 발급 날짜 및 구청장 텍스트 분류
        if re.match(r'^[0-9\s.]+$', text_cleaned) or "구청장" in text_cleaned or "시장" in text_cleaned:
            excluded_boxes.append(box)
            continue

        # 주소 추출 (주민등록번호 아래 텍스트)
        if address_start_y and coords[0][1] > address_start_y:
            address_boxes.append(box)
            continue

        if name_box and box[0] == name_box[0]:
            continue


        # 제외 대상
        excluded_boxes.append(box)

    # 주민등록번호 병합
    merged_resident_id = merge_close_boxes(resident_id_boxes)

    # 주소 병합
    merged_address = merge_close_boxes(address_boxes)

    return {
        "name": name_box,
        "resident_id": merged_resident_id,
        "address": merged_address,
        "excluded": excluded_boxes
    }


def main():
    # 분석할 이미지 경로
    image_path = '/home/hyejin/test_ocr/test_image/test101.png'

    # 텍스트와 위치 정보 추출
    ocr_results = extract_text_with_positions(image_path)

    # 텍스트 박스 분류 및 처리
    results = process_ocr_results(ocr_results)

    # 결과 출력
    print("\n[이름]")
    if results["name"]:
        print(f"텍스트: {results['name'][1]}, 바운딩 박스: {results['name'][0]}")

    print("\n[주민등록번호]")
    for box in results["resident_id"]:
        print(f"텍스트: {box[1]}, 바운딩 박스: {box[0]}")

    print("\n[주소]")
    for box in results["address"]:
        print(f"텍스트: {box[1]}, 바운딩 박스: {box[0]}")

    print("\n[제외된 텍스트]")
    for box in results["excluded"]:
        print(f"텍스트: {box[1]}, 바운딩 박스: {box[0]}")

if __name__ == "__main__":
    main()
