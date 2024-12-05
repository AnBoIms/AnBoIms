import easyocr
import json
import re

# EasyOCR Reader 초기화
reader = easyocr.Reader(['ko', 'en'])

def extract_text_with_positions(image_path):
    result = reader.readtext(image_path)
    return result

def merge_close_boxes(boxes, max_y_diff=10):
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
        x_min = min([g[0][0][0] for g in group]),  
        y_min = min([g[0][0][1] for g in group]),  
        x_max = max([g[0][2][0] for g in group]),  
        y_max = max([g[0][2][1] for g in group])   
        rect_coords = [
            [x_min, y_min],  
            [x_max, y_min],  
            [x_max, y_max], 
            [x_min, y_max]   
        ]
        result.append((rect_coords, all_text))
    return result

def process_ocr_results(ocr_results):
    """
    이름, 주민등록번호, 주소를 분류
    """
    resident_id_top_y = None
    address_start_y = None
    first_resident_id_box = None
    name_box = None
    resident_id_boxes = []
    address_boxes = []
    excluded_boxes = []

    # 주민등록증 찾기 (가장 위쪽 텍스트)
    ocr_results = sorted(ocr_results, key=lambda x: x[0][0][1]) 
    closest_distance = float('inf')
    for box in ocr_results:
        coords, text, confidence = box
        text_cleaned = re.sub(r'\s+', '', text) 
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
                first_resident_id_box = box
                resident_id_boxes.append(box)
            else:
              first_box_top_y = first_resident_id_box[0][0][1]
              current_box_top_y = coords[0][1]

              if abs(current_box_top_y - first_box_top_y) > 10:
                excluded_boxes.append(box)  
              else:
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

def save_results_as_json(results, output_path):
    '''
        결과 json 파일로 저장
    '''
    def extract_coordinate(coord):
        """
        좌표를 튜플에서 정수형으로 추출
        """
        if isinstance(coord, tuple):  # 튜플인 경우 내부 값 추출
            return int(coord[0])
        return int(coord)
    
    def adjust_coordinates(coords):
        """
        좌표를 5씩 조정
        """
        adjusted_coords = [
            [coords[0][0] - 5, coords[0][1] - 5], 
            [coords[1][0] + 5, coords[1][1] - 5],  
            [coords[2][0] + 5, coords[2][1] + 5],  
            [coords[3][0] - 5, coords[3][1] + 5]  
        ]
        return adjusted_coords

    annotations = []

    # 이름 데이터 처리
    if results["name"]:
        annotations.append({
            "type": "name",
            "bounding_box": [[extract_coordinate(coord[0]), extract_coordinate(coord[1])] for coord in results["name"][0]]
        })

    # 주민등록번호 데이터 처리
    for box in results["resident_id"]:
        original_coords = [[extract_coordinate(coord[0]), extract_coordinate(coord[1])] for coord in box[0]]
        adjusted_coords = adjust_coordinates(original_coords)
        annotations.append({
            "type": "resident_id",
            "bounding_box": adjusted_coords
        })

    # 주소 데이터 처리 (전체 주소 바운딩 박스를 하나로 합치기)
    if results["address"]:
        all_x = [extract_coordinate(coord[0]) for box in results["address"] for coord in box[0]]
        all_y = [extract_coordinate(coord[1]) for box in results["address"] for coord in box[0]]

        min_x = min(all_x)
        max_x = max(all_x)
        min_y = min(all_y)
        max_y = max(all_y)

        merged_bounding_box = [
            [min_x, min_y],  
            [max_x, min_y],  
            [max_x, max_y],  
            [min_x, max_y]  
        ]

        annotations.append({
            "type": "address",
            "bounding_box": merged_bounding_box
        })

    # JSON 파일 생성
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"annotations": annotations}, f, ensure_ascii=False, indent=2)

    print(f"JSON 파일 저장 완료: {output_path}")




def main():
    # 분석할 이미지 경로
    image_path = 'Image_Inpainting/test_image/test1.png'
    output_json_path = 'Image_Inpainting/idcard_category.json'

    # 텍스트와 위치 정보 추출
    ocr_results = extract_text_with_positions(image_path)

    # 텍스트 박스 분류 및 처리
    results = process_ocr_results(ocr_results)

    # JSON 파일로 저장
    save_results_as_json(results, output_json_path)


if __name__ == "__main__":
    main()