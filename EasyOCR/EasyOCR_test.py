import easyocr
reader = easyocr.Reader(['ko','en']) 
result = reader.readtext('./test7.png', detail = 0)   # 이미지 파일 경로
print(result)