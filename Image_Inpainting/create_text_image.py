from PIL import Image, ImageDraw, ImageFont

def create_text_image(text, font_path, font_size, image_size, output_path):

    image = Image.new("RGBA", image_size, (255, 255, 255, 0))  
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"폰트를 로드할 수 없습니다: {font_path}")
        return

    text_bbox = draw.textbbox((0, 0), text, font=font)  
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    text_position = ((image_size[0] - text_width) // 2, (image_size[1] - text_height) // 2)

    draw.text(text_position, text, font=font, fill=(0, 0, 0, 255)) 

    image.save(output_path, "PNG")
    print(f"이미지 생성 완료: {output_path}")

# 폰트 경로
font_path = "/usr/share/fonts/truetype/nanum/NanumMyeongjoBold.ttf"


# 저장 디렉터리 경로
output_directory = "/home/hyejin/test_ocr/create_image/"  

# 이름 이미지 생성
create_text_image(
    text="안개미",
    font_path=font_path,
    font_size=60,
    image_size=(400, 100),
    output_path=f"{output_directory}name.png"
)

# 주민등록번호 이미지 생성
create_text_image(
    text="123456-4112334",
    font_path=font_path,
    font_size=48,
    image_size=(400, 100),
    output_path=f"{output_directory}number.png"
)
# 주소 이미지 생성
create_text_image(
    text="허리도 가늘군 만지면 부러지리\n2345동 6789호\n(가나동, 다라마바아파트)",
    font_path=font_path,
    font_size=45,
    image_size=(600, 150),
    output_path=f"{output_directory}address.png"
)