import easyocr
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image


reader = easyocr.Reader(['ko', 'en'])
result = reader.readtext('./test7.png')
img    = cv2.imread('./test7.png')
img = Image.fromarray(img)
font = ImageFont.truetype("/usr/share/fonts/truetype/nanum/NanumGothic.ttf", 25)
draw = ImageDraw.Draw(img)
np.random.seed(42)
for i in result :
    x = i[0][0][0] 
    y = i[0][0][1] 
    w = i[0][1][0] - i[0][0][0] 
    h = i[0][2][1] - i[0][1][1]
 
    color = (255, 0, 0)

    draw.rectangle(((x, y), (x+w, y+h)), outline=tuple(color), width=2)
    draw.text((int((x + x + w) / 2) , y-2),str(i[1]), font=font, fill=tuple(color),)

plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()