import cv2
import numpy as np
import json
from shapely.geometry import Polygon
from shapely.affinity import scale
import math

with open("./test/test1.json", "r") as f:
    data = json.load(f)

image_path = './test/test1.jpg'
origin_texture_path = './test/origin.png'

image = cv2.imread(image_path)
origin_texture = cv2.imread(origin_texture_path, cv2.IMREAD_UNCHANGED)

polygon_coords = np.array(data["boxes"][0]["points"], dtype=np.float32)

polygon = Polygon(polygon_coords)
shrinked_polygon = scale(polygon, xfact=0.97, yfact=0.97, origin='center') 
shrinked_coords = np.array(shrinked_polygon.exterior.coords[:-1], dtype=np.float32)

def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def get_rectangle_points(coords):
    coords_sorted_x = sorted(coords, key=lambda x: x[0])  
    coords_sorted_y = sorted(coords, key=lambda x: x[1])  

    x_min = coords_sorted_x[0]
    x_max = coords_sorted_x[-1]
    y_min = coords_sorted_y[0]
    y_max = coords_sorted_y[-1]

    if(distance(x_min, y_min) > distance(x_max, y_max)):
        return [x_min, y_min, x_max,y_max]
    else:
        return [y_min, x_max, y_max, x_min]

rect_points = get_rectangle_points(shrinked_coords)
rect_points_np = np.array(rect_points)

mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.fillPoly(mask, [np.int32(shrinked_coords)], 255)

inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

h, w, _ = origin_texture.shape
texture_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

M = cv2.getPerspectiveTransform(texture_points, rect_points_np)

warped_texture = cv2.warpPerspective(origin_texture, M, (image.shape[1], image.shape[0]))

if warped_texture.shape[2] == 4: 
    alpha_channel = warped_texture[:, :, 3] 
    bgr_texture = warped_texture[:, :, :3]  
else:
    alpha_channel = np.ones_like(warped_texture[:, :, 0]) * 0 
    bgr_texture = warped_texture

alpha_blend_ratio = 0.8
alpha_channel = alpha_channel * alpha_blend_ratio 

masked_texture = cv2.bitwise_and(bgr_texture, bgr_texture, mask=mask)

alpha_channel_masked = alpha_channel.copy()
alpha_channel_masked[mask == 0] = 0  

for c in range(0, 3): 
    inpainted_image[:, :, c] = (inpainted_image[:, :, c] * (255 - alpha_channel_masked) + masked_texture[:, :, c] * alpha_channel_masked) / 255

cv2.imwrite("./test_output/test1_synthesized.jpg", inpainted_image)