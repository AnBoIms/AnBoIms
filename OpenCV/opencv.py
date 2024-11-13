import cv2
import numpy as np
import os

# box masking
def box_mask_drawing(image):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    temp_image = image.copy()
    drawing = False
    ix, iy = -1, -1
    rectangles = [] 

    def draw_rectangle(event, x, y, flags, param):
        nonlocal ix, iy, drawing, temp_image, mask, rectangles
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                temp_image = image.copy() 
                cv2.rectangle(temp_image, (ix, iy), (x, y), (0, 255, 0), 2) 
                cv2.imshow('Image', temp_image)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(mask, (ix, iy), (x, y), 255, -1) 
            rectangles.append((ix, iy, x, y)) 
            temp_image = image.copy() 
            for rect in rectangles:
                cv2.rectangle(temp_image, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 2)
            cv2.imshow('Image', temp_image)

    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', draw_rectangle)
    
    while True:
        cv2.imshow('Image', temp_image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13: #Enter
            break
    
    cv2.destroyAllWindows()
    return mask

# read border masking(auto)
def auto_detect_red_boxes(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask_red1, mask_red2)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask

# polygon masking
def polygon_mask_drawing(image):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    temp_image = image.copy()
    points = []

    def draw_polygon(event, x, y, flags, param):
        nonlocal points, temp_image, mask
        
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y)) 
            temp_image = image.copy()
            if len(points) > 1:
                cv2.polylines(temp_image, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=2)
            cv2.imshow('Image', temp_image)
        
        elif event == cv2.EVENT_RBUTTONDOWN:  # right click to close polygon
            if len(points) >= 3: 
                cv2.fillPoly(mask, [np.array(points)], color=255) 
                cv2.polylines(temp_image, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
                cv2.imshow('Image', temp_image)
                points = [] 
    
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', draw_polygon)
    
    while True:
        cv2.imshow('Image', temp_image)
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter
            break
    
    cv2.destroyAllWindows()
    return mask

#file masking
def file_masking(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) 

    if image is None or mask is None:
        print("no image or mask")
        return

    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_folder = "output/file_masking"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_path = os.path.join(output_folder, f"{base_name}_file_masking_inpainted.png")
    
    # save result
    cv2.imwrite(output_path, inpainted_image)

# inpainting
def inpaint_image(image_path, method='box', mask_path=None):
    image = cv2.imread(image_path)

    if image is None:
        print("no image.")
        return

    if method == 'box':
        mask = box_mask_drawing(image)
    elif method == 'auto':
        mask = auto_detect_red_boxes(image)
    elif method == 'polygon':
        mask = polygon_mask_drawing(image)
    elif method == 'file' and mask_path:
        file_masking(image_path, mask_path)  
        return
    else:
        return

    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_folder = f"output/{method}"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_path = os.path.join(output_folder, f"{base_name}_{method}_inpainted.png")
    
    # save result
    cv2.imwrite(output_path, inpainted_image)

# example
image_path = 'input/original.png'
mask_path = 'input/original_mask.jpg' 

inpaint_image(image_path, method='file', mask_path=mask_path)
# inpaint_image(image_path, method='box')
# inpaint_image(image_path, method='auto')
# inpaint_image(image_path, method='poligon')