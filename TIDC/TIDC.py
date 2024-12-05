import os
from matplotlib.pyplot import imshow
import numpy as np
import random
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize


def textIpaintingDatasetsCreate(
  input_text_file,
  output_text_file,
  font_dir,
  standard_font_file,
  color_file,
  background_dir,
  output_path,
  # num_samples,
  img_width,
  img_height,
  text_size,
  # text_size_min,
  # text_size_max,
  start_num,
  gpu
  ):

  # Use gpu
  if(gpu != "cpu"):
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=gpu
  
  # Make dir
  os.makedirs(output_path, exist_ok=True)
  i_s_path = os.path.join(output_path, 'i_s')
  os.makedirs(i_s_path, exist_ok=True)
  i_t_path = os.path.join(output_path, 'i_t')
  os.makedirs(i_t_path, exist_ok=True)
  mask_t_path = os.path.join(output_path, 'mask_t')
  os.makedirs(mask_t_path, exist_ok=True)
  t_sk_path = os.path.join(output_path, 't_sk')
  os.makedirs(t_sk_path, exist_ok=True)
  t_t_path = os.path.join(output_path, 't_t')
  os.makedirs(t_t_path, exist_ok=True)
  t_b_path = os.path.join(output_path, 't_b')
  os.makedirs(t_b_path, exist_ok=True)
  t_f_path = os.path.join(output_path, 't_f')
  os.makedirs(t_f_path, exist_ok=True)
  
  fonts = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if f.endswith(".ttf")]
  backgrounds = [os.path.join(background_dir, f) for f in os.listdir(background_dir) if f.endswith(("jpeg", ".jpg", ".png"))]
  with open(color_file, 'r', encoding='utf-8') as cf:
    colors = [color_line.strip() for color_line in cf.readlines()]
  
  n = start_num
  for background in backgrounds:
    t_b_image = create_t_b(background, img_width, img_height)
    for font_file in fonts:
      for color in colors:
        for matched_input, matched_output in matchText(input_text_file, output_text_file):
          # for i in range(num_samples):
          t_b_image.save(os.path.join(t_b_path, f"{n}_t_b.png")) # Save t_b_image
          """
          text_size = random.randint(text_size_min, text_size_max) # Text size random
          font = ImageFont.truetype(font_file, text_size)
          text_location = textLocation(t_b_image, matched_input, matched_output, font, img_width, img_height)
          """
          i_s_font, i_s_text_location = textSet(matched_input, font_file, text_size, img_width, img_height)
          i_t_font, i_t_text_location = textSet(matched_output, standard_font_file, text_size, img_width, img_height)
          font, text_location = textSet(matched_output, font_file, text_size, img_width, img_height)
          create_i_s(matched_input, i_s_font, color, t_b_image, i_s_path, i_s_text_location, n)
          create_i_t(matched_output, ImageFont.truetype(standard_font_file, text_size), i_t_path, img_width, img_height, i_t_text_location, n)
          create_mask_t_t_sk(matched_output, font, mask_t_path, t_sk_path, img_width, img_height, text_location, n)
          create_t_t(matched_output, font, color, t_t_path, img_width, img_height, text_location, n)
          create_t_f(matched_output, font, color, t_b_image, t_f_path, text_location, n)
          n += 1


def matchText(input_text_file, output_text_file):
  with open(input_text_file, 'r', encoding='utf-8') as input_file:
    input_lines = [line.strip() for line in input_file.readlines()]
  with open(output_text_file, 'r', encoding='utf-8') as output_file:
    output_lines = [line.strip() for line in output_file.readlines()]
  # Matching lines
  matches = [(input_line, output_line) for input_line in input_lines for output_line in output_lines]
  return matches


def textSet(text, font_file, text_size, img_width, img_height):
  # Dynamically calculate text size to fit image
  font = ImageFont.truetype(font_file, text_size)
  img = Image.new('RGB', (img_width, img_height), color='white')
  draw = ImageDraw.Draw(img)
  # Use getbbox instead of getsize to get the text dimensions
  text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:] 

  while text_width > img_width * 1.0 and text_height > img_height * 1.0:  # 여백 0%
    text_size -= 1
  font = ImageFont.truetype(font_file, text_size)
  text_width, text_height = draw.textbbox((0, 0), text, font=font)[2:]
  # Calculate text position (center alignment)
  text_location = ((img_width - text_width) // 2, (img_height - text_height) // 2)

  return font, text_location


"""
def textLocation(background, text1, text2, font, img_width, img_height):
  draw = ImageDraw.Draw(background)
  text1_bbox = draw.textbbox((0, 0), text1, font=font)  # Get the bounding box of the text
  text1_width = text1_bbox[2] - text1_bbox[0]  # Calculate width
  text1_height = text1_bbox[3] - text1_bbox[1]  # Calculate height
  text2_bbox = draw.textbbox((0, 0), text2, font=font)
  text2_width = text2_bbox[2] - text2_bbox[0]
  text2_height = text2_bbox[3] - text2_bbox[1]

  text_width = text1_width if (text1_width > text2_width) else text2_width
  text_height = text1_height if (text1_height > text2_height) else text2_height
  x = random.randint(0, (img_width - text_width) if (img_width > text_width) else img_width)
  y = random.randint(0, (img_height - text_height) if (img_height > text_height) else img_height)
  return (x, y)
"""


def create_t_b(background, img_width, img_height):
  # Load background
  background = Image.open(background).convert('RGB')
  image_width, image_height = background.size

  # Adjust image proportions
  if ((image_width / img_width) < (image_height / img_height)):
    left = 0
    top = (image_height - (image_width / img_width * img_height)) // 2
    right = image_width
    bottom = top + (image_width / img_width * img_height)
  else:
    left = (image_width - (image_height / img_height * img_width)) // 2
    top = 0
    right = left + (image_height / img_height * img_width)
    bottom = image_height

  t_b_image = background.crop((left, top, right, bottom)) # Image crop
  t_b_image = t_b_image.resize((img_width, img_height)) # Image resize
  return t_b_image


def create_i_s(text, font, color, background, i_s_path, text_location, n):
  i_s_image = background.copy()
  draw = ImageDraw.Draw(i_s_image)
  draw.text(text_location, text, fill=color, font=font) # Add text
  # Save image
  i_s_image.save(os.path.join(i_s_path, f"{n}_i_s.png"))


def create_i_t(text, standard_font, i_t_path, img_width, img_height, text_location, n):
  # Create gray background
  i_t_image = Image.new('RGB', (img_width, img_height), color='gray')
  draw = ImageDraw.Draw(i_t_image)
  draw.text(text_location, text, fill='black', font=standard_font) # Add text
  # Save image
  i_t_image.save(os.path.join(i_t_path, f"{n}_i_t.png"))


def create_mask_t_t_sk(text, font, mask_t_path, t_sk_path, img_width, img_height, text_location, n):
  # Create black background
  mask_t_image = Image.new('RGB', (img_width, img_height), color='black')
  draw = ImageDraw.Draw(mask_t_image)
  draw.text(text_location, text, fill='white', font=font) # Add text
  # Save image
  mask_t_image.save(os.path.join(mask_t_path, f"{n}_mask_t.png"))
  # Convert cropped_image to grayscale
  grayscale_image = mask_t_image.convert('L')
  skeleton_image = skeletonize(np.array(grayscale_image))
  # Convert the skeleton image back to a PIL Image
  t_sk_image = Image.fromarray(skeleton_image.astype('uint8') * 255)
  t_sk_image.save(os.path.join(t_sk_path, f"{n}_t_sk.png"))


def create_t_t(text, font, color, t_t_path, img_width, img_height, text_location, n):
  # Create gray background
  t_t_image = Image.new('RGB', (img_width, img_height), color='gray')
  draw = ImageDraw.Draw(t_t_image)
  draw.text(text_location, text, fill=color, font=font) # Add text
  # Save image
  t_t_image.save(os.path.join(t_t_path, f"{n}_t_t.png"))



def create_t_f(text, font, color, background, t_f_path, text_location, n):
  t_f_image = background.copy()
  draw = ImageDraw.Draw(t_f_image)
  draw.text(text_location, text, fill=color, font=font) # Add text
  # Save image
  t_f_image.save(os.path.join(t_f_path, f"{n}_t_f.png"))