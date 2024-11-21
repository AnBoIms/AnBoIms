import os
from matplotlib.pyplot import imshow
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.morphology import skeletonize


def textIpaintingDatasetsCreate(
  input_text_file,
  output_text_file,
  font_file,
  standard_font_file,
  color,
  background_path,
  orientation,
  output_path,
  num_samples,
  image_size,
  start_num
  ):
  os.makedirs(output_path, exist_ok=True) # Make dir
  matched_input, matched_output = matchText(input_text_file, output_text_file)
  creat_i_s(matched_input, font_file, color, background_path, orientation, output_path, image_size, start_num)
  creat_i_t(matched_output, standard_font_file, orientation, output_path, image_size, start_num)
  creat_mask_t_t_sk(matched_output, font_file, orientation, output_path, image_size, start_num)
  creat_t_t(matched_output, font_file, color, orientation, output_path, image_size, start_num)
  creat_t_b(matched_output, font_file, background_path, orientation, output_path, image_size, start_num)
  creat_t_f(matched_output, font_file, color, background_path, orientation, output_path, image_size, start_num)


def matchText(input_text_file, output_text_file):
  with open(input_text_file, 'r', encoding='utf-8') as input_file:
    input_lines = [line.strip() for line in input_file.readlines()]
  with open(output_text_file, 'r', encoding='utf-8') as output_file:
    output_lines = [line.strip() for line in output_file.readlines()]
  # Matching lines
  matches = [(input_line, output_line) for input_line in input_lines for output_line in output_lines]
  # Extraction
  matched_input = [match[0] for match in matches]
  matched_output = [match[1] for match in matches]
  return matched_input, matched_output


def creat_i_s(texts, font_file, color, background_path, orientation, output_path, image_size, start_num):
  i_s_path = os.path.join(output_path, 'i_s')
  os.makedirs(i_s_path, exist_ok=True)
  i = start_num
  for text in texts:
    # Load background and resize
    background = Image.open(background_path).convert('RGB')
    background = background.resize(image_size)
    i_s_image = drawText(text, font_file, color, background, orientation, image_size)
    # Save image
    i_s_image.save(os.path.join(i_s_path, f"{i}_i_s.png"))
    i += 1


def creat_i_t(texts, standard_font_file, orientation, output_path, image_size, start_num):
  i_t_path = os.path.join(output_path, 'i_t')
  os.makedirs(i_t_path, exist_ok=True)
  i = start_num
  for text in texts:
    # Create gray background
    background = Image.new('RGB', image_size, color='gray')
    i_t_image = drawText(text, standard_font_file, 'black', background, orientation, image_size)
    # Save image
    i_t_image.save(os.path.join(i_t_path, f"{i}_i_t.png"))
    i += 1


def creat_mask_t_t_sk(texts, font_file, orientation, output_path, image_size, start_num):
  mask_t_path = os.path.join(output_path, 'mask_t')
  os.makedirs(mask_t_path, exist_ok=True)
  t_sk_path = os.path.join(output_path, 't_sk')
  os.makedirs(t_sk_path, exist_ok=True)
  i = start_num
  for text in texts:
    # Create black background
    background = Image.new('RGB', image_size, color='black')
    mask_t_image = drawText(text, font_file, 'white', background, orientation, image_size)
    # Save image
    mask_t_image.save(os.path.join(mask_t_path, f"{i}_mask_t.png"))
    # Convert cropped_image to grayscale
    grayscale_image = mask_t_image.convert('L')
    skeleton_image = skeletonize(np.array(grayscale_image))
    # Convert the skeleton image back to a PIL Image
    t_sk_image = Image.fromarray(skeleton_image.astype('uint8') * 255)
    t_sk_image.save(os.path.join(t_sk_path, f"{i}_t_sk.png"))
    i += 1


def creat_t_t(texts, font_file, color, orientation, output_path, image_size, start_num):
  t_t_path = os.path.join(output_path, 't_t')
  os.makedirs(t_t_path, exist_ok=True)
  i = start_num
  for text in texts:
    # Create gray background
    background = Image.new('RGB', image_size, color='gray')
    t_t_image = drawText(text, font_file, color, background, orientation, image_size)
    # Save image
    t_t_image.save(os.path.join(t_t_path, f"{i}_t_t.png"))
    i += 1


def creat_t_b(texts, font_file, background_path, orientation, output_path, image_size, start_num):
  t_b_path = os.path.join(output_path, 't_b')
  os.makedirs(t_b_path, exist_ok=True)
  i = start_num
  for text in texts:
    # Load background and resize
    background = Image.open(background_path).convert('RGB')
    background = background.resize(image_size)
    t_b_image = drawText(text, font_file, 'black', background, orientation, image_size, True)
    # Save image
    t_b_image.save(os.path.join(t_b_path, f"{i}_t_b.png"))
    i += 1


def creat_t_f(texts, font_file, color, background_path, orientation, output_path, image_size, start_num):
  t_f_path = os.path.join(output_path, 't_f')
  os.makedirs(t_f_path, exist_ok=True)
  i = start_num
  for text in texts:
    # Load background and resize
    background = Image.open(background_path).convert('RGB')
    background = background.resize(image_size)
    t_f_image = drawText(text, font_file, color, background, orientation, image_size)
    # Save image
    t_f_image.save(os.path.join(t_f_path, f"{i}_t_f.png"))
    i += 1


def drawText(text, font_file, color, background, orientation, image_size, t_b=False):
  # t_b
  bg = background.copy()
  # Font and textsize
  try:
      font = ImageFont.truetype(font_file, 50)
  except:
      font = ImageFont.load_default()

  # Draw text in image
  draw = ImageDraw.Draw(background)
  if orientation == 'horizontal':
      # Use textbbox to get the bounding box of the text
      text_bbox = draw.textbbox((0, 0), text, font=font)
      text_width = text_bbox[2] - text_bbox[0]  # Calculate width
      text_height = text_bbox[3] - text_bbox[1]  # Calculate height

      width, height = image_size[0], image_size[1]
      text_x = max((width - text_width) / 2, 0)
      text_y = max((height - text_height) / 2, 0)
      draw.text((text_x, text_y), text, fill=color, font=font)
      bbox = draw.textbbox((text_x, text_y), text, font=font)
  elif orientation == 'vertical':
      lines = list(text)
      line_height = font.getbbox("A")[3] - font.getbbox("A")[1]
      max_line_width = max([font.getbbox(line)[2] - font.getbbox(line)[0] for line in lines])
      total_text_height = len(lines) * line_height
      if total_text_height > height:
          height = total_text_height + 20  # Add free space
          background = background.new('RGB', (width, height), color='black')
          draw = ImageDraw.Draw(background)
      text_x = (width - max_line_width) / 2
      text_y = (height - total_text_height) / 2
      for i, line in enumerate(lines):
          draw.text((text_x, text_y + i * line_height), line, fill=color, font=font)
      # Crop to fit the widest text width
      bbox = (text_x, text_y, text_x + max_line_width, text_y + total_text_height)

  if t_b: # if t_b
    # Cut only the area containing text (cut to fit text both horizontally and vertically)
    if bbox:
        left, upper, right, lower = bbox
        # Additional margin adjustments to cut the horizontal length shorter
        padding = 10
        cropped_image = bg.crop((max(left - padding, 0), max(upper - padding, 0), min(right + padding, width), min(lower + padding, height)))
    else:
        cropped_image = bg
  else: # not t_b
    # Cut only the area containing text (cut to fit text both horizontally and vertically)
    if bbox:
        left, upper, right, lower = bbox
        # Additional margin adjustments to cut the horizontal length shorter
        padding = 10
        cropped_image = background.crop((max(left - padding, 0), max(upper - padding, 0), min(right + padding, width), min(lower + padding, height)))
    else:
        cropped_image = background
  return cropped_image