import random
import cv2 as cv
import cupy as cp
from os import scandir
from PIL import Image, ImageDraw, ImageFont

def digit_to_image(str_digit):
    folder = scandir("./Generate_image_digits/.fonts")
    font_names = (file.name for file in folder if file.is_file())
    font_name = random.choice(list(font_names))
    background_color = 0
    digit_color = 255, 255, 255
    font = ImageFont.truetype(f'./Generate_image_digits/.fonts/{font_name}', 20)
    image = Image.new('RGB', (28, 28), color=background_color)
    draw_digit = ImageDraw.Draw(image)
    draw_digit.text((28/2, 28/2), str_digit, anchor="mm", fill=digit_color, font=font)
    image_cupy_array = cp.asarray(image)
    image_gray = cv.cvtColor(cp.asnumpy(image_cupy_array), cv.COLOR_RGB2GRAY)
    image_arr_normalized = cv.normalize(image_gray, None, 0, 1, cv.NORM_MINMAX, cv.CV_32F)
    return image, image_arr_normalized.reshape(1, -1)
