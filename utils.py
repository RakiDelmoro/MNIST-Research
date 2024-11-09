import random
from cupy_utils.utils import one_hot
from Generate_image_digits.digit_generator import digit_to_image

def digit_generator():
    while True:
        generated_digit = random.randint(1, 9)
        one_hot_digit = one_hot(x=generated_digit, number_of_classes=10)
        pil_image, array_image = digit_to_image(str(generated_digit))
        pil_image.save("test.png")
        yield array_image, one_hot_digit
