from noise_utils import *


def option_1(src):
    opt1_image = src.copy()
    opt1_image = striping_noise_generator(opt1_image)
    opt1_image = salt_pepper_noise_generator(opt1_image)
    opt1_image = haze_noise_generator(opt1_image)
    return opt1_image


def option_2(src):
    opt2_image = src.copy()
    opt2_image = missing_line_generator(opt2_image)
    opt2_image = atmospheric_noise_generator(opt2_image)
    opt2_image = vignetting_noise_generator(opt2_image)
    return opt2_image


def option_3(src):
    opt3_image = src.copy()
    opt3_image = striping_noise_generator(opt3_image, direction='vertical')
    opt3_image = sun_angle_noise_generator(opt3_image)
    opt3_image = terrain_noise_generator(opt3_image)
    return opt3_image
