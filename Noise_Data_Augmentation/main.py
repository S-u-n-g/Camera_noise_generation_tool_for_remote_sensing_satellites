import numpy as np
import cv2
from noise_utils import *

if __name__ == '__main__':
    # 이미지 불러오기 (cv2는 BGR 형식으로 이미지를 불러옴)
    image_path = 'input_images/P0000__512__2304___1536.png'  # 원본 이미지 경로
    src = cv2.imread(image_path)

    # striping noise 생성
    striped_image = striping_noise_generator(src, 12, 2, 'horizontal')

    # missing line noise 생성
    missing_line_image = missing_line_generator(src, 15, 512)

    # haze_noise 생성
    hazed_image = haze_noise_generator(src, 150)

    # Gaussian noise 생성
    gaussian_image = add_gaussian_noise(src, mean=0, var=100)

    # Salt-and-Pepper noise 생성
    salt_pepper_image = add_salt_pepper_noise(src, s_vs_p=0.5, amount=0.02)

    # Poisson noise 생성
    poisson_image = add_poisson_noise(src)

    # Speckle noise 생성
    speckle_image = add_speckle_noise(src, mean=0, var=0.01)

    # Vignetting noise 생성
    vignetting_image = add_vignetting_noise(src, strength=0.4)

    # Sun Angle noise 생성
    sun_angle_image = add_sun_angle_noise(src, angle=90, intensity=1)

    # 새로운 이미지 저장 (cv2는 BGR 형식이므로 BGR로 저장)
    # cv2.imwrite('striped_image.png', striped_image)
    # cv2.imwrite('missing_line_image.png', missing_line_image)
    # cv2.imwrite('hazed_image.png', hazed_image)
    # cv2.imwrite('hazed_image.png', hazed_image)
    # cv2.imwrite('gaussian_image.png', gaussian_image)
    # cv2.imwrite('salt_and_pepper_image.png', salt_pepper_image)
    # cv2.imwrite('poisson_image.png', poisson_image)
    # cv2.imwrite('speckle_image.png', speckle_image)

    # 결과 이미지 보기
    cv2.imshow('Original Image', src)
    cv2.imshow('Striped Image', striped_image)
    cv2.imshow('Missing Line Image', missing_line_image)
    cv2.imshow('Hazed Image', hazed_image)
    cv2.imshow("Gaussian Noise", gaussian_image)
    cv2.imshow("Salt and Pepper Noise", salt_pepper_image)
    cv2.imshow("Poisson Noise", poisson_image)
    cv2.imshow("Speckle Noise", speckle_image)
    cv2.imshow("Vignetting Noise", vignetting_image)
    cv2.imshow("Sun Angle Noise", sun_angle_image)

    cv2.waitKey(0)  # 키 입력을 기다림
    cv2.destroyAllWindows()
