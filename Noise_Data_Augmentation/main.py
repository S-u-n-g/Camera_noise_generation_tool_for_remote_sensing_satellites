from option import *

if __name__ == '__main__':
    # 이미지 불러오기 (cv2는 BGR 형식으로 이미지를 불러옴)
    image_path = 'input_images/P0000__512__2304___1536.png'  # 원본 이미지 경로
    src = cv2.imread(image_path)

    # striping noise 생성
    striped_image = striping_noise_generator(src, noise_strength=12, stripe_width=2, direction='horizontal')

    # missing line noise 생성
    missing_line_image = missing_line_generator(src, num_threshold=15, len_threshold=512)

    # haze_noise 생성
    hazed_image = haze_noise_generator(src, intensity=0.5)

    # Salt-and-Pepper noise 생성
    salt_pepper_image = salt_pepper_noise_generator(src)

    # Poisson noise 생성
    poisson_image = poisson_noise_generator(src)

    # Atmospheric noise 생성
    atmospheric_image = atmospheric_noise_generator(src, blue_intensity=0.3, green_intensity=0.1, red_intensity=0.05, contrast=0.7)

    # Terrain noise 생성
    terrain_image = terrain_noise_generator(src, noise_intensity=0.3, scale=5)

    # Vignetting noise 생성
    vignetting_image = vignetting_noise_generator(src, strength=0.4)

    # Sun Angle noise 생성
    sun_angle_image = sun_angle_noise_generator(src, angle=45, intensity=0.7, gamma=1.0)

    # 새로운 이미지 저장 (cv2는 BGR 형식이므로 BGR로 저장)
    # cv2.imwrite('striped_image.png', striped_image)
    # cv2.imwrite('missing_line_image.png', missing_line_image)
    # cv2.imwrite('hazed_image.png', hazed_image)
    # cv2.imwrite('hazed_image.png', hazed_image)
    # cv2.imwrite('salt_and_pepper_image.png', salt_pepper_image)
    # cv2.imwrite('poisson_image.png', poisson_image)

    # 결과 이미지 보기
    cv2.imshow('Original Image', src)
    cv2.imshow('Striped Image', striped_image)
    cv2.imshow('Missing Line Image', missing_line_image)
    cv2.imshow('Hazed Image', hazed_image)
    cv2.imshow("Salt and Pepper Noise", salt_pepper_image)
    cv2.imshow("Poisson Noise", poisson_image)
    cv2.imshow("Atmospheric Noise", atmospheric_image)
    cv2.imshow("Terrain Noise", terrain_image)
    cv2.imshow("Vignetting Noise", vignetting_image)
    cv2.imshow("Sun Angle Noise", sun_angle_image)

    opt1 = option_1(src)
    opt2 = option_2(src)
    opt3 = option_3(src)

    cv2.imshow("Option1 Image", opt1)
    cv2.imshow("Option2 Image", opt2)
    cv2.imshow("Option3 Image", opt3)

    cv2.waitKey(0)  # 키 입력을 기다림
    cv2.destroyAllWindows()
