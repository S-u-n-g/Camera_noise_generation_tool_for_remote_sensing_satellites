import numpy as np
import cv2


def striping_noise_generator(src, noise_strength=15, stripe_width=2):
    rows, cols, channels = src.shape

    # 각 stripe_width 개의 행마다 동일한 노이즈 값을 가지도록 노이즈 생성
    num_stripes = rows // stripe_width  # stripe_width 줄씩 묶어서 노이즈를 적용할 수 있는 줄의 개수
    stripe_noise = np.random.randint(-noise_strength, noise_strength, size=(num_stripes, 1, 1))

    # stripe_width 줄씩 동일한 노이즈를 적용하여 새로운 노이즈 배열 생성
    stripe_noise = np.repeat(stripe_noise, stripe_width, axis=0)

    # 만약 총 행 수가 stripe_width로 나누어 떨어지지 않으면 남은 행에도 노이즈를 추가
    if stripe_noise.shape[0] < rows:
        extra_rows = rows - stripe_noise.shape[0]
        stripe_noise = np.vstack(
            [stripe_noise, np.random.randint(-noise_strength, noise_strength, size=(extra_rows, 1, 1))])

    # 채널 수와 동일한 크기로 노이즈 확장 (BGR 모두 같은 노이즈 적용)
    stripe_noise = np.repeat(stripe_noise, channels, axis=2)

    # 원본 이미지에 노이즈 더하기
    striped_image_array = src + stripe_noise
    striped_image_array = np.clip(striped_image_array, 0, 255)  # 유효한 픽셀 값으로 클리핑

    # 데이터 타입 변환 (uint8로 변환)
    striped_image = striped_image_array.astype(np.uint8)

    return striped_image


def missing_line_generator(src):
    return missing_line_image


if __name__ == '__main__':
    # 이미지 불러오기 (cv2는 BGR 형식으로 이미지를 불러옴)
    image_path = 'input_images/P0000__512__2304___1920.png'  # 원본 이미지 경로
    image = cv2.imread(image_path)

    # striping noise 생성
    striped_image = striping_noise_generator(image, 15, 3)

    # 새로운 이미지 저장 (cv2는 BGR 형식이므로 BGR로 저장)
    # cv2.imwrite('striped_image.png', striped_image)

    # 결과 이미지 보기
    cv2.imshow('Striped Image', striped_image)
    cv2.waitKey(0)  # 키 입력을 기다림
    cv2.destroyAllWindows()
