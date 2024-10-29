import numpy as np
import cv2


def striping_noise_generator(src, noise_strength=10, stripe_width=2):
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


def missing_line_generator(src, num_threshold=10, len_threshold=300):
    rows, cols, channels = src.shape
    missing_line_image = src.copy()

    # 무작위로 삭제할 줄의 인덱스 선택
    missing_rows = np.random.choice(rows, size=np.random.randint(1, num_threshold + 1), replace=False)
    for row in missing_rows:
        # 결손 시작 위치와 길이 설정
        start_col = np.random.randint(0, cols)  # 결손 시작 위치
        line_length = np.random.randint(1, len_threshold + 1)  # 결손 길이 (1부터 threshold까지)

        # 결손 구간이 이미지 경계를 넘어가지 않도록 설정
        end_col = min(start_col + line_length, cols)

        # 해당 가로 줄에서 결손 구간을 0으로 설정
        missing_line_image[row, start_col:end_col, :] = 0

    return missing_line_image

def haze_noise_generator(src, haze_strength=150):
    rows, cols, channels = src.shape
    haze_image = src.copy()

    # 안개의 밝기 레벨 설정 (랜덤한 밝기를 가지는 행렬 생성)
    haze_layer = np.random.randint(0, haze_strength, (rows, cols, 1)).astype(np.uint8)
    haze_layer = np.repeat(haze_layer, channels, axis=2)  # 채널 수 맞추기 (1채널을 3채널로 확장)

    # 이미지에 안개 층을 덧씌우기 (haze_image:haze_layer = 7:3 비율로)
    haze_image = cv2.addWeighted(haze_image, 0.7, haze_layer, 0.3, 0)

    return haze_image

if __name__ == '__main__':
    # 이미지 불러오기 (cv2는 BGR 형식으로 이미지를 불러옴)
    image_path = 'input_images/P0000__512__2304___1536.png'  # 원본 이미지 경로
    image = cv2.imread(image_path)

    # striping noise 생성
    striped_image = striping_noise_generator(image, 12, 2)

    # missing line noise 생성
    missing_line_image = missing_line_generator(image, 10, 300)

    # haze_noise 생성
    hazed_image = haze_noise_generator(image, 150)

    # 새로운 이미지 저장 (cv2는 BGR 형식이므로 BGR로 저장)
    # cv2.imwrite('striped_image.png', striped_image)
    # cv2.imwrite('missing_line_image.png', missing_line_image)
    # cv2.imwrite('hazed_image.png', hazed_image)

    # 결과 이미지 보기
    cv2.imshow('Striped Image', striped_image)
    cv2.imshow('Missing Line Image', missing_line_image)
    cv2.imshow('Hazed Image', hazed_image)
    cv2.waitKey(0)  # 키 입력을 기다림
    cv2.destroyAllWindows()
