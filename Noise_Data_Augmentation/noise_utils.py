import numpy as np
import cv2


def striping_noise_generator(src, noise_strength=10, stripe_width=2, direction='horizontal'):
    rows, cols, channels = src.shape

    if direction == "horizontal":
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

    elif direction == "vertical":
        # 각 stripe_width 개의 열마다 동일한 노이즈 값을 가지도록 노이즈 생성
        num_stripes = cols // stripe_width
        stripe_noise = np.random.randint(-noise_strength, noise_strength, size=(1, num_stripes, 1))

        # stripe_width 열씩 동일한 노이즈를 적용하여 새로운 노이즈 배열 생성
        stripe_noise = np.repeat(stripe_noise, stripe_width, axis=1)

        # 만약 총 열 수가 stripe_width로 나누어 떨어지지 않으면 남은 열에도 노이즈를 추가
        if stripe_noise.shape[1] < cols:
            extra_cols = cols - stripe_noise.shape[1]
            stripe_noise = np.hstack(
                [stripe_noise, np.random.randint(-noise_strength, noise_strength, size=(1, extra_cols, 1))])

        # 채널 수와 동일한 크기로 노이즈 확장 (BGR 모두 같은 노이즈 적용)
        stripe_noise = np.repeat(stripe_noise, channels, axis=2)

        # 원본 이미지에 노이즈 더하기
        striped_image_array = src + stripe_noise

    else:
        raise ValueError("direction 파라미터는 'horizontal' 또는 'vertical' 값만 허용합니다.")

    # 유효한 픽셀 값으로 클리핑
    striped_image_array = np.clip(striped_image_array, 0, 255)

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
        line_length = np.random.randint(100, len_threshold + 1)  # 결손 길이 (100부터 len_threshold까지)

        # 결손 구간이 이미지 경계를 넘어가지 않도록 설정
        end_col = min(start_col + line_length, cols)

        # 해당 가로 줄에서 결손 구간을 0으로 설정
        missing_line_image[row, start_col:end_col, :] = 0

    return missing_line_image


def haze_noise_generator(src, intensity=0.5):
    # 이미지 크기 확인
    height, width, channels = src.shape

    # 흰색 조명의 헤이즈 효과 생성
    white_haze = np.full((height, width, channels), 200, dtype=np.uint8)

    # 원본 이미지와 헤이즈 블렌딩
    hazy_image = cv2.addWeighted(src, 1 - intensity, white_haze, intensity, 0)

    return hazy_image

def gaussian_noise_generator(src, mean=0, var=50):
    sigma = var ** 0.5
    gaussian_noise = np.random.normal(mean, sigma, src.shape)
    noisy_image = src + gaussian_noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def salt_pepper_noise_generator(src):
    noisy_image = np.copy(src)
    s_vs_p = 0.5
    amount = 0.02

    # Salt 노이즈 (흰색) 추가
    num_salt = np.ceil(amount * src.size * s_vs_p / src.shape[2])  # 채널 수로 나누어 픽셀 수로 맞춤
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in src.shape[:2]]
    noisy_image[coords[0], coords[1], :] = 255  # 해당 좌표에 흰색 노이즈 적용

    # Pepper 노이즈 (검은색) 추가
    num_pepper = np.ceil(amount * src.size * (1.0 - s_vs_p) / src.shape[2])
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in src.shape[:2]]
    noisy_image[coords[0], coords[1], :] = 0  # 해당 좌표에 검은색 노이즈 적용

    return noisy_image


def poisson_noise_generator(src):
    noisy_image = np.copy(src)
    for i in range(src.shape[2]):  # 채널별로 독립적으로 적용
        noisy_image[:, :, i] = np.random.poisson(src[:, :, i]).astype(np.uint8)
    return noisy_image


def atmospheric_noise_generator(src, blue_intensity=0.3, green_intensity=0.1, red_intensity=0.05, contrast=0.7):
    # 이미지 정규화 (0~1)
    src = src / 255.0

    # 채널별로 다른 강도의 산란 효과 추가
    haze_effect = src.copy()
    haze_effect[:, :, 0] += blue_intensity  # 파란색 채널
    haze_effect[:, :, 1] += green_intensity  # 녹색 채널
    haze_effect[:, :, 2] += red_intensity  # 빨간색 채널
    haze_effect = np.clip(haze_effect, 0, 1)

    # 대비 감소 효과
    mean_intensity = np.mean(haze_effect, axis=(0, 1), keepdims=True)
    contrast_effect = mean_intensity + contrast * (haze_effect - mean_intensity)

    # 결과 이미지
    noisy_image = np.clip(contrast_effect, 0, 1)
    return (noisy_image * 255).astype(np.uint8)


def terrain_noise_generator(src, noise_intensity=0.3, scale=5):
    # 이미지를 0~1 범위로 정규화
    src = src / 255.0

    # Terrain 노이즈 패턴 생성
    terrain_pattern = np.random.rand(src.shape[0], src.shape[1], 1) * noise_intensity
    terrain_noise = terrain_pattern * np.sin(np.linspace(0, scale * np.pi, src.shape[1])).reshape(1, -1, 1)
    noisy_image = src + terrain_noise
    noisy_image = np.clip(noisy_image, 0, 1)

    return (noisy_image * 255).astype(np.uint8)


def vignetting_noise_generator(src, strength=0.4):
    # strength: 비네팅 강도 (0 ~ 1 사이의 값, 높을수록 강함)

    rows, cols = src.shape[:2]

    # power: 거리의 지수 적용 값 (높을수록 급격히 어두워짐)
    power = 2.3

    # 거리 기반 마스크 생성 (중앙에서 가장자리로 갈수록 값이 작아짐)
    X_result, Y_result = np.meshgrid(np.linspace(-1, 1, cols), np.linspace(-1, 1, rows))
    distance = np.sqrt(X_result ** 2 + Y_result ** 2)

    # 거리의 지수화를 통해 가장자리와 중앙의 차이를 크게 만듦
    mask = 1 - (distance ** power * strength)
    mask = np.clip(mask, 0, 1)

    # 컬러 이미지인 경우 마스크를 3채널로 확장
    if len(src.shape) == 3 and src.shape[2] == 3:
        mask = cv2.merge([mask] * 3)

    # 비네팅 마스크를 이미지에 곱해서 가장자리 어둡게
    vignetting_image = src * mask
    return np.clip(vignetting_image, 0, 255).astype(np.uint8)


def sun_angle_noise_generator(src, angle=45, intensity=0.5, gamma=1.0):
    """
    태양 고도각에 따른 노이즈를 생성합니다.

    Parameters:
    src (numpy.ndarray): 원본 이미지
    angle (float): 태양의 고도각 (0 ~ 90도)
    intensity (float): 조명의 강도 (0 ~ 1 사이의 값)
    gamma (float): 감마 보정 값 (1.0 이상 권장)

    Returns:
    numpy.ndarray: 노이즈가 추가된 이미지
    """
    rows, cols = src.shape[:2]

    # 고도각을 라디안으로 변환 후 sin 값을 계산
    angle_rad = np.deg2rad(angle)
    sin_alpha = np.sin(angle_rad)

    # 고도각이 낮을 때 안전 장치 추가
    if sin_alpha < 0.1:  # 특정 각도 이하에서는 최소 밝기값으로 설정
        sin_alpha = 0.1

    # 노이즈 강도 계산 (감마 보정을 추가하여 비선형 효과 적용)
    noise_factor = 1 / (sin_alpha ** gamma * intensity + (1 - intensity))

    # 이미지에 노이즈 적용
    noisy_image = src * noise_factor

    # 값의 범위를 0 ~ 255로 클리핑하고 정수형으로 변환
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image
