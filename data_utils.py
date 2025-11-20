# data_utils.py

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

# --- 1. 줄무늬 노이즈 시뮬레이션 함수 (주기성 및 밝기 조절 포함) ---
def simulate_banding_noise(image_np, intensity=0.1, band_width=10, brightness_factor=1.0):
    """numpy 이미지에 인위적인 줄무늬 노이즈를 추가하고 밝기를 조절합니다."""
    h, w, c = image_np.shape
    noisy_image = image_np.copy().astype(np.float32)

    # 1. 줄무늬 패턴 적용 (공간적 노이즈)
    for y in range(h):
        band_intensity = np.sin(y / band_width * np.pi) * 0.5 + 0.5 
        noise_factor = 1.0 - (band_intensity * intensity)
        noisy_image[y, :, :] *= noise_factor

    # 2. 주기적 밝기 조절 적용 (시간적 플리커 시뮬레이션)
    noisy_image *= brightness_factor

    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

# --- 2. PyTorch 데이터셋 클래스 (하위 폴더 재귀 검색 및 3가지 반환) ---
class BandingNoiseDataset(Dataset):
    def __init__(self, root_dir, transform=None, intensity=0.15, band_width=15, resize=None):
        """
        Args:
            root_dir (str): 이미지 파일의 최상위 부모 디렉토리 경로 (하위 폴더 포함).
            resize (tuple or None): (width, height)로 리사이즈. None이면 원본 크기 유지.
        """
        # ⭐⭐ 파일 확장자 케이스 무시 검색 추가 및 정렬 ⭐⭐
        exts = ['png','jpg','jpeg','bmp','tif','tiff']
        files = []
        for e in exts:
            files += glob.glob(os.path.join(root_dir, '**', f'*.{e}'), recursive=True)
            files += glob.glob(os.path.join(root_dir, '**', f'*.{e.upper()}'), recursive=True)
        # 중복 제거 및 정렬(재현성)
        self.image_files = sorted(list(dict.fromkeys(files)))

        if not self.image_files:
            raise FileNotFoundError(f"No image files found in {root_dir}")

        self.transform = transform
        self.intensity = intensity
        self.band_width = band_width
        self.num_images = len(self.image_files)
        # 리사이즈 저장 (width, height) 형태로 받음
        self.resize = resize

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # t-1, t, t+1 세 장의 이미지를 순차적으로 가져오기 위해 인덱스 계산
        idx_t_minus_1 = (idx - 1 + self.num_images) % self.num_images
        idx_t         = idx
        idx_t_plus_1  = (idx + 1) % self.num_images
        
        paths = [self.image_files[idx_t_minus_1], self.image_files[idx_t], self.image_files[idx_t_plus_1]]
        
        # 주기성 시뮬레이션을 위한 밝기 계수
        brightness_factors = [1.0 + 0.05 * np.sin(idx_t * np.pi / 50),
                              1.0 + 0.05 * np.sin((idx_t + 1) * np.pi / 50),
                              1.0 + 0.05 * np.sin((idx_t + 2) * np.pi / 50)]
        
        noisy_images = []
        clean_t_image_float = None

        for i, img_path in enumerate(paths):
            # 안전하게 파일 바이트 읽기 (Windows 유니코드 경로 대응)
            buf = None
            try:
                buf = np.fromfile(img_path, dtype=np.uint8)
            except Exception as e:
                raise FileNotFoundError(f"Failed to read file bytes: {img_path}") from e

            if buf is None or buf.size == 0:
                raise FileNotFoundError(f"Image file is empty or not found: {img_path}")

            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to decode image (not a valid image?): {img_path}")

            # 채널 정리: 그레이스케일->RGB, 알파 채널 제거
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            # BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 리사이즈가 지정된 경우 (width, height)
            if self.resize is not None:
                # 입력이 (width, height)로 들어온다고 가정
                w, h = self.resize
                img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)

            # 노이즈 시뮬레이션 적용 (numpy uint8 이미지)
            noisy_np = simulate_banding_noise(img, intensity=self.intensity, band_width=self.band_width, brightness_factor=brightness_factors[i])

            # 0..255 -> 0..1 float32
            noisy_np_float = noisy_np.astype(np.float32) / 255.0
            clean_np_float = img.astype(np.float32) / 255.0

            # t 프레임(clean)은 이후 반환용으로 따로 저장
            if i == 1:
                clean_t_image_float = clean_np_float

            # HWC -> CHW 텐서 변환
            noisy_tensor = torch.from_numpy(noisy_np_float.transpose(2,0,1)).float()
            noisy_images.append(noisy_tensor)

        # t-1, t, t+1의 노이즈 이미지를 채널 방향으로 합쳐 9채널 입력 텐서 생성
        noisy_input_9ch = torch.cat(noisy_images, dim=0)
        clean_t_tensor = torch.from_numpy(clean_t_image_float.transpose(2,0,1)).float()

        # 반환: (noisy_9ch, clean_t, noisy_t_plus_1) — noisy_t_plus_1는 원래 noisy_images[2]로 반환
        noisy_t_plus_1 = noisy_images[2].clone()
        return noisy_input_9ch, clean_t_tensor, noisy_t_plus_1