# data_utils.py

import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def simulate_banding_noise(image_np, intensity=0.1, band_width=10, brightness_factor=1.0):
    h, w, c = image_np.shape
    noisy_image = image_np.copy().astype(np.float32)
    for y in range(h):
        band_intensity = np.sin(y / band_width * np.pi) * 0.5 + 0.5
        noise_factor = 1.0 - (band_intensity * intensity)
        noisy_image[y, :, :] *= noise_factor
    noisy_image *= brightness_factor
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

class BandingNoiseDataset(Dataset):
    def __init__(self, root_dir, transform=None, intensity=0.15, band_width=15, resize=None):
        exts = ['png','jpg','jpeg','bmp','tif','tiff']
        files = []
        for e in exts:
            files += glob.glob(os.path.join(root_dir, '**', f'*.{e}'), recursive=True)
            files += glob.glob(os.path.join(root_dir, '**', f'*.{e.upper()}'), recursive=True)
        self.image_files = sorted(list(dict.fromkeys(files)))
        if not self.image_files:
            raise FileNotFoundError(f"No image files found in {root_dir}")
        self.transform = transform
        self.intensity = intensity
        self.band_width = band_width
        self.num_images = len(self.image_files)
        self.resize = resize

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        idx_t_minus_1 = (idx - 1 + self.num_images) % self.num_images
        idx_t = idx
        idx_t_plus_1 = (idx + 1) % self.num_images
        paths = [self.image_files[idx_t_minus_1], self.image_files[idx_t], self.image_files[idx_t_plus_1]]

        brightness_factors = [
            1.0 + 0.05 * np.sin(idx_t * np.pi / 50),
            1.0 + 0.05 * np.sin((idx_t + 1) * np.pi / 50),
            1.0 + 0.05 * np.sin((idx_t + 2) * np.pi / 50)
        ]

        noisy_images = []
        clean_t_image_float = None

        for i, img_path in enumerate(paths):
            # 안전하게 읽기
            try:
                buf = np.fromfile(img_path, dtype=np.uint8)
            except Exception as e:
                raise FileNotFoundError(f"Failed to read file bytes: {img_path}") from e

            if buf is None or buf.size == 0:
                raise FileNotFoundError(f"Image file is empty or not found: {img_path}")

            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to decode image (not a valid image?): {img_path}")

            # 채널 정리
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError(f"Unsupported image shape after channel processing: {img.shape} ({img_path})")

            if self.resize is not None:
                w, h = self.resize
                img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)

            noisy_np = simulate_banding_noise(img, intensity=self.intensity, band_width=self.band_width, brightness_factor=brightness_factors[i])
            noisy_np_float = noisy_np.astype(np.float32) / 255.0
            clean_np_float = img.astype(np.float32) / 255.0

            if i == 1:
                clean_t_image_float = clean_np_float

            noisy_tensor = torch.from_numpy(noisy_np_float.transpose(2,0,1)).float()
            noisy_images.append(noisy_tensor)

        # 반환 전 안전 검사: 중앙 프레임이 없으면 재시도/오류
        if clean_t_image_float is None:
            center_path = paths[1]
            try:
                buf = np.fromfile(center_path, dtype=np.uint8)
                if buf is None or buf.size == 0:
                    raise FileNotFoundError(f"Center image empty: {center_path}")
                img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError(f"Failed to decode center image: {center_path}")
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.ndim == 3 and img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                if img.ndim == 3 and img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                else:
                    raise ValueError(f"Unsupported center image shape: {img.shape} ({center_path})")
                if self.resize is not None:
                    w, h = self.resize
                    img = cv2.resize(img, (int(w), int(h)), interpolation=cv2.INTER_LINEAR)
                clean_t_image_float = img.astype(np.float32) / 255.0
            except Exception as e:
                raise RuntimeError(f"clean_t_image_float is None for idx={idx}, paths={paths}") from e

        noisy_input_9ch = torch.cat(noisy_images, dim=0)
        clean_t_tensor = torch.from_numpy(clean_t_image_float.transpose(2,0,1)).float()
        noisy_t_plus_1 = noisy_images[2].clone()
        return noisy_input_9ch, clean_t_tensor, noisy_t_plus_1