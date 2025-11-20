# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import random
import time
import torch.backends.cudnn as cudnn

# 다른 파일에서 필요한 모듈 가져오기
from model import SimpleUNet
from data_utils import BandingNoiseDataset
from loss import ReconstructionLoss, TemporalConsistencyLoss

torch.cuda.empty_cache()

# --- 하이퍼파라미터 설정 ---
BATCH_SIZE = 2          
EPOCHS = 10
LEARNING_RATE = 1e-4
LAMBDA_TEMP = 0.5       

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# 데이터 변환 (data_utils에서 직접 [0, 1] float32 변환하므로 transform=None으로 설정)
transform = transforms.Compose([])

# ⭐⭐⭐ 데이터 경로 설정: 이 부분을 실제 경로로 반드시 수정하세요! ⭐⭐⭐
# 예시: ROOT_DIR = './Training/01.원천데이터'
ROOT_DIR = './Training/01.Photo/TS_CombBlur_Bright_01.AV'

# cuDNN 튜닝 (입력 크기 고정이면 속도 향상)
cudnn.benchmark = True
# optional: CPU 스레드 제한(환경에 맞게 조정)
torch.set_num_threads(min(8, max(1, (os.cpu_count() or 4) - 1)))
# AMP 초기화용 스케일러 (main 가드 아래 모델/optimizer 이후 초기화해도 무방)
scaler = torch.cuda.amp.GradScaler()

# --- 트레이닝 루프 ---
def train_model():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        last_spatial_loss = 0.0
        last_temporal_loss = 0.0
        epoch_start = time.time()
        
        for batch_idx, (noisy_input_9ch, clean_t, noisy_t_plus_1) in enumerate(dataloader):
            # 비동기 전송 (pin_memory=True 필요)
            noisy_input_9ch = noisy_input_9ch.to(device, dtype=torch.float, non_blocking=True)
            clean_t = clean_t.to(device, dtype=torch.float, non_blocking=True)
            noisy_t_plus_1 = noisy_t_plus_1.to(device, dtype=torch.float, non_blocking=True)

            # AMP 자동 mixed precision
            with torch.cuda.amp.autocast():
                output_t = model(noisy_input_9ch)
                spatial_loss = recon_loss_fn(output_t, clean_t)
                noisy_input_9ch_plus_1 = torch.cat([noisy_t_plus_1] * 3, dim=1).to(device, dtype=torch.float, non_blocking=True)
                output_t_plus_1 = model(noisy_input_9ch_plus_1)
                temporal_loss = temp_loss_fn(output_t, output_t_plus_1)
                loss = spatial_loss + LAMBDA_TEMP * temporal_loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # (디버그) GPU 동기화 후 시간 측정 — 프로덕션에선 주석 가능
            torch.cuda.synchronize()
            batch_time = time.time() - epoch_start
            epoch_start = time.time()
            
            total_loss += loss.item()
            last_spatial_loss = spatial_loss.item()
            last_temporal_loss = temporal_loss.item()
            
            # ⭐⭐⭐ 배치 단위 로깅 추가 ⭐⭐⭐
            if ((batch_idx + 1) % 5 == 0) or (batch_idx + 1 == len(dataloader)):
                print(f'  [Batch {batch_idx+1}/{len(dataloader)}] Loss:{loss.item():.6f} Spat:{spatial_loss.item():.6f} Temp:{temporal_loss.item():.6f} batch_time:{batch_time:.3f}s', flush=True)

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Spatial Loss: {last_spatial_loss:.4f}, Temporal Loss: {last_temporal_loss:.4f}')

        # 모델 저장
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'unet_debanding_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    print(f"Using device: {device}")

    # 데이터 경로 및 데이터셋 초기화
    # 이미지 크기를 (width, height) = (1280, 720)으로 리사이즈하여 로드
    dataset = BandingNoiseDataset(root_dir=ROOT_DIR, transform=None, resize=(1280, 720))

    # --- 전체 이미지 중 랜덤으로 최대 N개만 사용 ---
    SAMPLE_SIZE = 5000
    RANDOM_SEED = 42
    if len(dataset.image_files) > SAMPLE_SIZE:
        random.seed(RANDOM_SEED)
        sampled_files = random.sample(dataset.image_files, SAMPLE_SIZE)
        dataset.image_files = sampled_files
        dataset.num_images = len(dataset.image_files)
        print(f"샘플링: 전체 {len(sampled_files)}개 이미지 사용 (랜덤 시드={RANDOM_SEED})")

    # DataLoader: 워커 수는 필요시 조정
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,             # 시스템에 따라 2~12로 실험
        pin_memory=True,           # non_blocking 전송과 함께 사용
        persistent_workers=True,   # 워커 재사용(오버헤드 감소)
        prefetch_factor=2
    )

    # 모델, 손실 함수, 옵티마이저 초기화
    model = SimpleUNet(in_channels=9, out_channels=3).to(device)
    recon_loss_fn = ReconstructionLoss(loss_type='L1').to(device)
    temp_loss_fn = TemporalConsistencyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_model()