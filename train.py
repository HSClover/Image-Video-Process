# train.py
import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import torch.backends.cudnn as cudnn

# 다른 파일에서 필요한 모듈 가져오기
# (파일이 없으면 에러가 나므로 해당 파일들이 존재한다고 가정합니다)
from model import SimpleUNet
from data_utils import BandingNoiseDataset
from loss import ReconstructionLoss, TemporalConsistencyLoss

# --- 하이퍼파라미터 설정 ---
BATCH_SIZE = 4          
EPOCHS = 10
LEARNING_RATE = 1e-4
LAMBDA_TEMP = 0.5       
MODEL_SAVE_PATH = "model_final.pth"  # 모델 저장 파일명

# 데이터 경로 (4개 폴더 리스트)
ROOT_DIRS = [
    './Training/01.Photo/TS_CombBlur_Bright_01.AV',
    './Training/01.Photo/TS_CombBlur_Bright_02.BO_Box',
    './Training/01.Photo/TS_CombBlur_Bright_03.GA_Gausian',
    './Training/01.Photo/TS_CombBlur_Bright_04.LS_Lens'
]

def train_model(model, dataloader, optimizer, recon_loss_fn, temp_loss_fn, scaler, autocast_fn, autocast_kwargs, device, save_path):
    """
    학습 루프 함수
    모든 필요한 객체를 인자로 받아서 실행 (Scope 문제 해결)
    """
    model.train()
    print(f"Start training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        last_spatial_loss = 0.0
        last_temporal_loss = 0.0
        epoch_start = time.time()
        
        for batch_idx, (noisy_input_9ch, clean_t, noisy_t_plus_1) in enumerate(dataloader):
            # 비동기 전송
            noisy_input_9ch = noisy_input_9ch.to(device, dtype=torch.float, non_blocking=True)
            clean_t = clean_t.to(device, dtype=torch.float, non_blocking=True)
            noisy_t_plus_1 = noisy_t_plus_1.to(device, dtype=torch.float, non_blocking=True)

            # AMP 자동 mixed precision
            # Pylance가 타입 경고를 낼 수 있으나 런타임에는 정상 동작합니다.
            with autocast_fn(**autocast_kwargs):
                # 1. 순전파 (t 프레임 복원)
                output_t = model(noisy_input_9ch)
                
                # 2. 손실 계산
                spatial_loss = recon_loss_fn(output_t, clean_t)
                
                # t+1 예측을 위한 입력 구성
                noisy_input_9ch_plus_1 = torch.cat([noisy_t_plus_1] * 3, dim=1).to(device, dtype=torch.float, non_blocking=True)
                output_t_plus_1 = model(noisy_input_9ch_plus_1)
                
                temporal_loss = temp_loss_fn(output_t, output_t_plus_1)
                
                loss = spatial_loss + LAMBDA_TEMP * temporal_loss

            # 3. 역전파 및 최적화 (Scaler 사용)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # (디버그) GPU 동기화 후 시간 측정
            # torch.cuda.synchronize()
            
            total_loss += loss.item()
            last_spatial_loss = spatial_loss.item()
            last_temporal_loss = temporal_loss.item()
            
            # 로그 출력
            if ((batch_idx + 1) % 5 == 0) or (batch_idx + 1 == len(dataloader)):
                elapsed = time.time() - epoch_start
                print(f'  [Epoch {epoch+1}/{EPOCHS}] [Batch {batch_idx+1}/{len(dataloader)}] '
                      f'Loss:{loss.item():.6f} Spat:{spatial_loss.item():.6f} Temp:{temporal_loss.item():.6f} '
                      f'Time:{elapsed:.2f}s', flush=True)

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_loss:.6f}")

        # 에포크마다 모델 가중치 업데이트 (덮어쓰기)
        torch.save(model.state_dict(), save_path)
        print(f"  >> Model weights updated to '{save_path}'")

    print("Training Finished.")
    
    # 학습 완료 후 모델 저장
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to '{save_path}'")


if __name__ == '__main__':
    # 1. 기본 설정
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # cuDNN 및 스레드 설정
    cudnn.benchmark = True
    torch.set_num_threads(min(8, max(1, (os.cpu_count() or 4) - 1)))

    # 2. 데이터셋 및 데이터로더 초기화
    # 학습 속도를 위해 1280x720 리사이즈 적용
    
    datasets = []
    SAMPLE_SIZE = 5000   # 폴더별 샘플링 개수 (4개 폴더 * 3000 = 약 12000장)

    for root_dir in ROOT_DIRS:
        if not os.path.exists(root_dir):
            print(f"Warning: Directory not found: {root_dir}")
            continue
            
        ds = BandingNoiseDataset(root_dir=root_dir, transform=None, resize=(1280, 720))
        
        # 샘플링
        if len(ds.image_files) > SAMPLE_SIZE:
            random.seed(42)
            ds.image_files = random.sample(ds.image_files, SAMPLE_SIZE)
            ds.num_images = len(ds.image_files)
            print(f"[{os.path.basename(root_dir)}] 샘플링: {len(ds.image_files)}개 이미지 사용")
        else:
            print(f"[{os.path.basename(root_dir)}] 전체 사용: {len(ds.image_files)}개 이미지")
            
        datasets.append(ds)

    if not datasets:
        print("Error: No valid datasets found.")
        exit()

    combined_dataset = ConcatDataset(datasets)
    print(f"Total training images: {len(combined_dataset)}")

    dataloader = DataLoader(
        combined_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,              # 시스템에 맞춰 조정 (2~8)
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )

    # 3. 모델, 손실함수, 옵티마이저 초기화
    model = SimpleUNet().to(device)

    # --- [추가됨] 기존 학습된 가중치가 있으면 불러오기 ---
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Found existing model weights at '{MODEL_SAVE_PATH}'. Loading...")
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
            print(">> Weights loaded successfully. Resuming training...")
        except Exception as e:
            print(f">> Error loading weights: {e}")
            print(">> Starting training from scratch.")
    else:
        print("No existing model weights found. Starting training from scratch.")
    # -------------------------------------------------------

    recon_loss_fn = ReconstructionLoss().to(device)
    temp_loss_fn = TemporalConsistencyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. AMP(Mixed Precision) 호환성 설정
    # torch.amp (PyTorch 1.10+) 또는 torch.cuda.amp (구버전) 자동 선택
    if hasattr(torch, 'amp') and hasattr(torch.amp, 'GradScaler'):
        # Pylance 오류(reportPrivateImportUsage) 회피를 위해 getattr 사용
        GradScalerClass = getattr(torch.amp, 'GradScaler')
    else:
        GradScalerClass = getattr(torch.cuda.amp, 'GradScaler')
    scaler = GradScalerClass()

    if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast'):
        # Pylance 오류(reportPrivateImportUsage) 회피를 위해 getattr 사용
        autocast_fn = getattr(torch.amp, 'autocast')
        autocast_kwargs = {'device_type': DEVICE_TYPE}
    else:
        autocast_fn = torch.cuda.amp.autocast
        autocast_kwargs = {}

    # 5. 학습 시작 (모든 객체를 인자로 전달)
    train_model(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        recon_loss_fn=recon_loss_fn,
        temp_loss_fn=temp_loss_fn,
        scaler=scaler,
        autocast_fn=autocast_fn,
        autocast_kwargs=autocast_kwargs,
        device=device,
        save_path=MODEL_SAVE_PATH
    )
