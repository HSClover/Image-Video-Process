# loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. 시간적 일관성 손실 함수 (Temporal Consistency Loss) ---
class TemporalConsistencyLoss(nn.Module):
    def __init__(self):
        super(TemporalConsistencyLoss, self).__init__()

    def forward(self, restored_frame_t, restored_frame_t_plus_1):
        temporal_diff = torch.abs(restored_frame_t - restored_frame_t_plus_1)
        loss = torch.mean(temporal_diff)
        return loss
    
# --- 2. 공간적 복원 손실 함수 ---
class ReconstructionLoss(nn.Module):
    def __init__(self, loss_type='L1'):
        super(ReconstructionLoss, self).__init__()
        if loss_type == 'L1':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'MSE':
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError("Unsupported loss type")

    def forward(self, output, target):
        return self.loss_fn(output, target)