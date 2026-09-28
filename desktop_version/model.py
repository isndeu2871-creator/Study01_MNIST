"""손글씨 숫자 인식 신경망.

합성곱 2개, 최대풀링 2개, 전결합 2개로 이루어진 CNN이다.
파라미터는 모두 421,642개이고, 내보낸 가중치는 약 1.6MB이다.

    입력 1x28x28
     → 합성곱1(1→32, 3x3, 패딩 1) → ReLU → 최대풀링 2x2      (32x14x14)
     → 합성곱2(32→64, 3x3, 패딩 1) → ReLU → 최대풀링 2x2     (64x7x7)
     → 펼치기(3136) → 전결합1(3136→128) → ReLU → 드롭아웃
     → 전결합2(128→10)

드롭아웃은 학습할 때만 동작하고 예측할 때는 아무 일도 하지 않는다.
그래서 web_version/모델.js 에는 드롭아웃이 없다.
"""
import torch.nn as nn
import torch.nn.functional as F


class 손글씨모델(nn.Module):
    def __init__(self):
        super().__init__()
        self.합성곱1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.합성곱2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.드롭아웃 = nn.Dropout(0.25)
        self.전결합1 = nn.Linear(64 * 7 * 7, 128)
        self.전결합2 = nn.Linear(128, 10)

    def forward(self, 입력):
        값 = F.max_pool2d(F.relu(self.합성곱1(입력)), 2)
        값 = F.max_pool2d(F.relu(self.합성곱2(값)), 2)
        값 = 값.flatten(1)
        값 = self.드롭아웃(F.relu(self.전결합1(값)))
        return self.전결합2(값)  # 로짓. 확률이 필요하면 softmax 를 따로 건다.
