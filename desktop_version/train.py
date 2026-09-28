"""MNIST로 손글씨 모델을 학습하고 mnist_cnn.pt 로 저장한다.

    cd desktop_version
    python3 train.py

자료는 상대 경로 data 에 받으므로 반드시 desktop_version 폴더 안에서 실행한다.
원본 MNIST는 저장소에 넣지 않는다. 처음 실행할 때 내려받는다.
"""
import argparse
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import 손글씨모델
from preprocess import 평균, 표준편차

자료폴더 = "data"
가중치파일 = "mnist_cnn.pt"


def 자료불러오기(묶음크기):
    다듬기 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((평균,), (표준편차,)),
    ])
    학습자료 = datasets.MNIST(자료폴더, train=True, download=True, transform=다듬기)
    시험자료 = datasets.MNIST(자료폴더, train=False, download=True, transform=다듬기)
    return (DataLoader(학습자료, batch_size=묶음크기, shuffle=True),
            DataLoader(시험자료, batch_size=1000))


def 한바퀴학습(모델, 장치, 학습묶음, 최적화기):
    모델.train()
    총손실 = 0.0
    for 그림, 정답 in 학습묶음:
        그림, 정답 = 그림.to(장치), 정답.to(장치)
        최적화기.zero_grad()
        손실 = F.cross_entropy(모델(그림), 정답)
        손실.backward()
        최적화기.step()
        총손실 += 손실.item()
    return 총손실 / len(학습묶음)


@torch.no_grad()
def 정확도재기(모델, 장치, 시험묶음):
    모델.eval()
    맞은개수 = 0
    전체 = 0
    for 그림, 정답 in 시험묶음:
        그림, 정답 = 그림.to(장치), 정답.to(장치)
        맞은개수 += (모델(그림).argmax(dim=1) == 정답).sum().item()
        전체 += len(정답)
    return 맞은개수 / 전체


def main():
    받은인자 = argparse.ArgumentParser(description="MNIST 손글씨 모델 학습")
    받은인자.add_argument("--에포크", type=int, default=5)
    받은인자.add_argument("--묶음크기", type=int, default=128)
    받은인자.add_argument("--학습률", type=float, default=1e-3)
    설정 = 받은인자.parse_args()

    장치 = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    학습묶음, 시험묶음 = 자료불러오기(설정.묶음크기)
    모델 = 손글씨모델().to(장치)
    최적화기 = torch.optim.Adam(모델.parameters(), lr=설정.학습률)
    일정 = torch.optim.lr_scheduler.StepLR(최적화기, step_size=2, gamma=0.5)

    파라미터수 = sum(p.numel() for p in 모델.parameters())
    print(f"장치 {장치} · 파라미터 {파라미터수:,}개")

    for 회차 in range(1, 설정.에포크 + 1):
        시작 = time.time()
        손실 = 한바퀴학습(모델, 장치, 학습묶음, 최적화기)
        일정.step()
        정확도 = 정확도재기(모델, 장치, 시험묶음)
        print(f"{회차}/{설정.에포크} 손실 {손실:.4f} 시험정확도 {정확도 * 100:.2f}% "
              f"({time.time() - 시작:.0f}초)", flush=True)

    torch.save(모델.state_dict(), 가중치파일)
    print(f"{가중치파일} 저장 완료")


if __name__ == "__main__":
    main()
