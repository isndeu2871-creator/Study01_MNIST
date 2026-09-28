"""mnist_cnn.pt 를 웹 버전이 읽을 수 있는 파일 2개로 내보낸다.

    cd desktop_version
    python3 가중치내보내기.py

만드는 파일
    ../web_version/가중치.bin       층 4개의 weight 와 bias(텐서 8개)를 float32 로 이어 붙인 것
    ../web_version/가중치정보.json  텐서별 이름·형상·위치와 정규화 상수

ONNX 로 바꾸지 않고 가중치만 내보낸다. 순전파는 web_version/모델.js 가 다시 구현한다.
정규화 상수를 자바스크립트에 직접 적지 않고 이 파일에 담는 이유는 값의 출처를
preprocess.py 하나로 유지하기 위해서다.
"""
import json
import os

import numpy as np
import torch

from model import 손글씨모델
from preprocess import 평균, 표준편차

여기 = os.path.dirname(os.path.abspath(__file__))
가중치파일 = os.path.join(여기, "mnist_cnn.pt")
웹폴더 = os.path.join(여기, "..", "web_version")

# 자바스크립트가 계산하는 순서와 같게 늘어놓는다
내보낼텐서 = [
    "합성곱1.weight", "합성곱1.bias",
    "합성곱2.weight", "합성곱2.bias",
    "전결합1.weight", "전결합1.bias",
    "전결합2.weight", "전결합2.bias",
]


def main():
    상태 = torch.load(가중치파일, map_location="cpu")
    빠진것 = [이름 for 이름 in 내보낼텐서 if 이름 not in 상태]
    if 빠진것:
        raise SystemExit(f"가중치에 없는 텐서: {빠진것}")

    os.makedirs(웹폴더, exist_ok=True)
    정보 = []
    위치 = 0
    with open(os.path.join(웹폴더, "가중치.bin"), "wb") as 파일:
        for 이름 in 내보낼텐서:
            값 = 상태[이름].detach().cpu().numpy().astype("<f4")
            파일.write(np.ascontiguousarray(값).tobytes())
            정보.append({"이름": 이름, "형상": list(값.shape),
                        "위치": 위치, "개수": int(값.size)})
            위치 += 값.size * 4

    파라미터수 = sum(항목["개수"] for 항목 in 정보)
    문서 = {
        "설명": "desktop_version/가중치내보내기.py 가 만든 파일",
        "구조": "합성곱2 + 최대풀링2 + 전결합2",
        "입력형상": [1, 28, 28],
        "자료형": "float32",
        "정규화": {"평균": 평균, "표준편차": 표준편차},
        "파라미터수": 파라미터수,
        "텐서": 정보,
    }
    with open(os.path.join(웹폴더, "가중치정보.json"), "w", encoding="utf-8") as 파일:
        json.dump(문서, 파일, ensure_ascii=False, indent=2)

    print(f"가중치.bin {위치:,}바이트 · 파라미터 {파라미터수:,}개")


if __name__ == "__main__":
    main()
