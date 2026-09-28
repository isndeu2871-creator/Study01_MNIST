"""자바스크립트 구현을 검사할 정답 데이터를 만든다.

    cd desktop_version
    python3 검증데이터만들기.py

만드는 파일
    ../web_version/검증데이터.json   (용량이 커서 깃에 넣지 않는다)

담는 것 — MNIST 시험 이미지 200장에 대해
    그림        앱과 같은 280x280 으로 키운 그림 (PNG)
    전처리결과   preprocess.py 를 통과한 28x28 밝기 배열 (float32)
    확률        모델이 낸 숫자 10개의 확률
    정답        MNIST 라벨

web_version/검증.html 이 같은 그림을 자바스크립트 경로로 통과시켜 이 값과 대조한다.
구현보다 검사 수단을 먼저 만들어 둔다.
"""
import base64
import io
import json
import os

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

from model import 손글씨모델
from preprocess import 전처리, 정규화

여기 = os.path.dirname(os.path.abspath(__file__))
가중치파일 = os.path.join(여기, "mnist_cnn.pt")
웹폴더 = os.path.join(여기, "..", "web_version")
자료폴더 = os.path.join(여기, "data")

장수 = 200
앱칸크기 = 280


def png문자열(그림):
    버퍼 = io.BytesIO()
    그림.save(버퍼, format="PNG")
    return "data:image/png;base64," + base64.b64encode(버퍼.getvalue()).decode()


@torch.no_grad()
def main():
    모델 = 손글씨모델()
    모델.load_state_dict(torch.load(가중치파일, map_location="cpu"))
    모델.eval()

    시험자료 = datasets.MNIST(자료폴더, train=False, download=True)
    항목들 = []
    맞은개수 = 0

    for 번호 in range(장수):
        원본, 정답 = 시험자료[번호]
        큰그림 = 원본.resize((앱칸크기, 앱칸크기), Image.LANCZOS)

        배열 = 전처리(큰그림)
        if 배열 is None:
            continue
        입력 = torch.from_numpy(정규화(배열)).view(1, 1, 28, 28)
        확률 = torch.softmax(모델(입력), dim=1)[0].numpy()
        if int(확률.argmax()) == int(정답):
            맞은개수 += 1

        항목들.append({
            "그림": png문자열(큰그림),
            "전처리결과": base64.b64encode(
                np.ascontiguousarray(배열, dtype="<f4").tobytes()).decode(),
            "확률": [round(float(값), 8) for 값 in 확률],
            "정답": int(정답),
        })

    os.makedirs(웹폴더, exist_ok=True)
    경로 = os.path.join(웹폴더, "검증데이터.json")
    with open(경로, "w", encoding="utf-8") as 파일:
        json.dump({"설명": "desktop_version/검증데이터만들기.py 가 만든 파일",
                   "장수": len(항목들), "칸크기": 앱칸크기,
                   "항목": 항목들}, 파일, ensure_ascii=False)

    크기 = os.path.getsize(경로) / 1e6
    print(f"검증데이터.json {len(항목들)}장 · {크기:.1f}MB · "
          f"파이썬 경로 정확도 {맞은개수 / len(항목들) * 100:.1f}%")


if __name__ == "__main__":
    main()
