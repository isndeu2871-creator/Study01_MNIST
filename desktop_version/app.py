"""마우스로 숫자를 쓰면 알아맞히는 데스크톱 앱 (Tkinter).

    cd desktop_version
    python3 app.py

mnist_cnn.pt 가 같은 폴더에 있어야 한다. 없으면 먼저 train.py 를 실행한다.
"""
import os
import tkinter as tk

import torch
from PIL import Image, ImageDraw

from model import 손글씨모델
from preprocess import 전처리, 정규화

여기 = os.path.dirname(os.path.abspath(__file__))
가중치파일 = os.path.join(여기, "mnist_cnn.pt")

칸크기 = 280
붓굵기 = 24


class 손글씨인식앱:
    def __init__(self, 창):
        self.창 = 창
        창.title("손글씨 숫자 인식기")
        창.resizable(False, False)

        self.모델 = 손글씨모델()
        self.모델.load_state_dict(torch.load(가중치파일, map_location="cpu"))
        self.모델.eval()

        self.그림칸 = tk.Canvas(창, width=칸크기, height=칸크기, bg="black",
                              highlightthickness=0, cursor="crosshair")
        self.그림칸.grid(row=0, column=0, padx=12, pady=12)

        오른쪽 = tk.Frame(창)
        오른쪽.grid(row=0, column=1, padx=(0, 12), pady=12, sticky="n")

        self.예측표시 = tk.Label(오른쪽, text="–", font=("", 64))
        self.예측표시.pack()
        self.후보표시 = tk.Label(오른쪽, text="칸에 숫자를 하나 써 보세요.",
                              font=("", 12), justify="left")
        self.후보표시.pack(pady=(8, 0))
        tk.Button(오른쪽, text="지우기", command=self.지우기).pack(pady=(16, 0))

        # 화면과 똑같은 그림을 따로 그려 두었다가 인식에 쓴다
        self.그림 = Image.new("L", (칸크기, 칸크기), 0)
        self.붓 = ImageDraw.Draw(self.그림)
        self.이전점 = None

        self.그림칸.bind("<Button-1>", self.누름)
        self.그림칸.bind("<B1-Motion>", self.끌기)
        self.그림칸.bind("<ButtonRelease-1>", self.뗌)

    def 누름(self, 사건):
        self.이전점 = (사건.x, 사건.y)
        반지름 = 붓굵기 / 2
        self.그림칸.create_oval(사건.x - 반지름, 사건.y - 반지름,
                              사건.x + 반지름, 사건.y + 반지름,
                              fill="white", outline="white")
        self.붓.ellipse([사건.x - 반지름, 사건.y - 반지름,
                        사건.x + 반지름, 사건.y + 반지름], fill=255)
        self.인식하기()

    def 끌기(self, 사건):
        if self.이전점 is None:
            return
        self.그림칸.create_line(*self.이전점, 사건.x, 사건.y, fill="white",
                              width=붓굵기, capstyle=tk.ROUND, smooth=True)
        self.붓.line([self.이전점, (사건.x, 사건.y)], fill=255, width=붓굵기)
        self.이전점 = (사건.x, 사건.y)
        self.인식하기()

    def 뗌(self, _사건):
        self.이전점 = None
        self.인식하기()

    def 지우기(self):
        self.그림칸.delete("all")
        self.붓.rectangle([0, 0, 칸크기, 칸크기], fill=0)
        self.예측표시.config(text="–")
        self.후보표시.config(text="칸에 숫자를 하나 써 보세요.")

    @torch.no_grad()
    def 인식하기(self):
        배열 = 전처리(self.그림)
        if 배열 is None:
            return
        입력 = torch.from_numpy(정규화(배열)).view(1, 1, 28, 28)
        확률 = torch.softmax(self.모델(입력), dim=1)[0]
        상위값, 상위숫자 = 확률.topk(3)
        self.예측표시.config(text=str(상위숫자[0].item()))
        self.후보표시.config(text="\n".join(
            f"{숫자.item()}   {값.item() * 100:5.1f}%"
            for 값, 숫자 in zip(상위값, 상위숫자)))


def main():
    if not os.path.exists(가중치파일):
        raise SystemExit("mnist_cnn.pt 가 없습니다. 먼저 train.py 를 실행하세요.")
    창 = tk.Tk()
    손글씨인식앱(창)
    창.mainloop()


if __name__ == "__main__":
    main()
