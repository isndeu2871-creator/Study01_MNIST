# desktop_version

PyTorch 로 학습하고 Tkinter 로 보여 주는 데스크톱 버전.

## 실행

**반드시 이 폴더 안에서 실행한다.** `train.py` 가 상대 경로 `data` 를 쓰기 때문이다.

```
cd desktop_version
python3 train.py          # 학습 → mnist_cnn.pt (5에포크, CPU 기준 약 2분)
python3 app.py            # 손글씨 인식 창 띄우기
python3 가중치내보내기.py      # 웹 버전이 읽을 가중치 내보내기
python3 검증데이터만들기.py    # 웹 버전 검사용 정답 데이터 만들기
```

윈도우에서는 `python3` 대신 `py` 를 쓴다.

## 필요한 것

`torch`, `torchvision`, `pillow`, `numpy`. `app.py` 는 `tkinter` 도 쓴다.
리눅스에서는 `python3-tk` 를 따로 깔아야 한다.

## 파일들의 연결 관계

```
preprocess.py ──▶ train.py   정규화 상수를 가져다 쓴다
              ──▶ app.py     그린 그림을 다듬는다
              ──▶ 검증데이터만들기.py

model.py ──▶ train.py, app.py, 가중치내보내기.py, 검증데이터만들기.py

train.py ──▶ mnist_cnn.pt ──▶ app.py
                          ──▶ 가중치내보내기.py ──▶ ../web_version/가중치.bin
                          ──▶ 검증데이터만들기.py ──▶ ../web_version/검증데이터.json
```

## 모델 구조

```
입력 1x28x28
 → 합성곱1(1→32, 3x3, 패딩 1) → ReLU → 최대풀링 2x2      (32x14x14)
 → 합성곱2(32→64, 3x3, 패딩 1) → ReLU → 최대풀링 2x2     (64x7x7)
 → 펼치기(3136) → 전결합1(3136→128) → ReLU → 드롭아웃(0.25)
 → 전결합2(128→10)
```

파라미터 421,642개. 내보낸 가중치는 1,686,568바이트.

드롭아웃은 `model.eval()` 상태에서 아무 일도 하지 않는다.
그래서 `web_version/모델.js` 에는 드롭아웃이 없다. 이것은 차이가 아니라 같은 계산이다.

`forward` 는 로짓을 낸다. 확률이 필요하면 `torch.softmax` 를 따로 건다.
학습에서는 `F.cross_entropy` 가 안에서 처리하므로 소프트맥스를 두 번 걸지 않는다.

## 자료

`data/` 에 MNIST 를 내려받는다. 저장소에 넣지 않는다(`.gitignore`).
`mnist_cnn.pt` 는 저장소에 넣는다. 받자마자 `app.py` 가 돌아가야 하기 때문이다.

## 고칠 때

층 구조나 전처리를 고치면 다음을 반드시 다시 돌린다.

```
python3 train.py
python3 가중치내보내기.py
python3 검증데이터만들기.py
```

그다음 `web_version/검증.html` 을 열어 검사 2개가 모두 통과하는지 본다.
