# Study01_MNIST

손글씨 숫자(0~9) 인식 프로그램. 같은 모델을 데스크톱과 웹 두 가지로 만들었습니다.

웹 버전 배포 주소: https://isndeu2871-creator.github.io/Study01_MNIST/

## 구성

```
desktop_version/     PyTorch + Tkinter. 학습도 여기서 한다
  model.py           합성곱2 + 최대풀링2 + 전결합2 (파라미터 421,642개)
  preprocess.py      전처리 3단계
  train.py           학습 → mnist_cnn.pt
  app.py             손글씨 인식 창
  가중치내보내기.py      .pt → 웹용 가중치
  검증데이터만들기.py    웹 검사용 정답 데이터
  mnist_cnn.pt       학습된 가중치 (받자마자 실행되도록 넣어 둠)

web_version/         외부 라이브러리 없는 순수 자바스크립트
  index.html 스타일.css 그림판.js 전처리.js 모델.js 앱.js
  검증.html          파이썬 결과와 대조하는 검사 페이지
  가중치.bin 가중치정보.json
```

## 실행

데스크톱 버전 — 반드시 `desktop_version` 안에서 실행합니다.

```
cd desktop_version
python3 app.py
```

필요한 것: `torch`, `torchvision`, `pillow`, `numpy`, `tkinter`.
가중치를 다시 만들려면 `python3 train.py` 를 먼저 돌립니다.

웹 버전 — `file://` 로는 열리지 않습니다.

```
cd web_version
python3 -m http.server 8000
```

윈도우에서는 `python3` 대신 `py` 를 씁니다.

## 모델

```
입력 1x28x28
 → 합성곱1(1→32, 3x3, 패딩 1) → ReLU → 최대풀링 2x2      (32x14x14)
 → 합성곱2(32→64, 3x3, 패딩 1) → ReLU → 최대풀링 2x2     (64x7x7)
 → 펼치기(3136) → 전결합1(3136→128) → ReLU → 드롭아웃
 → 전결합2(128→10) → 소프트맥스
```

파라미터 421,642개. 내보낸 가중치 1,686,568바이트.

## 확인한 것

| 항목 | 기준 | 실측 |
| --- | --- | --- |
| MNIST 시험자료 정확도 | — | 99.21% |
| 순전파 일치 (파이썬 ↔ 자바스크립트) | 최대 절대차 1e-4 이하 | 2.42e-7 |
| 전체 정확도 (전처리 포함, 200장) | 97% 이상 | 98.5% |

`web_version/검증.html` 에서 직접 돌려 볼 수 있습니다.
먼저 `desktop_version` 에서 `python3 검증데이터만들기.py` 를 실행하세요.

## 참고

수업 자료: [logistex/Study01_MNIST](https://github.com/logistex/Study01_MNIST)
