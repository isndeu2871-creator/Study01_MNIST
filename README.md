# Study01_MNIST

MNIST 손글씨 숫자 인식기 — 브라우저에서 바로 돌아가는 웹 버전입니다.

배포된 주소: https://isndeu2871-creator.github.io/Study01_MNIST/

## 구성

| 경로 | 설명 |
| --- | --- |
| `index.html` | 페이지 화면. 맨 위에 학번과 이름이 보입니다. |
| `style.css` | 화면 꾸미기 |
| `src/model.js` | CNN 추론기 (외부 라이브러리 없음) |
| `src/preprocess.js` | 그린 그림을 MNIST 모양(28×28)으로 다듬기 |
| `src/pad.js` | 마우스·손가락으로 그리는 칸 |
| `src/app.js` | 화면 연결 |
| `model/weights.bin`, `model/weights.json` | 직접 학습시킨 가중치 |
| `training/train_cnn.py` | 가중치를 만든 학습 코드 (numpy만 사용) |
| `training/check_js_matches_numpy.js` | 자바스크립트 추론이 학습 결과와 같은지 확인하는 검사 |

## 모델

```
입력 1×28×28
 → 합성곱 8@5×5 → ReLU → 최대풀링 2×2      (8×12×12)
 → 합성곱 16@3×3 → ReLU → 최대풀링 2×2     (16×5×5)
 → 완전연결 400 → 10 → 소프트맥스
```

MNIST 6만 장으로 8에포크 학습했고, 시험자료 1만 장 정확도는 `model/weights.json`의
`test_accuracy` 값에 적혀 있습니다. 손으로 쓴 글씨에도 잘 맞도록 학습할 때 ±2픽셀
평행이동을 섞었습니다.

## 직접 돌려 보기

웹 버전은 그냥 파일을 열면 `fetch`가 막히므로 간단한 서버를 띄웁니다.

```
python3 -m http.server 8000
```

그다음 브라우저에서 http://localhost:8000 을 엽니다.

가중치를 다시 만들려면:

```
pip install numpy
python3 training/train_cnn.py
```

## 참고

수업 자료: [logistex/Study01_MNIST](https://github.com/logistex/Study01_MNIST)
