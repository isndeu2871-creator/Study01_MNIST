"""학습된 가중치로 MNIST 시험자료 앞 50장을 예측해 검사용 fixture를 만든다.

자바스크립트 추론기(src/model.js)가 같은 답을 내는지 확인하는 데 쓴다.
"""
import base64, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import train_cnn as T  # noqa: E402

N = 50


def main():
    x = T.load_images("test_x")[:N]
    y = T.load_labels("test_y")[:N]

    meta = json.load(open(os.path.join(HERE, "..", "model", "weights.json")))
    raw = open(os.path.join(HERE, "..", "model", "weights.bin"), "rb").read()
    w = {}
    for t in meta["tensors"]:
        w[t["name"]] = np.frombuffer(raw, dtype="<f4", count=t["count"],
                                     offset=t["offset"]).copy()

    # 웹에서는 0~255 정수로 전달하므로 같은 값으로 맞춰서 계산한다
    q = np.round(x * 255).astype(np.uint8)
    xin = (q.astype(np.float32) / 255.0).reshape(N, 1, 28, 28)

    c1 = T.Conv(1, 8, 5, np.random.default_rng(0))
    c1.W = w["conv1.W"].reshape(8, 25); c1.b = w["conv1.b"]
    c2 = T.Conv(8, 16, 3, np.random.default_rng(0))
    c2.W = w["conv2.W"].reshape(16, 72); c2.b = w["conv2.b"]
    fc = T.Dense(400, 10, np.random.default_rng(0))
    fc.W = w["fc.W"].reshape(10, 400); fc.b = w["fc.b"]

    h = np.maximum(c1.forward(xin), 0)
    h = T.MaxPool2().forward(h)
    h = np.maximum(c2.forward(h), 0)
    h = T.MaxPool2().forward(h)
    logits = fc.forward(h.reshape(N, -1))
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    prob = exp / exp.sum(axis=1, keepdims=True)

    out = {
        "note": "training/make_fixtures.py 가 만든 파일. src/model.js 검사에 쓴다.",
        "count": N,
        "images_base64_uint8_28x28": base64.b64encode(q.tobytes()).decode(),
        "labels": y.tolist(),
        "numpy_argmax": prob.argmax(axis=1).tolist(),
        "numpy_probabilities": [[round(float(v), 6) for v in row] for row in prob],
    }
    path = os.path.join(HERE, "fixtures.json")
    json.dump(out, open(path, "w"), ensure_ascii=False)
    agree = int((prob.argmax(axis=1) == y).sum())
    print(f"fixtures.json 작성 완료 — 앞 {N}장 중 {agree}장 정답")


if __name__ == "__main__":
    main()
