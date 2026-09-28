"""MNIST CNN 학습 (numpy만 사용) -> 웹용 가중치(float32 bin + json) 내보내기.

구조: conv1(8@5x5) -> relu -> maxpool2 -> conv2(16@3x3) -> relu -> maxpool2 -> fc(400->10)
"""
import gzip, json, os, struct, urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
os.makedirs(DATA, exist_ok=True)
BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
FILES = {
    "train_x": "train-images-idx3-ubyte.gz",
    "train_y": "train-labels-idx1-ubyte.gz",
    "test_x": "t10k-images-idx3-ubyte.gz",
    "test_y": "t10k-labels-idx1-ubyte.gz",
}


def fetch(name):
    path = os.path.join(DATA, FILES[name])
    if not os.path.exists(path):
        urllib.request.urlretrieve(BASE + FILES[name], path)
    return path


def load_images(name):
    with gzip.open(fetch(name), "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        buf = f.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols).astype(np.float32) / 255.0


def load_labels(name):
    with gzip.open(fetch(name), "rb") as f:
        struct.unpack(">II", f.read(8))
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


def im2col(x, kh, kw):
    """x: (N,C,H,W) -> (N, C*kh*kw, OH*OW), stride 1, padding 0"""
    n, c, h, w = x.shape
    oh, ow = h - kh + 1, w - kw + 1
    s = x.strides
    view = np.lib.stride_tricks.as_strided(
        x, shape=(n, c, kh, kw, oh, ow),
        strides=(s[0], s[1], s[2], s[3], s[2], s[3]), writeable=False)
    return view.reshape(n, c * kh * kw, oh * ow)


class Conv:
    def __init__(self, cin, cout, k, rng):
        self.k = k
        self.cin, self.cout = cin, cout
        self.W = (rng.standard_normal((cout, cin * k * k)) * np.sqrt(2.0 / (cin * k * k))).astype(np.float32)
        self.b = np.zeros(cout, dtype=np.float32)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self.x_shape = x.shape
        n, c, h, w = x.shape
        self.oh, self.ow = h - self.k + 1, w - self.k + 1
        self.cols = im2col(x, self.k, self.k)                  # (N, C*k*k, OH*OW)
        out = np.einsum("oc,ncp->nop", self.W, self.cols, optimize=True) + self.b[None, :, None]
        return out.reshape(n, self.cout, self.oh, self.ow)

    def backward(self, dout):
        n = dout.shape[0]
        d = dout.reshape(n, self.cout, -1)                      # (N,O,P)
        self.dW = np.einsum("nop,ncp->oc", d, self.cols, optimize=True)
        self.db = d.sum(axis=(0, 2))
        dcols = np.einsum("oc,nop->ncp", self.W, d, optimize=True)
        dx = np.zeros(self.x_shape, dtype=np.float32)
        c = self.x_shape[1]
        dcols = dcols.reshape(n, c, self.k, self.k, self.oh, self.ow)
        for i in range(self.k):
            for j in range(self.k):
                dx[:, :, i:i + self.oh, j:j + self.ow] += dcols[:, :, i, j]
        return dx

    def step(self, lr, mom, wd):
        self.vW = mom * self.vW - lr * (self.dW + wd * self.W)
        self.vb = mom * self.vb - lr * self.db
        self.W += self.vW
        self.b += self.vb


class MaxPool2:
    def forward(self, x):
        n, c, h, w = x.shape
        self.x_shape = x.shape
        v = x.reshape(n, c, h // 2, 2, w // 2, 2)
        v = v.transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h // 2, w // 2, 4)
        self.arg = v.argmax(axis=-1)
        return v.max(axis=-1)

    def backward(self, dout):
        n, c, h, w = self.x_shape
        flat = np.zeros((n, c, h // 2, w // 2, 4), dtype=np.float32)
        idx = np.indices(self.arg.shape)
        flat[idx[0], idx[1], idx[2], idx[3], self.arg] = dout
        flat = flat.reshape(n, c, h // 2, w // 2, 2, 2).transpose(0, 1, 2, 4, 3, 5)
        return flat.reshape(n, c, h, w)


class Dense:
    def __init__(self, nin, nout, rng):
        self.W = (rng.standard_normal((nout, nin)) * np.sqrt(2.0 / nin)).astype(np.float32)
        self.b = np.zeros(nout, dtype=np.float32)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return x @ self.W.T + self.b

    def backward(self, dout):
        self.dW = dout.T @ self.x
        self.db = dout.sum(axis=0)
        return dout @ self.W

    step = Conv.step


def augment(batch, rng):
    """±2px 평행이동으로 손그림 입력에 강하게 만든다."""
    out = np.zeros_like(batch)
    dx = rng.integers(-2, 3, size=batch.shape[0])
    dy = rng.integers(-2, 3, size=batch.shape[0])
    for i in range(batch.shape[0]):
        src = batch[i]
        sx, sy = int(dx[i]), int(dy[i])
        xs0, xs1 = max(0, sx), min(28, 28 + sx)
        ys0, ys1 = max(0, sy), min(28, 28 + sy)
        out[i, ys0 - sy:ys1 - sy, xs0 - sx:xs1 - sx] = src[ys0:ys1, xs0:xs1]
    return out


def main():
    rng = np.random.default_rng(0)
    xtr = load_images("train_x"); ytr = load_labels("train_y")
    xte = load_images("test_x"); yte = load_labels("test_y")
    print(f"train {xtr.shape} test {xte.shape}", flush=True)

    c1 = Conv(1, 8, 5, rng)
    p1 = MaxPool2()
    c2 = Conv(8, 16, 3, rng)
    p2 = MaxPool2()
    fc = Dense(16 * 5 * 5, 10, rng)
    convs = [c1, c2, fc]

    def forward(x, train=True):
        h = c1.forward(x)
        m1 = h > 0; h = h * m1
        h = p1.forward(h)
        h = c2.forward(h)
        m2 = h > 0; h = h * m2
        h = p2.forward(h)
        flat = h.reshape(h.shape[0], -1)
        logits = fc.forward(flat)
        if train:
            forward.cache = (m1, m2, h.shape)
        return logits

    def backward(dlogits):
        m1, m2, hshape = forward.cache
        d = fc.backward(dlogits).reshape(hshape)
        d = p2.backward(d) * m2
        d = c2.backward(d)
        d = p1.backward(d) * m1
        c1.backward(d)

    def evaluate(x, y, bs=1000):
        correct = 0
        for i in range(0, len(x), bs):
            logits = forward(x[i:i + bs].reshape(-1, 1, 28, 28), train=False)
            correct += int((logits.argmax(1) == y[i:i + bs]).sum())
        return correct / len(x)

    epochs, bs, mom, wd = 8, 128, 0.9, 1e-4
    n = len(xtr)
    for ep in range(epochs):
        lr = 0.05 * (0.5 ** ep) if ep >= 4 else 0.05
        order = rng.permutation(n)
        total = 0.0
        for bi, i in enumerate(range(0, n - bs + 1, bs)):
            idx = order[i:i + bs]
            xb = augment(xtr[idx], rng).reshape(-1, 1, 28, 28)
            yb = ytr[idx]
            logits = forward(xb)
            logits -= logits.max(axis=1, keepdims=True)
            exp = np.exp(logits)
            prob = exp / exp.sum(axis=1, keepdims=True)
            total += float(-np.log(prob[np.arange(len(yb)), yb] + 1e-9).mean())
            d = prob
            d[np.arange(len(yb)), yb] -= 1.0
            backward((d / len(yb)).astype(np.float32))
            for layer in convs:
                layer.step(lr, mom, wd)
        acc = evaluate(xte, yte)
        print(f"epoch {ep + 1}/{epochs} loss {total / (n // bs):.4f} test_acc {acc * 100:.2f}%", flush=True)

    acc = evaluate(xte, yte)
    out_dir = os.path.join(HERE, "..", "model")
    os.makedirs(out_dir, exist_ok=True)
    blobs = [c1.W, c1.b, c2.W, c2.b, fc.W, fc.b]
    names = ["conv1.W", "conv1.b", "conv2.W", "conv2.b", "fc.W", "fc.b"]
    shapes = [[8, 1, 5, 5], [8], [16, 8, 3, 3], [16], [10, 400], [10]]
    manifest, offset = [], 0
    with open(os.path.join(out_dir, "weights.bin"), "wb") as f:
        for name, arr, shape in zip(names, blobs, shapes):
            data = np.ascontiguousarray(arr, dtype="<f4").tobytes()
            f.write(data)
            manifest.append({"name": name, "shape": shape, "offset": offset, "count": arr.size})
            offset += len(data)
    with open(os.path.join(out_dir, "weights.json"), "w") as f:
        json.dump({"arch": "conv8-pool-conv16-pool-fc10", "input": [1, 28, 28],
                   "test_accuracy": round(acc * 100, 2), "dtype": "float32",
                   "tensors": manifest}, f, indent=2)
    print(f"DONE test_acc {acc * 100:.2f}% bytes {offset}", flush=True)


if __name__ == "__main__":
    main()
