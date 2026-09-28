/* MNIST CNN 추론기 — 외부 라이브러리 없이 순수 자바스크립트로 계산한다.
   구조: conv(8@5x5) -> ReLU -> 최대풀링2 -> conv(16@3x3) -> ReLU -> 최대풀링2 -> 완전연결(400->10) */
const Model = (function () {
  let tensors = null;
  let meta = null;

  async function load(base) {
    base = base || "model/";
    const info = await fetch(base + "weights.json").then(function (r) {
      if (!r.ok) throw new Error("weights.json을 불러올 수 없습니다 (" + r.status + ")");
      return r.json();
    });
    const buffer = await fetch(base + "weights.bin").then(function (r) {
      if (!r.ok) throw new Error("weights.bin을 불러올 수 없습니다 (" + r.status + ")");
      return r.arrayBuffer();
    });
    tensors = {};
    info.tensors.forEach(function (t) {
      tensors[t.name] = new Float32Array(buffer, t.offset, t.count);
    });
    meta = info;
    return info;
  }

  /* 유효 합성곱(패딩 없음, 보폭 1). src 는 (cin, hin, hin)을 한 줄로 펼친 배열. */
  function conv(src, hin, cin, W, b, cout, k, relu) {
    const hout = hin - k + 1;
    const out = new Float32Array(cout * hout * hout);
    const plane = hout * hout;
    for (let o = 0; o < cout; o++) {
      const wOut = o * cin * k * k;
      const bias = b[o];
      for (let row = 0; row < hout; row++) {
        for (let col = 0; col < hout; col++) {
          let sum = bias;
          for (let c = 0; c < cin; c++) {
            const srcPlane = c * hin * hin;
            const wPlane = wOut + c * k * k;
            for (let ky = 0; ky < k; ky++) {
              const srcRow = srcPlane + (row + ky) * hin + col;
              const wRow = wPlane + ky * k;
              for (let kx = 0; kx < k; kx++) sum += src[srcRow + kx] * W[wRow + kx];
            }
          }
          out[o * plane + row * hout + col] = relu && sum < 0 ? 0 : sum;
        }
      }
    }
    return out;
  }

  /* 2x2 최대풀링 */
  function maxpool2(src, hin, channels) {
    const hout = hin >> 1;
    const out = new Float32Array(channels * hout * hout);
    for (let c = 0; c < channels; c++) {
      const inPlane = c * hin * hin;
      const outPlane = c * hout * hout;
      for (let row = 0; row < hout; row++) {
        for (let col = 0; col < hout; col++) {
          const i = inPlane + 2 * row * hin + 2 * col;
          let m = src[i];
          if (src[i + 1] > m) m = src[i + 1];
          if (src[i + hin] > m) m = src[i + hin];
          if (src[i + hin + 1] > m) m = src[i + hin + 1];
          out[outPlane + row * hout + col] = m;
        }
      }
    }
    return out;
  }

  function dense(x, W, b, nout) {
    const nin = x.length;
    const out = new Float32Array(nout);
    for (let o = 0; o < nout; o++) {
      let sum = b[o];
      const row = o * nin;
      for (let i = 0; i < nin; i++) sum += W[row + i] * x[i];
      out[o] = sum;
    }
    return out;
  }

  function softmax(z) {
    let max = -Infinity;
    for (let i = 0; i < z.length; i++) if (z[i] > max) max = z[i];
    let total = 0;
    const out = new Float32Array(z.length);
    for (let i = 0; i < z.length; i++) { out[i] = Math.exp(z[i] - max); total += out[i]; }
    for (let i = 0; i < z.length; i++) out[i] /= total;
    return out;
  }

  /* input: 길이 784(28x28), 값 0~1 */
  function predict(input) {
    if (!tensors) throw new Error("모델이 아직 준비되지 않았습니다.");
    let h = conv(input, 28, 1, tensors["conv1.W"], tensors["conv1.b"], 8, 5, true); // 8x24x24
    h = maxpool2(h, 24, 8);                                                        // 8x12x12
    h = conv(h, 12, 8, tensors["conv2.W"], tensors["conv2.b"], 16, 3, true);       // 16x10x10
    h = maxpool2(h, 10, 16);                                                       // 16x5x5
    return softmax(dense(h, tensors["fc.W"], tensors["fc.b"], 10));                // 10
  }

  function describe() { return meta; }

  return { load: load, predict: predict, describe: describe };
})();
