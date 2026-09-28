/* 그린 그림을 MNIST와 같은 모양으로 다듬는다.
   1) 숫자가 있는 영역만 잘라내기  2) 긴 변을 20px로 줄이기  3) 무게중심을 28x28 가운데로 옮기기 */
const Preprocess = (function () {
  const SIZE = 28;
  const BOX = 20;
  const INK = 0.08; // 이 밝기보다 밝으면 획으로 본다

  const work = document.createElement("canvas");
  work.width = SIZE;
  work.height = SIZE;
  const workCtx = work.getContext("2d", { willReadFrequently: true });

  function boundingBox(pixels, w, h) {
    let minX = w, minY = h, maxX = -1, maxY = -1;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        const v = (pixels[i] + pixels[i + 1] + pixels[i + 2]) / 765;
        if (v > INK) {
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }
    return maxX < 0 ? null : { minX: minX, minY: minY, maxX: maxX, maxY: maxY };
  }

  /* source: 흰 획 / 검은 바탕인 캔버스. 반환: 길이 784 Float32Array 또는 빈 그림이면 null */
  function fromCanvas(source) {
    const w = source.width;
    const h = source.height;
    const pixels = source.getContext("2d", { willReadFrequently: true })
      .getImageData(0, 0, w, h).data;

    const box = boundingBox(pixels, w, h);
    if (!box) return null;

    const bw = box.maxX - box.minX + 1;
    const bh = box.maxY - box.minY + 1;
    const scale = BOX / Math.max(bw, bh);
    const tw = Math.max(1, Math.round(bw * scale));
    const th = Math.max(1, Math.round(bh * scale));

    workCtx.fillStyle = "#000";
    workCtx.fillRect(0, 0, SIZE, SIZE);
    workCtx.imageSmoothingEnabled = true;
    workCtx.drawImage(source, box.minX, box.minY, bw, bh,
      Math.round((SIZE - tw) / 2), Math.round((SIZE - th) / 2), tw, th);

    const small = workCtx.getImageData(0, 0, SIZE, SIZE).data;
    const grid = new Float32Array(SIZE * SIZE);
    let mass = 0, mx = 0, my = 0;
    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const v = (small[(y * SIZE + x) * 4] + small[(y * SIZE + x) * 4 + 1] +
          small[(y * SIZE + x) * 4 + 2]) / 765;
        grid[y * SIZE + x] = v;
        mass += v;
        mx += v * x;
        my += v * y;
      }
    }
    if (mass === 0) return null;

    const shiftX = Math.round((SIZE - 1) / 2 - mx / mass);
    const shiftY = Math.round((SIZE - 1) / 2 - my / mass);
    if (shiftX === 0 && shiftY === 0) return grid;

    const out = new Float32Array(SIZE * SIZE);
    for (let y = 0; y < SIZE; y++) {
      const ny = y + shiftY;
      if (ny < 0 || ny >= SIZE) continue;
      for (let x = 0; x < SIZE; x++) {
        const nx = x + shiftX;
        if (nx < 0 || nx >= SIZE) continue;
        out[ny * SIZE + nx] = grid[y * SIZE + x];
      }
    }
    return out;
  }

  /* 모델에 실제로 들어가는 28x28을 화면에 보여 준다 */
  function drawTo(canvas, grid) {
    const ctx = canvas.getContext("2d");
    const image = ctx.createImageData(SIZE, SIZE);
    for (let i = 0; i < SIZE * SIZE; i++) {
      const v = grid ? Math.round(Math.min(1, Math.max(0, grid[i])) * 255) : 0;
      image.data[i * 4] = v;
      image.data[i * 4 + 1] = v;
      image.data[i * 4 + 2] = v;
      image.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
  }

  return { fromCanvas: fromCanvas, drawTo: drawTo, SIZE: SIZE };
})();
