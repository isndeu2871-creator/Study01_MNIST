/* 마우스와 손가락으로 숫자를 그리는 칸 */
function DrawingPad(canvas, onChange) {
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  let drawing = false;
  let last = null;

  function reset() {
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = 24;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    if (onChange) onChange();
  }

  function pointAt(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * (canvas.width / rect.width),
      y: (event.clientY - rect.top) * (canvas.height / rect.height)
    };
  }

  function start(event) {
    event.preventDefault();
    drawing = true;
    last = pointAt(event);
    // 점 하나만 찍어도 보이도록
    ctx.beginPath();
    ctx.arc(last.x, last.y, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fillStyle = "#fff";
    ctx.fill();
    if (onChange) onChange();
  }

  function move(event) {
    if (!drawing) return;
    event.preventDefault();
    const point = pointAt(event);
    ctx.beginPath();
    ctx.moveTo(last.x, last.y);
    ctx.lineTo(point.x, point.y);
    ctx.stroke();
    last = point;
    if (onChange) onChange();
  }

  function end() {
    if (!drawing) return;
    drawing = false;
    last = null;
    if (onChange) onChange();
  }

  canvas.addEventListener("pointerdown", start);
  canvas.addEventListener("pointermove", move);
  window.addEventListener("pointerup", end);
  window.addEventListener("pointercancel", end);

  reset();
  return { clear: reset };
}
