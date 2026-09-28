/* 마우스와 손가락으로 숫자를 그리는 칸.
   desktop_version/app.py 의 Tkinter 캔버스와 같은 크기(280x280), 같은 붓굵기(24)를 쓴다.
   데스크톱에 없는 것은 손가락 입력뿐이다. */
function 그림판만들기(캔버스, 바뀔때마다) {
  const 그리개 = 캔버스.getContext("2d", { willReadFrequently: true });
  const 붓굵기 = 24;
  let 그리는중 = false;
  let 이전점 = null;

  function 지우기() {
    그리개.fillStyle = "#000";
    그리개.fillRect(0, 0, 캔버스.width, 캔버스.height);
    그리개.strokeStyle = "#fff";
    그리개.fillStyle = "#fff";
    그리개.lineWidth = 붓굵기;
    그리개.lineCap = "round";
    그리개.lineJoin = "round";
    if (바뀔때마다) 바뀔때마다();
  }

  function 점찾기(사건) {
    const 자리 = 캔버스.getBoundingClientRect();
    return {
      x: (사건.clientX - 자리.left) * (캔버스.width / 자리.width),
      y: (사건.clientY - 자리.top) * (캔버스.height / 자리.height)
    };
  }

  function 누름(사건) {
    사건.preventDefault();
    그리는중 = true;
    이전점 = 점찾기(사건);
    그리개.beginPath();                       // 점 하나만 찍어도 보이도록
    그리개.arc(이전점.x, 이전점.y, 붓굵기 / 2, 0, Math.PI * 2);
    그리개.fill();
    if (바뀔때마다) 바뀔때마다();
  }

  function 끌기(사건) {
    if (!그리는중) return;
    사건.preventDefault();
    const 점 = 점찾기(사건);
    그리개.beginPath();
    그리개.moveTo(이전점.x, 이전점.y);
    그리개.lineTo(점.x, 점.y);
    그리개.stroke();
    이전점 = 점;
    if (바뀔때마다) 바뀔때마다();
  }

  function 뗌() {
    if (!그리는중) return;
    그리는중 = false;
    이전점 = null;
    if (바뀔때마다) 바뀔때마다();
  }

  캔버스.addEventListener("pointerdown", 누름);
  캔버스.addEventListener("pointermove", 끌기);
  window.addEventListener("pointerup", 뗌);
  window.addEventListener("pointercancel", 뗌);

  지우기();
  return { 지우기: 지우기 };
}
