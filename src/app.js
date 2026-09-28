/* 화면 연결: 그리면 곧바로 예측해서 보여 준다 */
(function () {
  const pad = document.getElementById("pad");
  const preview = document.getElementById("preview");
  const answer = document.getElementById("answer");
  const confidence = document.getElementById("confidence");
  const status = document.getElementById("status");
  const barList = document.getElementById("bars");
  const modelInfo = document.getElementById("model-info");

  const rows = [];
  for (let digit = 0; digit <= 9; digit++) {
    const li = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = digit;
    const track = document.createElement("span");
    track.className = "track";
    const fill = document.createElement("span");
    fill.className = "fill";
    track.appendChild(fill);
    const pct = document.createElement("span");
    pct.className = "pct";
    pct.textContent = "0%";
    li.appendChild(label);
    li.appendChild(track);
    li.appendChild(pct);
    barList.appendChild(li);
    rows.push({ li: li, fill: fill, pct: pct });
  }

  let ready = false;
  let queued = false;

  function show(probabilities) {
    let best = 0;
    for (let i = 1; i < 10; i++) if (probabilities[i] > probabilities[best]) best = i;
    answer.textContent = best;
    confidence.textContent = "확신 " + (probabilities[best] * 100).toFixed(1) + "%";
    rows.forEach(function (row, digit) {
      row.fill.style.width = (probabilities[digit] * 100).toFixed(1) + "%";
      row.pct.textContent = (probabilities[digit] * 100).toFixed(1) + "%";
      row.li.classList.toggle("top", digit === best);
    });
  }

  function clearResult() {
    answer.textContent = "–";
    confidence.textContent = "";
    rows.forEach(function (row) {
      row.fill.style.width = "0%";
      row.pct.textContent = "0%";
      row.li.classList.remove("top");
    });
    Preprocess.drawTo(preview, null);
  }

  function run() {
    if (!ready) return;
    const input = Preprocess.fromCanvas(pad);
    Preprocess.drawTo(preview, input);
    if (!input) {
      clearResult();
      status.textContent = "칸에 숫자를 하나 써 보세요.";
      return;
    }
    show(Model.predict(input));
    status.textContent = "";
  }

  function schedule() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(function () {
      queued = false;
      run();
    });
  }

  const drawing = DrawingPad(pad, schedule);
  document.getElementById("clear").addEventListener("click", function () {
    drawing.clear();
    clearResult();
    status.textContent = "칸에 숫자를 하나 써 보세요.";
  });

  clearResult();

  Model.load("model/").then(function (meta) {
    ready = true;
    status.textContent = "칸에 숫자를 하나 써 보세요.";
    modelInfo.textContent = "직접 학습시킨 모델입니다. 구조 " + meta.arch +
      ", MNIST 시험자료 정확도 " + meta.test_accuracy + "%.";
    run();
  }).catch(function (error) {
    status.textContent = "모델을 불러오지 못했습니다: " + error.message;
    modelInfo.textContent = "";
  });
})();
