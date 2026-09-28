/* src/model.js 의 자바스크립트 추론이 학습(numpy) 결과와 같은 답을 내는지 확인한다.
   실행: node training/check_js_matches_numpy.js  */
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const root = path.join(__dirname, "..");

// model.js 는 브라우저의 fetch 를 쓰므로 파일에서 읽어 오는 fetch 를 끼워 넣는다
function fakeFetch(target) {
  const file = path.join(root, target);
  return Promise.resolve({
    ok: fs.existsSync(file),
    status: fs.existsSync(file) ? 200 : 404,
    json: function () { return Promise.resolve(JSON.parse(fs.readFileSync(file, "utf8"))); },
    arrayBuffer: function () {
      const buffer = fs.readFileSync(file);
      return Promise.resolve(buffer.buffer.slice(buffer.byteOffset,
        buffer.byteOffset + buffer.byteLength));
    }
  });
}

const context = vm.createContext({
  fetch: fakeFetch, console: console, Math: Math, Error: Error,
  Promise: Promise, Float32Array: Float32Array, Infinity: Infinity
});
// vm 안의 const 는 바깥에서 안 보이므로 전역에 붙여 준다
vm.runInContext(fs.readFileSync(path.join(root, "src", "model.js"), "utf8") +
  "\nthis.Model = Model;", context);

const fixtures = JSON.parse(fs.readFileSync(path.join(__dirname, "fixtures.json"), "utf8"));
const images = Buffer.from(fixtures.images_base64_uint8_28x28, "base64");

context.Model.load("model/").then(function () {
  let mismatched = 0;
  let worst = 0;
  let correct = 0;
  for (let n = 0; n < fixtures.count; n++) {
    const input = new Float32Array(784);
    for (let i = 0; i < 784; i++) input[i] = images[n * 784 + i] / 255;
    const probabilities = context.Model.predict(input);
    let best = 0;
    for (let i = 1; i < 10; i++) if (probabilities[i] > probabilities[best]) best = i;
    if (best !== fixtures.numpy_argmax[n]) mismatched++;
    if (best === fixtures.labels[n]) correct++;
    for (let i = 0; i < 10; i++) {
      const diff = Math.abs(probabilities[i] - fixtures.numpy_probabilities[n][i]);
      if (diff > worst) worst = diff;
    }
  }
  console.log("표본 " + fixtures.count + "장");
  console.log("numpy 와 예측이 다른 장수: " + mismatched);
  console.log("확률 최대 오차: " + worst.toExponential(3));
  console.log("정답 맞힌 장수: " + correct + "/" + fixtures.count);
  if (mismatched !== 0 || worst > 1e-4) {
    console.error("실패: 자바스크립트 추론이 학습 결과와 일치하지 않습니다.");
    process.exit(1);
  }
  console.log("통과: 자바스크립트 추론이 학습 결과와 일치합니다.");
}).catch(function (error) {
  console.error(error);
  process.exit(1);
});
