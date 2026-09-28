/* 화면 연결. 그리면 곧바로 인식해서 상위 3개 후보를 보여 준다.
   보여 주는 내용은 desktop_version/app.py 와 같다. */
(function () {
  const 그림칸 = document.getElementById("그림칸");
  const 예측표시 = document.getElementById("예측");
  const 후보표시 = document.getElementById("후보");
  const 안내표시 = document.getElementById("안내");
  const 모델설명 = document.getElementById("모델설명");

  let 준비됨 = false;
  let 예약됨 = false;

  function 비우기() {
    예측표시.textContent = "–";
    후보표시.textContent = "";
  }

  function 인식하기() {
    if (!준비됨) return;
    const 밝기배열 = 전처리.캔버스에서(그림칸);
    if (!밝기배열) {
      비우기();
      안내표시.textContent = "칸에 숫자를 하나 써 보세요.";
      return;
    }
    const 확률 = 모델.예측(밝기배열);
    const 순서 = Array.from(확률.keys()).sort(function (가, 나) {
      return 확률[나] - 확률[가];
    });
    예측표시.textContent = 순서[0];
    후보표시.textContent = 순서.slice(0, 3).map(function (숫자) {
      return 숫자 + "   " + (확률[숫자] * 100).toFixed(1) + "%";
    }).join("\n");
    안내표시.textContent = "";
  }

  function 예약하기() {
    if (예약됨) return;
    예약됨 = true;
    requestAnimationFrame(function () {
      예약됨 = false;
      인식하기();
    });
  }

  const 그림판 = 그림판만들기(그림칸, 예약하기);
  document.getElementById("지우기").addEventListener("click", function () {
    그림판.지우기();
    비우기();
    안내표시.textContent = "칸에 숫자를 하나 써 보세요.";
  });

  비우기();

  모델.불러오기("").then(function (정보) {
    준비됨 = true;
    안내표시.textContent = "칸에 숫자를 하나 써 보세요.";
    모델설명.textContent = "구조 " + 정보.구조 + " · 파라미터 " +
      정보.파라미터수.toLocaleString() + "개 · 데스크톱 버전과 같은 가중치를 씁니다.";
    인식하기();
  }).catch(function (오류) {
    안내표시.textContent = "모델을 불러오지 못했습니다: " + 오류.message;
  });
})();
