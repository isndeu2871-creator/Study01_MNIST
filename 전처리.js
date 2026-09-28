/* desktop_version/preprocess.py 의 3단계를 자바스크립트로 옮긴 것.

     1단계 여백 자르기      획이 있는 영역만 남긴다
     2단계 20x20 축소       비율을 유지하며 긴 변을 20픽셀로 줄인다
     3단계 무게중심 이동     밝기 무게중심을 28x28 칸 가운데로 옮긴다

   한 곳만 파이썬과 다르다. 파이썬은 PIL 의 LANCZOS 로 줄이는데 브라우저에서
   그것을 그대로 재현할 수 없어 면적 평균으로 줄인다. 캔버스의 drawImage 를 쓰지
   않는 이유는 브라우저마다 축소 방식이 달라 결과가 흔들리기 때문이다. */
const 전처리 = (function () {
  const 칸크기 = 28;
  const 숫자칸 = 20;
  const 획기준 = 0.08;  // preprocess.py 의 획기준과 같아야 한다

  /* 캔버스 → 밝기배열(0~1) */
  function 밝기읽기(캔버스) {
    const 너비 = 캔버스.width;
    const 높이 = 캔버스.height;
    const 화소 = 캔버스.getContext("2d", { willReadFrequently: true })
      .getImageData(0, 0, 너비, 높이).data;
    const 밝기 = new Float32Array(너비 * 높이);
    for (let i = 0; i < 너비 * 높이; i++) {
      밝기[i] = (화소[i * 4] + 화소[i * 4 + 1] + 화소[i * 4 + 2]) / 765;
    }
    return { 밝기: 밝기, 너비: 너비, 높이: 높이 };
  }

  /* 1단계 */
  function 여백자르기(밝기, 너비, 높이) {
    let 왼쪽 = 너비, 위쪽 = 높이, 오른쪽 = -1, 아래쪽 = -1;
    for (let 세로 = 0; 세로 < 높이; 세로++) {
      for (let 가로 = 0; 가로 < 너비; 가로++) {
        if (밝기[세로 * 너비 + 가로] > 획기준) {
          if (가로 < 왼쪽) 왼쪽 = 가로;
          if (가로 > 오른쪽) 오른쪽 = 가로;
          if (세로 < 위쪽) 위쪽 = 세로;
          if (세로 > 아래쪽) 아래쪽 = 세로;
        }
      }
    }
    if (오른쪽 < 0) return null;
    return { 왼쪽: 왼쪽, 위쪽: 위쪽, 너비: 오른쪽 - 왼쪽 + 1, 높이: 아래쪽 - 위쪽 + 1 };
  }

  /* 면적 평균 축소. 겹치는 넓이를 무게로 삼아 평균을 낸다. */
  function 면적평균축소(밝기, 원본너비, 영역, 새너비, 새높이) {
    const 결과 = new Float32Array(새너비 * 새높이);
    const 가로비 = 영역.너비 / 새너비;
    const 세로비 = 영역.높이 / 새높이;
    for (let 세로 = 0; 세로 < 새높이; 세로++) {
      const 위 = 세로 * 세로비;
      const 아래 = (세로 + 1) * 세로비;
      for (let 가로 = 0; 가로 < 새너비; 가로++) {
        const 왼 = 가로 * 가로비;
        const 오른 = (가로 + 1) * 가로비;
        let 합 = 0;
        let 무게합 = 0;
        for (let 원세로 = Math.floor(위); 원세로 < Math.min(영역.높이, Math.ceil(아래)); 원세로++) {
          const 세로무게 = Math.min(원세로 + 1, 아래) - Math.max(원세로, 위);
          if (세로무게 <= 0) continue;
          const 줄 = (영역.위쪽 + 원세로) * 원본너비 + 영역.왼쪽;
          for (let 원가로 = Math.floor(왼); 원가로 < Math.min(영역.너비, Math.ceil(오른)); 원가로++) {
            const 가로무게 = Math.min(원가로 + 1, 오른) - Math.max(원가로, 왼);
            if (가로무게 <= 0) continue;
            const 무게 = 세로무게 * 가로무게;
            합 += 밝기[줄 + 원가로] * 무게;
            무게합 += 무게;
          }
        }
        결과[세로 * 새너비 + 가로] = 무게합 > 0 ? 합 / 무게합 : 0;
      }
    }
    return 결과;
  }

  /* 3단계 */
  function 무게중심이동(칸) {
    let 무게 = 0;
    let 가로합 = 0;
    let 세로합 = 0;
    for (let 세로 = 0; 세로 < 칸크기; 세로++) {
      for (let 가로 = 0; 가로 < 칸크기; 가로++) {
        const 값 = 칸[세로 * 칸크기 + 가로];
        무게 += 값;
        가로합 += 값 * 가로;
        세로합 += 값 * 세로;
      }
    }
    if (무게 === 0) return 칸;
    const 가운데 = (칸크기 - 1) / 2;
    // 파이썬과 같은 반올림 규칙을 쓴다 (floor(x + 0.5))
    const 옮길가로 = Math.floor(가운데 - 가로합 / 무게 + 0.5);
    const 옮길세로 = Math.floor(가운데 - 세로합 / 무게 + 0.5);
    if (옮길가로 === 0 && 옮길세로 === 0) return 칸;
    const 옮긴칸 = new Float32Array(칸크기 * 칸크기);
    for (let 세로 = 0; 세로 < 칸크기; 세로++) {
      const 새세로 = 세로 + 옮길세로;
      if (새세로 < 0 || 새세로 >= 칸크기) continue;
      for (let 가로 = 0; 가로 < 칸크기; 가로++) {
        const 새가로 = 가로 + 옮길가로;
        if (새가로 < 0 || 새가로 >= 칸크기) continue;
        옮긴칸[새세로 * 칸크기 + 새가로] = 칸[세로 * 칸크기 + 가로];
      }
    }
    return 옮긴칸;
  }

  /* 밝기배열 → 28x28 밝기배열(0~1). 빈 그림이면 null. */
  function 밝기배열에서(밝기, 너비, 높이) {
    const 영역 = 여백자르기(밝기, 너비, 높이);
    if (!영역) return null;

    const 비율 = 숫자칸 / Math.max(영역.너비, 영역.높이);
    const 새너비 = Math.max(1, Math.floor(영역.너비 * 비율 + 0.5));
    const 새높이 = Math.max(1, Math.floor(영역.높이 * 비율 + 0.5));
    const 줄인것 = 면적평균축소(밝기, 너비, 영역, 새너비, 새높이);

    const 칸 = new Float32Array(칸크기 * 칸크기);
    const 붙일가로 = (칸크기 - 새너비) >> 1;   // 파이썬의 // 와 같다
    const 붙일세로 = (칸크기 - 새높이) >> 1;
    for (let 세로 = 0; 세로 < 새높이; 세로++) {
      for (let 가로 = 0; 가로 < 새너비; 가로++) {
        칸[(붙일세로 + 세로) * 칸크기 + 붙일가로 + 가로] = 줄인것[세로 * 새너비 + 가로];
      }
    }
    return 무게중심이동(칸);
  }

  function 캔버스에서(캔버스) {
    const 읽은것 = 밝기읽기(캔버스);
    return 밝기배열에서(읽은것.밝기, 읽은것.너비, 읽은것.높이);
  }

  /* 28x28 배열을 캔버스에 그려 눈으로 확인할 수 있게 한다 (검증.html 에서 쓴다) */
  function 그리기(캔버스, 칸) {
    const 그리개 = 캔버스.getContext("2d");
    const 그림 = 그리개.createImageData(칸크기, 칸크기);
    for (let i = 0; i < 칸크기 * 칸크기; i++) {
      const 값 = 칸 ? Math.round(Math.min(1, Math.max(0, 칸[i])) * 255) : 0;
      그림.data[i * 4] = 값;
      그림.data[i * 4 + 1] = 값;
      그림.data[i * 4 + 2] = 값;
      그림.data[i * 4 + 3] = 255;
    }
    그리개.putImageData(그림, 0, 0);
  }

  return { 캔버스에서: 캔버스에서, 밝기배열에서: 밝기배열에서, 그리기: 그리기, 칸크기: 칸크기 };
})();
