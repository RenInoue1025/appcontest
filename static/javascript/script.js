document.getElementById("cameraBtn").addEventListener("click", () => {
  // 仮で改善率をランダム表示
  const rate = Math.floor(Math.random() * 100);
  document.getElementById("improveRate").textContent = rate + "%";

  // キャラクターコメントを切り替え
  const msg = rate > 70 ? "いい感じ！この調子で続けましょう！" 
              : rate > 40 ? "もう少し頑張って！"
              : "ちゃぶ台返し！！😡";
  document.getElementById("characterMessage").textContent = msg;
});

document.getElementById("logoutBtn").addEventListener("click", () => {
  alert("ログアウトしました（仮実装）");
  // 実際はサーバー側でセッション削除
});
