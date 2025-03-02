function checkNews() {
  let text = document.getElementById("newsText").value;
  fetch("/predict", {
    method: "POST",
    body: new URLSearchParams({ news_text: text }),
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById("result").innerText =
        "Prediction: " + data.prediction;
    });
}

function uploadFile() {
  let fileInput = document.getElementById("fileInput").files[0];
  let formData = new FormData();
  formData.append("file", fileInput);

  fetch("/upload", { method: "POST", body: formData })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById("fileResult").innerText =
        "Prediction: " + data.prediction;
    });
}
