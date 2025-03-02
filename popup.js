function checkNews() {
  let text = document.getElementById("newsText").value;

  fetch("https://your-deployed-app.com/predict", {
    // Change URL to your live app
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
