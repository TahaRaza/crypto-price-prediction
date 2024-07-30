document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("predictForm");

    form.addEventListener("submit", function (e) {
        e.preventDefault();

        const formData = new FormData(form);
        const data = {
            symbol: formData.get("symbol"),
            model_option: formData.get("model_option")
        };

        fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
            } else {
                document.getElementById("actualVsPredictedPlot").src = data.actual_vs_predicted_plot;
                document.getElementById("futurePredictionsPlot").src = data.future_predictions_plot;
                document.getElementById("accuracy").textContent = "Accuracy: " + data.accuracy + "%";
                document.getElementById("rmse").textContent = "RMSE: " + data.rmse;
            }
        })
        .catch(error => {
            console.error("Error:", error);
        });
    });
});
