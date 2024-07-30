document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('predictForm');
    form.addEventListener('submit', function(event) {
        event.preventDefault();

        const formData = new FormData(form);
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert(data.error);
            } else {
                // Display the plots
                document.getElementById('actualVsPredictedPlot').src = data.actual_vs_predicted_plot;
                document.getElementById('futurePredictionsPlot').src = data.future_predictions_plot;

                // Display metrics
                document.getElementById('accuracy').innerText = 'Accuracy: ' + data.accuracy + '%';
                document.getElementById('rmse').innerText = 'RMSE: ' + data.rmse;
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    });
});
