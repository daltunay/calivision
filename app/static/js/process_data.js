function updateMetrics() {
    const featureTypeDropdown = document.getElementById("feature_type");
    const metricDropdown = document.getElementById("metric");

    const validMetrics = {
        joints: ["l1", "l2", "dtw", "lcss"],
        angles: ["l1", "l2", "dtw", "lcss"],
        fourier: ["l1", "l2", "emd"]
    };

    const selectedFeatureType = featureTypeDropdown.value;
    const availableMetrics = validMetrics[selectedFeatureType];

    for (let option of metricDropdown.options) {
        if (availableMetrics.includes(option.value)) {
            option.disabled = false;
        } else {
            option.disabled = true;
        }
    }
}

function updateParameters() {
    const modelTypeDropdown = document.getElementById("model_type");
    const featureTypeDropdown = document.getElementById("feature_type");
    const kInput = document.getElementById("k");
    const metricDropdown = document.getElementById("metric");
    const weightsDropdown = document.getElementById("weights");

    const kInputLabel = document.querySelector('label[for="k"]');
    const metricInputLabel = document.querySelector('label[for="metric"]');
    const weightsInputLabel = document.querySelector('label[for="weights"]');
    const featureTypeInputLabel = document.querySelector('label[for="feature_type"]');

    const selectedModelType = modelTypeDropdown.value;
    const selectedFeatureType = featureTypeDropdown.value;

    // Toggle classes to enable/disable parameters and their labels
    const disableElements = selectedModelType === "lstm";
    kInput.disabled = disableElements;
    kInputLabel.classList.toggle("disabled", disableElements);
    metricDropdown.disabled = disableElements;
    metricInputLabel.classList.toggle("disabled", disableElements);
    weightsDropdown.disabled = disableElements;
    weightsInputLabel.classList.toggle("disabled", disableElements);

    // If model type is lstm, limit feature type options
    if (selectedModelType === "lstm") {
        // Set default feature type if "fourier" is selected
        if (selectedFeatureType === "fourier") {
            featureTypeDropdown.value = "angles";
        }
        // Disable "fourier" as an option
        const featureTypeOptions = featureTypeDropdown.querySelectorAll("option");
        featureTypeOptions.forEach(option => {
            if (option.value === "fourier") {
                option.disabled = true;
            } else {
                option.disabled = false;
            }
        });
        // Update the feature_type input label style
        featureTypeInputLabel.classList.toggle("disabled", true);
    } else {
        // Enable all feature type options and labels
        const featureTypeOptions = featureTypeDropdown.querySelectorAll("option");
        featureTypeOptions.forEach(option => {
            option.disabled = false;
        });
        featureTypeInputLabel.classList.toggle("disabled", false);
    }

    // Call updateMetrics to reflect changes in feature type
    updateMetrics();
}
