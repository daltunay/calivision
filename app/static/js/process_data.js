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