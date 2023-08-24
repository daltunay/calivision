// Function to update minimum detection confidence value
function updateMinDetectionConfidence(value) {
    document.getElementById("min_detection_value").innerHTML = "<code>" + value + "</code>";
}

// Function to update minimum tracking confidence value
function updateMinTrackingConfidence(value) {
    document.getElementById("min_tracking_value").innerHTML = "<code>" + value + "</code>";
}

// Function to update button state based on input elements
function updateButtonState() {
    const sourceTypeInput = document.querySelector('input[name="source_type"]:checked');
    const actionButton = document.getElementById("actionButton");
    const webcamChecked = document.querySelector('input[name="webcam"]:checked');
    const videoUploadInput = document.querySelector('input[name="video_upload"]');

    if ((sourceTypeInput && sourceTypeInput.value === "webcam" && webcamChecked) ||
        (sourceTypeInput && sourceTypeInput.value === "upload" && videoUploadInput.files.length > 0) ||
        (actionButton.value == "END POSE ESTIMATION")) {
        actionButton.classList.remove("button-disabled");
        actionButton.classList.add("button");
        actionButton.removeAttribute("disabled");
    } else {
        actionButton.classList.remove("button");
        actionButton.classList.add("button-disabled");
        actionButton.setAttribute("disabled", "disabled");
    }
}


setInterval(updateButtonState, 1);

// Function to handle change in video source
function handleSourceChange() {
    const sourceTypeInput = document.querySelector('input[name="source_type"]:checked');
    const sourceType = sourceTypeInput ? sourceTypeInput.value : null;
    const webcamContainer = document.getElementById("webcam-container");
    const uploadContainer = document.getElementById("upload-container");

    // Handle action button clickability
    updateButtonState();

    // Handle source types clickability
    webcamContainer.classList.toggle("selected", sourceType === "webcam");
    webcamContainer.classList.toggle("unclickable", sourceType !== "webcam");
    uploadContainer.classList.toggle("selected", sourceType === "upload");
    uploadContainer.classList.toggle("unclickable", sourceType !== "upload");
}

// Initially, make both containers unclickable
const webcamContainer = document.getElementById("webcam-container");
const uploadContainer = document.getElementById("upload-container");
webcamContainer.classList.add("unclickable");
uploadContainer.classList.add("unclickable");

// Add event listeners for input changes
const sourceTypeInputs = document.querySelectorAll('input[name="source_type"]');
const webcamInputs = document.querySelectorAll('input[name="webcam"]');
const videoUploadInput = document.querySelector('input[name="video_upload"]');

sourceTypeInputs.forEach(input => {
    input.addEventListener("change", handleSourceChange);
});
webcamInputs.forEach(input => {
    input.addEventListener("change", handleSourceChange);
});
videoUploadInput.addEventListener("change", handleSourceChange);

// Trigger initial state
handleSourceChange();
