let selectedImageName = '';
const CONFIDENCE_THRESHOLD = 0.80; // 80% (For CONFIRMED status)
const AMBIGUOUS_THRESHOLD = 0.50;  // 50% (For WARNING status)

const COLOR_MAP = {
    CONFIRMED: '#10b981', 
    WARNING: '#f59e0b',  
    UNRELIABLE: '#ef4444',
    BAR: '#3b82f6'   
};

// pagination
document.addEventListener("DOMContentLoaded", function () {
  const itemsPerPage = 9;
  const items = document.querySelectorAll(".img-item");
  const pager = document.getElementById("pager");
  let currentPage = 1;
  const totalPages = Math.ceil(items.length / itemsPerPage);

  function showPage(page) {
    currentPage = page;

    items.forEach(item => item.style.display = "none");

    const start = (page - 1) * itemsPerPage;
    const end = start + itemsPerPage;

    for (let i = start; i < end && i < items.length; i++) {
      items[i].style.display = "block";
    }

    document.querySelectorAll(".page-item").forEach(li => li.classList.remove("active"));
    document.getElementById("page-" + page)?.classList.add("active");
  }

  function buildPager() {
    pager.innerHTML = "";
    for (let i = 1; i <= totalPages; i++) {
      pager.innerHTML += `
        <li class="page-item" id="page-${i}">
          <a class="page-link" href="javascript:void(0)">${i}</a>
        </li>`;
    }

    document.querySelectorAll(".page-link").forEach((link, index) => {
      link.addEventListener("click", () => showPage(index + 1));
    });
  }

  if (items.length > 0) {
    buildPager();
    showPage(1);
  } else {
    pager.innerHTML = "<li class='text-muted'>No images found</li>";
  }
});

// Image List function
function selectImage(imgName) {
    selectedImageName = imgName;
    document.getElementById('selectedImage').src = '/static/images/' + imgName;
    clearCharts();
    clearTop3();

    // Remove previous highlights
    document.querySelectorAll('#imageList .img-thumb').forEach(img => {
        img.classList.remove('selected');
    });

    // Highlight the clicked image
    document.querySelectorAll('#imageList .img-thumb').forEach(img => {
        if (img.dataset.img === imgName) {
            img.classList.add('selected');
        }
    });
}

// Clear gauge chart
function clearCharts() {
    const gaugeContainer = document.getElementById('gaugeChart');
    const graph1 = document.getElementById('graph1');
    const graph2 = document.getElementById('graph2');
    if (gaugeContainer) {
        gaugeContainer.innerHTML = ''; // remove any existing gauge
    }
    if (graph1) {
        graph1.innerHTML = ''; // remove any existing graph1
    }
    if (graph2) {
        graph2.innerHTML = ''; // remove any existing graph2
    }
}

// Clear gauge chart
function clearTop3() {
    const wrapper = document.getElementById('topClassesWrapper');
    const tableBody = document.getElementById('topClassesTable');

    // Show the table when updating
    wrapper.style.display = 'none';

    // Clear old rows
    tableBody.innerHTML = '';
}

// Show modal helper
function showModal(title, message) {
    
    const modalElement = document.getElementById('noImageModal');
    const modalTitle = modalElement.querySelector('.modal-title');
    const modalBody = modalElement.querySelector('.modal-body');

    modalTitle.textContent = title;
    modalBody.textContent = message;

    const myModal = new bootstrap.Modal(modalElement);
    myModal.show();
}

function getStatusColor(confidencePercentage) {
    // Convert 0-100 percentage to 0.0-1.0 scale for threshold comparison
    const confidenceRatio = confidencePercentage / 100.0; 

    if (confidenceRatio >= CONFIDENCE_THRESHOLD) {
        return COLOR_MAP.CONFIRMED;
    } else if (confidenceRatio >= AMBIGUOUS_THRESHOLD) {
        return COLOR_MAP.WARNING;
    } else {
        return COLOR_MAP.UNRELIABLE;
    }
}

// Make Prediction
function predictImage() {
    // get the selected model id
    const modelDropdown = document.getElementById('model-select');
    const selectedModelId = modelDropdown ? modelDropdown.value : 'mobilenet_v2'; 

    if (!selectedImageName) {
        showModal('No Image Selected', 'Please upload an image before clicking Predict.');
        logMessage("No image selected!");
        return;
    }

    // Show loading
    document.getElementById('loading').style.display = 'block';

    // Clear previous results
    Plotly.purge('gaugeChart');
    Plotly.purge('graph1');
    Plotly.purge('graph2');

    logMessage(`Processing image with model: ${selectedModelId.toUpperCase()}...`);

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: selectedImageName, action: 'selected', model_id: selectedModelId })
    })
    .then(res => res.json())
    .then(data => { 
        if(data.status === 'success'){
            const topClass = data.top_prediction;
            const topConfidence = data.top_confidence;
            const top3Confidence = data.top_3_predictions;
            const allClassesResults = data.all_classes;
            const predictionStatus = data.prediction_status;

            const statusColor = getStatusColor(topConfidence);

            logMessage("Image processed successfully!");
            logMessage("Top Prediction & Confidence: " + topClass +" "+topConfidence+" %");

            
            // Match 7 classes with their respective colors
            const CLASS_COLOR_MAP = {
                'Auto Rickshaw': '#007bff', 
                'Bike': '#28a745', 
                'Car': '#ffc107', 
                'Motorcycle': '#17a2b8',
                'Plane': '#dc3545', 
                'Ship': '#6c757d', 
                'Train': '#343a40'
            };

            // Define a special color for the top prediction 
            const TOP_PREDICTION_COLOR = '#8a2be2'; 

            let classesForChart = [];      
            let confidencesForChart = []; 
            let colorsForBars = [];    // to hold the mapped colors

            // Looping all 7 classes
            allClassesResults.forEach((prediction) => {
                const className = prediction.class;

                classesForChart.push(className);
                confidencesForChart.push(prediction.confidence); 
                
                let barColor = CLASS_COLOR_MAP[className] || '#cccccc';
                
                if (className === data.top_prediction) {
                    barColor = TOP_PREDICTION_COLOR; 
                }

                colorsForBars.push(barColor);
            });

            // ---------- Gauge Chart ----------
            const traceGauge = {
                type: "indicator",
                mode: "gauge+number",
                value: topConfidence,
                title: { text: `<b>Top Prediction : ${topClass}</b><br><span style="color: ${statusColor}; font-size: 10px;"><b>${predictionStatus}</b></span><br><br>Confidence (%)`, 
                font: { size: 14 } },
                
                gauge: {
                    axis: { range: [0, 100], tickwidth: 1, tickcolor: "darkblue" },
                    bar: { color: "green" },
                    bgcolor: "white",
                    borderwidth: 2,
                    bordercolor: "gray",
                    steps: [
                        { range: [0, 50], color: "#ff6666" },
                        { range: [50, 80], color: "#ffcc66" },
                        { range: [80, 100], color: "#66ff66" }
                    ],
                }
            };
            const layoutGauge = { width: 360, height: 350, margin: { t: 0, b: 0 } };

            // ---------- Bar Chart ----------
            const traceBar = {
                x: classesForChart,
                y: confidencesForChart,
                type: 'bar',
                marker: {
                    color: colorsForBars
                }
            };
            const layoutBar = {
                title: 'Prediction Probabilities',
                yaxis: { title: 'Confidence' },
                xaxis: { title: 'Vehicle Class' },
                margin: { t: 50, b: 50 }
            };

            // ---------- Pie Chart ----------
            const tracePie = {
                values: confidencesForChart,
                labels: classesForChart,
                type: 'pie',
                marker: {
                    colors: colorsForBars
                },
                textinfo: "label+percent",
                textposition: "inside",
            };
            const layoutPie = { title: 'Prediction Distribution', margin: { t: 50, b: 50 } };

            // Plot all charts and hide loading after done
            Promise.all([
                Plotly.newPlot('gaugeChart', [traceGauge], layoutGauge),
                Plotly.newPlot('graph1', [traceBar], layoutBar),
                Plotly.newPlot('graph2', [tracePie], layoutPie)
            ]).then(() => {
                document.getElementById('loading').style.display = 'none';
                
            });

            updateTopClasses(top3Confidence);
            logMessage("Top 3 classes updated in table.");

        }else{
            logMessage(`Error: ${data.error || 'Unknown error occurred.'}`);
        }
        
    })
    .catch(err => {
        logMessage("Error: " + err.message);
        console.error(err);
        document.getElementById('loading').style.display = 'none';
        alert("Prediction failed. Please try again!");
    });
}

// log for Console
function logMessage(msg) {
    const consoleArea = document.getElementById('consoleArea');
    const time = new Date().toLocaleTimeString();
    consoleArea.innerHTML += `[${time}] ${msg}<br>`;
    consoleArea.scrollTop = consoleArea.scrollHeight; // auto-scroll to bottom
}

function updateTopClasses(predictions) {
    const wrapper = document.getElementById('topClassesWrapper');
    const tableBody = document.getElementById('topClassesTable');
    
     if(!wrapper || !tableBody) return;

    // Show the top 3 table
    wrapper.style.display = 'block';

    // Clear old rows
    tableBody.innerHTML = '';

    predictions.slice(0, 3).forEach((item, index) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${index + 1}</td>
            <td>${item.class}</td>
            <td>${item.confidence.toFixed(1)}</td>
        `;
        tableBody.appendChild(row);
    });
}

function selectUploadedImage(imgName) {
    selectedImageName = imgName;

    document.getElementById('selectedImage').src = '/static/images/' + imgName;
    
    clearCharts();
    clearTop3();
    
    document.querySelectorAll('#imageList .img-thumb').forEach(img => {
        img.classList.remove('selected');
    });

    document.getElementById('placeholderText').style.display = 'none';
    document.getElementById('selectedImage').style.display = 'block'; 
}

function uploadImage(event) {
    const allowedExtensions = ['jpg', 'jpeg', 'png', 'webp'];
    const maxSizeMB = 2;
    const files = event.target.files;

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const fileName = file.name;
        const fileSizeMB = file.size / (1024 * 1024);
        const extension = fileName.split('.').pop().toLowerCase();

        if (!allowedExtensions.includes(extension)) {
            showModal('Invalid File Type', `File "${fileName}" is not allowed. Only JPG, PNG, or WebP images are accepted.`);
            event.target.value = '';
            return false;
        }

        if (fileSizeMB > maxSizeMB) {
            showModal('File Too Large', `File "${fileName}" is too large (${fileSizeMB.toFixed(2)} MB). Maximum allowed size is ${maxSizeMB} MB.`);
            event.target.value = '';
            return false;
        }
    }

    const file = event.target.files[0];
    if (!file) return; 

    const selectedImage = document.getElementById('selectedImage');
    const placeholderText = document.getElementById('placeholderText');
    const uploadInput = document.getElementById('uploadInput');

    // Display Image Preview 
    const reader = new FileReader();

    reader.onload = function(e) {
        selectedImage.src = e.target.result;
        placeholderText.style.display = 'none';
        selectedImage.style.display = 'block'; 
        uploadInput.value = ''; 
    };

    reader.readAsDataURL(file);

    logMessage(`[Upload] Preparing to send file: ${file.name}`);

    const formData = new FormData();
    formData.append('file', file); 

    fetch('/upload', {
        method: 'POST',
        //  'Content-Type: multipart/form-data' 
        body: formData,
    })
    .then(res => {
        if (!res.ok) {
            throw new Error(`Upload failed with status: ${res.status}`);
        }
        return res.json();
    })
    .then(data => {
        if (data.status === 'success') {
            logMessage(`[Upload] File saved as: ${data.filename}`);

            // 1. Update UI and the uploaded filename
            selectUploadedImage(data.filename);

        } else {
            logMessage(`[Upload] Error saving file: ${data.error}`);
        }
    })
    .catch(error => {
        logMessage(`[Upload] Network Error: ${error.message}`);
        console.error('Fetch Error:', error);
    });
}

// Batch report Export
function generateBatchReport() {
    const statusDiv = document.getElementById('statusMessage');
    const modelDropdown = document.getElementById('model-select');
    const selectedModel = modelDropdown ? modelDropdown.value : 'mobilenet_v2';
    const btn = document.getElementById('generateReportBtn');

    const uploadBtn = document.querySelector('button[onclick="document.getElementById(\'uploadInput\').click()"]');
    const predictBtn = document.getElementById('predictBtn');
    
    btn.disabled = true;
    if (uploadBtn) uploadBtn.disabled = true;
    if (predictBtn) predictBtn.disabled = true;

    statusDiv.innerHTML = `<span class="text-primary d-flex align-items-center">
                                <div class="spinner-border spinner-border-sm me-2" role="status">
                                    <span class="visually-hidden">Loading...</span>
                                </div>
                            </span>`;

    const reportUrl = `/batch_predict_and_report?model=${selectedModel}`;

    fetch(reportUrl)
        .then(response => {
            if (!response.ok) {
                return response.json().then(err => { 
                    throw new Error(err.error || 'Unknown server error'); 
                });
            }

            const contentDisposition = response.headers.get('Content-Disposition');
            let timestamp = new Date().toISOString().slice(0, 19).replace(/[:]/g, '-');
            let filename = `batch_report_${selectedModel}_${timestamp}.pdf`;
            
            if (contentDisposition) {
                const match = contentDisposition.match(/filename="(.+)"/);
                if (match) {
                    filename = match[1];
                }
            }
            
            return response.blob().then(blob => {
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                a.remove();
                window.URL.revokeObjectURL(url);

                statusDiv.innerHTML = `<span class="text-success d-flex align-items-center">
                                        <i class="bi bi-check-circle-fill me-1"></i>
                                        </span>`;
            });
        })
        .catch(error => {
            console.error("Report Generation Error:", error);
            // --- UPDATED: Use Bootstrap error icon ---
            statusDiv.innerHTML = `<span class="text-danger d-flex align-items-center">
                                    <i class="bi bi-x-octagon-fill me-1"></i> 
                                    Error! ${error.message}
                                    </span>`;
        })
        .finally(() => {
            btn.disabled = false;
            if (uploadBtn) uploadBtn.disabled = false;
            if (predictBtn) predictBtn.disabled = false;
        });
}




