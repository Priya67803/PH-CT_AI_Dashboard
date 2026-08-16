document.addEventListener('DOMContentLoaded', () => {

    const runBtn = document.getElementById('runPipelineBtn');
    const resultCard = document.getElementById('resultCard');
    const confBar = document.getElementById('confBar');
    const confText = document.getElementById('confText');
    const predText = document.getElementById('predText');

    // Viewers & Dropzone
    const dropZone = document.getElementById('dropZone');
    const imageUpload = document.getElementById('imageUpload');
    const originalViewer = document.getElementById('originalViewer');
    const gradcamViewer = document.getElementById('gradcamViewer');

    // Chart.js instances
    let lossChart = null;
    let rocChart = null;

    Chart.defaults.color = "rgba(255, 255, 255, 0.7)";
    Chart.defaults.font.family = "'Inter', sans-serif";

    function resetDashboard() {
        resultCard.style.display = 'none';

        gradcamViewer.style.backgroundImage = 'none';
        const gChildren = gradcamViewer.children;
        for (let i = 0; i < gChildren.length; i++) {
            gChildren[i].style.display = 'block';
        }

        // Reset metrics
        document.getElementById('statAcc').textContent = '--';
        document.getElementById('statF1').textContent = '--';
        document.getElementById('statRoc').textContent = '--';
        document.getElementById('statTime').textContent = '--';

        confBar.style.width = '0%';
        predText.className = "prediction-text";

        // Reset Charts UI
        document.getElementById('chart-loss').style.display = 'none';
        document.getElementById('chart-roc').style.display = 'none';
        document.getElementById('chart-cm').style.display = 'none';

        document.getElementById('placeholder-loss').style.display = 'flex';
        document.getElementById('placeholder-roc').style.display = 'flex';
        document.getElementById('placeholder-cm').style.display = 'flex';

        // Destroy existing Chart.js objects to prevent hover artifacts
        if (lossChart) { lossChart.destroy(); lossChart = null; }
        if (rocChart) { rocChart.destroy(); rocChart = null; }
        document.getElementById('chart-cm').innerHTML = '';
    }

    // --- DRAG AND DROP LOGIC ---
    if (dropZone && imageUpload) {
        dropZone.addEventListener('click', () => imageUpload.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) {
                imageUpload.files = e.dataTransfer.files;
                handleFileUpload(imageUpload.files[0]);
            }
        });

        imageUpload.addEventListener('change', function (event) {
            if (event.target.files.length) {
                handleFileUpload(event.target.files[0]);
            }
        });
    }

    function handleFileUpload(file) {
        resetDashboard();
        if (file) {
            const reader = new FileReader();
            reader.onload = function (e) {
                originalViewer.style.backgroundImage = `url(${e.target.result})`;

                const children = originalViewer.children;
                for (let i = 0; i < children.length; i++) {
                    children[i].style.display = 'none';
                }
            }
            reader.readAsDataURL(file);
        }
    }

    // Reset dashboard immediately when they explicitly toggle dropdowns
    const modelSelect = document.getElementById('modelSelect');
    if (modelSelect) {
        modelSelect.addEventListener('change', function () {
            if (imageUpload.files.length > 0) {
                resetDashboard();
            }
        });
    }

    // Interactive Render logic
    function renderCharts(chartData) {
        document.getElementById('placeholder-loss').style.display = 'none';
        document.getElementById('placeholder-roc').style.display = 'none';
        document.getElementById('placeholder-cm').style.display = 'none';

        document.getElementById('chart-loss').style.display = 'block';
        document.getElementById('chart-roc').style.display = 'block';

        // 1. Render Loss Curve
        const ctxLoss = document.getElementById('chart-loss').getContext('2d');
        lossChart = new Chart(ctxLoss, {
            type: 'line',
            data: {
                labels: chartData.loss.labels,
                datasets: [
                    { label: 'Train Loss', data: chartData.loss.train, borderColor: '#00f2fe', backgroundColor: 'rgba(0, 242, 254, 0.2)', fill: true, tension: 0.4, borderWidth: 2, pointRadius: 4, pointHoverRadius: 8, pointHoverBackgroundColor: '#fff' },
                    { label: 'Val Loss', data: chartData.loss.val, borderColor: '#ff4757', backgroundColor: 'rgba(255, 71, 87, 0.2)', fill: true, tension: 0.4, borderWidth: 2, pointRadius: 4, pointHoverRadius: 8, pointHoverBackgroundColor: '#fff' }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                plugins: { legend: { position: 'top', labels: { boxWidth: 10 } } },
                scales: { x: { grid: { color: 'rgba(255,255,255,0.05)' } }, y: { grid: { color: 'rgba(255,255,255,0.05)' } } }
            }
        });

        // 2. Render ROC
        const ctxRoc = document.getElementById('chart-roc').getContext('2d');
        rocChart = new Chart(ctxRoc, {
            type: 'line',
            data: {
                labels: chartData.roc.fpr,
                datasets: [
                    { label: 'Class ROC', data: chartData.roc.tpr, borderColor: '#facc15', fill: true, backgroundColor: 'rgba(250,204,21,0.25)', tension: 0.3, borderWidth: 3, pointRadius: 2, pointHoverRadius: 8, pointHoverBackgroundColor: '#fff' },
                    { label: 'Baseline', data: chartData.roc.fpr, borderColor: 'rgba(255,255,255,0.3)', borderDash: [5, 5], pointRadius: 0 }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { x: { grid: { color: 'rgba(255,255,255,0.05)' } }, y: { grid: { color: 'rgba(255,255,255,0.05)' } } }
            }
        });

        // 3. Render Plotly Heatmap Confusion Matrix
        const cmBox = document.getElementById('chart-cm');
        const matrix = chartData.cm;
        const x_classes = ["Normal", "Mild", "Severe", "Critical"];
        const y_classes = [...x_classes].reverse();
        let maxVal = 0;
        matrix.forEach(row => row.forEach(v => { if (v > maxVal) maxVal = v; }));

        const plotData = [{
            z: matrix.slice().reverse(),
            x: x_classes,
            y: y_classes,
            type: 'heatmap',
            colorscale: [
                [0, 'rgba(40, 40, 60, 0.4)'],
                [1, 'rgba(74, 222, 128, 0.9)']
            ],
            hoverongaps: false,
            showscale: false,
            hovertemplate: '<b>Actual:</b> %{y}<br><b>Predicted:</b> %{x}<br><b>Scans:</b> %{z}<extra></extra>'
        }];

        const annotations = [];
        for (let i = 0; i < matrix.length; i++) {
            for (let j = 0; j < matrix[i].length; j++) {
                let cellVal = matrix[matrix.length - 1 - i][j];
                annotations.push({
                    x: x_classes[j],
                    y: y_classes[i],
                    text: cellVal.toString(),
                    font: { family: 'Inter', size: 14, color: cellVal > (maxVal / 2) ? '#000' : 'rgba(255,255,255,0.9)', weight: 600 },
                    showarrow: false
                });
            }
        }

        const layout = {
            annotations: annotations,
            xaxis: { title: { text: 'Predicted Class', standoff: 10 }, side: 'bottom', color: 'rgba(255,255,255,0.6)', showgrid: false, automargin: true },
            yaxis: { title: { text: 'True Class', standoff: 10 }, color: 'rgba(255,255,255,0.6)', showgrid: false, automargin: true },
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { t: 10, r: 10, b: 50, l: 65 },
            font: { family: "'Inter', sans-serif" }
        };

        cmBox.style.display = 'block';
        Plotly.newPlot('chart-cm', plotData, layout, { displayModeBar: false, responsive: true });
    }

    // Run deep learning analysis
    runBtn.addEventListener('click', async () => {
        if (!imageUpload.files || imageUpload.files.length === 0) {
            alert("Please upload a CT scan image first.");
            return;
        }

        // UI Loading State
        runBtn.innerHTML = '<i class="ri-loader-4-line ri-spin"></i> Processing Tensor Data...';
        runBtn.classList.remove('pulse-glow');
        runBtn.style.opacity = '0.8';
        runBtn.disabled = true;

        try {
            const formData = new FormData();
            formData.append('file', imageUpload.files[0]);

            // Capture the dropdown model selection
            const modelCore = document.getElementById('modelSelect').value;
            formData.append('model', modelCore);

            // Call Flask API Endpoint
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();

            if (data.status === 'success') {
                // Unveil Result Card
                resultCard.style.display = 'block';
                resultCard.style.animation = 'slideDown 0.5s ease backwards';

                // Populate Data dynamically colored
                predText.textContent = data.prediction;
                predText.className = "prediction-text " + data.style_class;
                confText.textContent = data.confidence + '%';

                // Animate progress bar (inherit class color)
                let barColor = "#ff4757";
                if (data.style_class === "pred-normal") barColor = "#4ade80";
                if (data.style_class === "pred-mild") barColor = "#facc15";
                if (data.style_class === "pred-severe") barColor = "#fb923c";
                confBar.style.background = barColor;

                setTimeout(() => {
                    confBar.style.width = data.confidence + '%';
                }, 100);

                // Set GradCAM Image on right viewer
                if (data.gradcam_image) {
                    gradcamViewer.style.backgroundImage = 'url(data:image/png;base64,' + data.gradcam_image + ')';

                    const gChildren = gradcamViewer.children;
                    for (let i = 0; i < gChildren.length; i++) {
                        gChildren[i].style.display = 'none';
                    }
                }

                // Populate Metrics
                document.getElementById('statAcc').textContent = data.metrics.val_acc + '%';
                document.getElementById('statF1').textContent = data.metrics.f1_score;
                document.getElementById('statRoc').textContent = data.metrics.roc_auc;
                document.getElementById('statTime').textContent = data.metrics.inference_time;

                // Render Interactive Charts
                if (data.charts) {
                    renderCharts(data.charts);
                }
            }

        } catch (err) {
            console.error("API Error: ", err);
            alert("Connection to AI Engine Failed.");
            resetDashboard();
        } finally {
            // Restore button
            runBtn.innerHTML = '<i class="ri-check-double-line"></i> ANALYSIS COMPLETE';
            runBtn.style.background = 'linear-gradient(135deg, #4ade80, #22c55e)';
            setTimeout(() => {
                runBtn.innerHTML = '<i class="ri-flashlight-fill"></i> INITIALIZE PIPELINE';
                runBtn.style.background = '';
                runBtn.disabled = false;
            }, 3000);
        }

    });
});
