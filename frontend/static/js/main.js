document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("prediction-form");
    const btnPredict = document.getElementById("btn-predict");
    const btnFillDummy = document.getElementById("btn-fill-dummy");
    
    const resultPlaceholder = document.getElementById("result-placeholder");
    const resultContent = document.getElementById("result-content");
    const predictLoader = document.getElementById("predict-loader");
    
    const riskCircle = document.getElementById("risk-circle");
    const riskPercentage = document.getElementById("risk-percentage");
    const predictionLabel = document.getElementById("prediction-label");
    const predictionDesc = document.getElementById("prediction-desc");

    const explanationPlaceholder = document.getElementById("explanation-placeholder");
    const explanationContent = document.getElementById("explanation-content");
    const shapImage = document.getElementById("shap-image");

    const uploadForm = document.getElementById("upload-form-ingest");
    const csvFileInput = document.getElementById("csv-file-ingest");
    const uploadLabelSpan = document.querySelector("#upload-label-ingest span");
    const uploadStatus = document.getElementById("upload-status-ingest");

    // New Ingestion View Elements
    const viewPrediction = document.getElementById("view-prediction");
    const viewIngestion = document.getElementById("view-ingestion");
    const navItems = document.querySelectorAll(".nav-item");
    const ingestForm = document.getElementById("ingest-form");
    const btnFillDummyIngest = document.getElementById("btn-fill-dummy-ingest");

    // View Switching Logic
    navItems.forEach(item => {
        item.addEventListener("click", () => {
            const view = item.getAttribute("data-view");
            
            // Update Active Nav
            navItems.forEach(i => i.classList.remove("active"));
            item.classList.add("active");

            // Toggle Views
            if (view === "prediction") {
                viewPrediction.classList.remove("hidden");
                viewIngestion.classList.add("hidden");
            } else {
                viewPrediction.classList.add("hidden");
                viewIngestion.classList.remove("hidden");
            }
        });
    });

    const dummyData = {
        age: 62, gender: 1, education_years: 16, income_level: 3,
        smoker: 1, smoking_years: 25.0, cigarettes_per_day: 20.0,
        pack_years: 25.0, passive_smoking: 0, air_pollution_index: 45.0,
        occupational_exposure: 1, radon_exposure: 0, family_history_cancer: 1,
        copd: 0, asthma: 0, previous_tb: 0, chronic_cough: 1,
        chest_pain: 1, shortness_of_breath: 0, fatigue: 1, bmi: 26.5,
        oxygen_saturation: 98.0, fev1_x10: 38.0, crp_level: 1.5,
        xray_abnormal: 0, exercise_hours_per_week: 3.0, diet_quality: 4,
        alcohol_units_per_week: 5.0, healthcare_access: 4,
        lung_cancer_risk: 1
    };

    // Pre-fill dummy data for quick testing
    btnFillDummy.addEventListener("click", () => {
        for (const [key, value] of Object.entries(dummyData)) {
            const el = document.getElementById(key);
            if (el) el.value = value;
        }
    });

    btnFillDummyIngest.addEventListener("click", () => {
        for (const [key, value] of Object.entries(dummyData)) {
            const el = document.getElementById(`ingest-${key}`);
            if (el) el.value = value;
        }
    });

    // Handle Prediction Form Submission
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        // Show loader
        predictLoader.classList.remove("hidden");
        resultPlaceholder.classList.add("hidden");
        resultContent.classList.add("hidden");
        explanationPlaceholder.classList.remove("hidden");
        explanationContent.classList.add("hidden");

        // Collect form data
        const formData = new FormData(form);
        const dataObj = {};
        for (const [key, value] of formData.entries()) {
            dataObj[key] = isNaN(value) ? value : Number(value);
        }

        try {
            // First, run prediction for quick result
            const predictRes = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(dataObj)
            });
            
            const predictData = await predictRes.json();
            
            if (!predictRes.ok) throw new Error(predictData.detail || "Prediction failed");

            // Update UI with Prediction
            updatePredictionUI(predictData.prediction, predictData.risk_probability);

            // Then, fetch explanation in background
            fetchExplanation(dataObj);

        } catch (error) {
            console.error(error);
            alert("Error: " + error.message);
            predictLoader.classList.add("hidden");
            resultPlaceholder.classList.remove("hidden");
        }
    });

    function updatePredictionUI(prediction, riskProb) {
        predictLoader.classList.add("hidden");
        resultContent.classList.remove("hidden");

        const percentage = Math.round(riskProb * 100);
        riskPercentage.textContent = `${percentage}%`;
        
        // Update SVG circle dasharray (circumference is 100 in our viewBox 36x36)
        setTimeout(() => {
            riskCircle.style.strokeDasharray = `${percentage}, 100`;
        }, 100);

        // Update colors and text based on risk
        riskCircle.style.stroke = "var(--primary)";
        predictionLabel.className = "";
        
        if (percentage >= 70 || prediction === "YES") {
            riskCircle.style.stroke = "var(--danger)";
            predictionLabel.textContent = "High Risk";
            predictionLabel.classList.add("text-danger");
            predictionDesc.textContent = "The model indicates a HIGH probability of lung cancer. Immediate medical consultation is advised.";
        } else if (percentage >= 40) {
            riskCircle.style.stroke = "var(--warning)";
            predictionLabel.textContent = "Moderate Risk";
            predictionLabel.classList.add("text-warning");
            predictionDesc.textContent = "The model indicates moderate risk factors. Recommend routine screening and monitoring.";
        } else {
            riskCircle.style.stroke = "var(--success)";
            predictionLabel.textContent = "Low Risk";
            predictionLabel.classList.add("text-success");
            predictionDesc.textContent = "The model indicates a low probability of lung cancer based on provided metrics.";
        }
    }

    async function fetchExplanation(dataObj) {
        try {
            const expRes = await fetch("/explain", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(dataObj)
            });
            
            const expData = await expRes.json();
            if (expRes.ok && expData.explanation && expData.explanation.image_base64) {
                shapImage.src = `data:image/png;base64,${expData.explanation.image_base64}`;
                explanationPlaceholder.classList.add("hidden");
                explanationContent.classList.remove("hidden");
            }
        } catch (error) {
            console.error("Explanation error:", error);
            explanationPlaceholder.textContent = "Failed to load explanation.";
        }
    }

    // Handle CSV Upload UI
    csvFileInput.addEventListener("change", () => {
        if (csvFileInput.files.length > 0) {
            uploadLabelSpan.textContent = csvFileInput.files[0].name;
        } else {
            uploadLabelSpan.textContent = "Click or drag CSV here";
        }
    });

    // Handle Manual Ingestion Form Submission
    ingestForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        const btn = document.getElementById("btn-ingest-manual");
        const originalText = btn.innerHTML;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Adding...';
        btn.disabled = true;

        const formData = new FormData(ingestForm);
        const record = {};
        for (const [key, value] of formData.entries()) {
            record[key] = isNaN(value) ? value : Number(value);
        }

        try {
            const res = await fetch("/ingest-data", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ records: [record] })
            });
            const data = await res.json();

            if (res.ok) {
                alert("Success: " + data.message);
                ingestForm.reset();
            } else {
                throw new Error(data.detail || "Ingestion failed");
            }
        } catch (error) {
            console.error(error);
            alert("Error: " + error.message);
        } finally {
            btn.innerHTML = originalText;
            btn.disabled = false;
        }
    });

    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        
        if (csvFileInput.files.length === 0) return;

        const formData = new FormData();
        formData.append("file", csvFileInput.files[0]);

        uploadStatus.className = "status-msg text-warning";
        uploadStatus.textContent = "Uploading and validating...";

        try {
            const res = await fetch("/upload-csv", {
                method: "POST",
                body: formData
            });
            const data = await res.json();

            if (res.ok) {
                uploadStatus.className = "status-msg text-success";
                uploadStatus.textContent = data.message;
                csvFileInput.value = "";
                uploadLabelSpan.textContent = "Click or drag CSV here";
            } else {
                uploadStatus.className = "status-msg text-danger";
                uploadStatus.textContent = data.detail || "Upload failed";
            }
        } catch (error) {
            uploadStatus.className = "status-msg text-danger";
            uploadStatus.textContent = "Error: " + error.message;
        }
    });
});
