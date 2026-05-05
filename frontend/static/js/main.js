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

    const testScenarios = {
        low: {
            age: 25, gender: 0, education_years: 16, income_level: 4,
            smoker: 0, smoking_years: 0, cigarettes_per_day: 0,
            pack_years: 0, passive_smoking: 0, air_pollution_index: 20.0,
            occupational_exposure: 0, radon_exposure: 0, family_history_cancer: 0,
            copd: 0, asthma: 0, previous_tb: 0, chronic_cough: 0,
            chest_pain: 0, shortness_of_breath: 0, fatigue: 0, bmi: 22.5,
            oxygen_saturation: 98.0, fev1_x10: 40.0, crp_level: 1.2,
            xray_abnormal: 0, exercise_hours_per_week: 5.0, diet_quality: 5,
            alcohol_units_per_week: 0.0, healthcare_access: 5,
            lung_cancer_risk: 0
        },
        medium: {
            age: 55, gender: 1, education_years: 12, income_level: 3,
            smoker: 0, smoking_years: 0, cigarettes_per_day: 0,
            pack_years: 0, passive_smoking: 1, air_pollution_index: 68.0,
            occupational_exposure: 1, radon_exposure: 0, family_history_cancer: 1,
            copd: 0, asthma: 1, previous_tb: 0, chronic_cough: 1,
            chest_pain: 0, shortness_of_breath: 1, fatigue: 1, bmi: 25.1,
            oxygen_saturation: 93.5, fev1_x10: 32.0, crp_level: 7.5,
            xray_abnormal: 0, exercise_hours_per_week: 1.5, diet_quality: 3,
            alcohol_units_per_week: 5.0, healthcare_access: 4,
            lung_cancer_risk: 0
        },
        high: {
            age: 68, gender: 1, education_years: 8, income_level: 1,
            smoker: 1, smoking_years: 45.0, cigarettes_per_day: 20.0,
            pack_years: 45.0, passive_smoking: 1, air_pollution_index: 85.0,
            occupational_exposure: 1, radon_exposure: 1, family_history_cancer: 1,
            copd: 1, asthma: 0, previous_tb: 1, chronic_cough: 1,
            chest_pain: 1, shortness_of_breath: 1, fatigue: 1, bmi: 29.2,
            oxygen_saturation: 91.0, fev1_x10: 25.0, crp_level: 15.4,
            xray_abnormal: 1, exercise_hours_per_week: 0.5, diet_quality: 1,
            alcohol_units_per_week: 15.0, healthcare_access: 2,
            lung_cancer_risk: 1
        },
        special: {
            age: 30, gender: 1, education_years: 16, income_level: 4,
            smoker: 0, smoking_years: 0, cigarettes_per_day: 0,
            pack_years: 0, passive_smoking: 0, air_pollution_index: 35.0,
            occupational_exposure: 1, radon_exposure: 1, family_history_cancer: 1,
            copd: 0, asthma: 0, previous_tb: 0, chronic_cough: 1,
            chest_pain: 1, shortness_of_breath: 0, fatigue: 1, bmi: 22.0,
            oxygen_saturation: 92.0, fev1_x10: 28.0, crp_level: 12.0,
            xray_abnormal: 1, exercise_hours_per_week: 4.0, diet_quality: 4,
            alcohol_units_per_week: 1.0, healthcare_access: 5,
            lung_cancer_risk: 1
        }
    };

    const scenarioDescriptions = {
        low: "<strong>🟢 Nguy cơ Thấp:</strong> Bệnh nhân trẻ (25 tuổi), lối sống lành mạnh.<br><strong>Thuộc tính then chốt:</strong> Không hút thuốc (Smoker: 0), Môi trường sạch (Pollution: 12.0), Oxy cao (98.5%) và không có triệu chứng.<br><strong>Mục tiêu:</strong> Kiểm tra độ đặc hiệu (Specificity), đảm bảo AI không báo động giả với người khỏe mạnh.",
        medium: "<strong>🟡 Nguy cơ Trung bình:</strong> Người trung niên (55 tuổi), bắt đầu có triệu chứng.<br><strong>Thuộc tính then chốt:</strong> Ô nhiễm cao (68.0), <strong>Tiếp xúc nghề nghiệp (Occupational: 1)</strong>, có <strong>Ho mãn tính</strong> và <strong>Oxy bắt đầu giảm (93.5%)</strong>.<br><strong>Mục tiêu:</strong> Kiểm tra khả năng nhận diện các ca đang chuyển biến từ trung bình sang cao.",
        high: "<strong>🔴 Nguy cơ Cao:</strong> Bệnh nhân lớn tuổi (68 tuổi), bệnh lý rõ rệt.<br><strong>Thuộc tính then chốt:</strong> Tiền sử hút thuốc cực nặng (<strong>45 Pack-Years</strong>), <strong>Oxy thấp (91%)</strong>, <strong>Chỉ số viêm CRP cao (15.4)</strong> và <strong>X-quang bất thường</strong>.<br><strong>Mục tiêu:</strong> Đảm bảo độ nhạy (Recall) tối đa, không được bỏ sót các ca bệnh điển hình.",
        special: "<strong>🟣 Trường hợp Đặc biệt:</strong> Thanh niên (30 tuổi), sống lành mạnh nhưng rủi ro 'ngầm'.<br><strong>Thuộc tính then chốt:</strong> <strong>Phơi nhiễm Radon & Độc hại nghề nghiệp</strong>, có <strong>Di truyền (Family History: 1)</strong>, xuất hiện <strong>Đau ngực</strong> và <strong>X-quang bất thường</strong> dù không hút thuốc.<br><strong>Mục tiêu:</strong> Kiểm tra độ nhạy của AI với các tác nhân gây ung thư không liên quan đến thuốc lá ở người trẻ tuổi."
    };

    // Handle Scenario Button Clicks
    document.querySelectorAll(".btn-scenario").forEach(btn => {
        btn.addEventListener("click", () => {
            const scenarioKey = btn.getAttribute("data-scenario");
            const data = testScenarios[scenarioKey];
            console.log(`Loading scenario: ${scenarioKey}`);

            if (data) {
                // Fill Prediction Form
                let filledCount = 0;
                for (const [key, value] of Object.entries(data)) {
                    const el = document.getElementById(key);
                    if (el) {
                        el.value = value;
                        filledCount++;
                    }
                }

                // Also fill Ingest Form if elements exist
                for (const [key, value] of Object.entries(data)) {
                    const el = document.getElementById(`ingest-${key}`);
                    if (el) el.value = value;
                }

                console.log(`Filled ${filledCount} fields for prediction.`);

                // Show description on UI
                const descBox = document.getElementById("scenario-description");
                if (descBox) {
                    descBox.innerHTML = scenarioDescriptions[scenarioKey] || "";
                    descBox.classList.remove("hidden");
                }

                // Visual feedback
                btn.style.background = "var(--success)";

                btn.style.color = "white";
                setTimeout(() => {
                    btn.style.background = "";
                    btn.style.color = "";
                }, 500);
            }
        });
    });

    // Keeping the original Fill Dummy Data button listener for Ingestion view compatibility
    if (btnFillDummyIngest) {
        btnFillDummyIngest.addEventListener("click", () => {
            console.log("Filling dummy data for ingestion...");
            const data = testScenarios.high;
            for (const [key, value] of Object.entries(data)) {
                const el = document.getElementById(`ingest-${key}`);
                if (el) el.value = value;
            }
        });
    }

    // Handle Prediction Form Submission
    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        console.log("Form submitted, collecting data...");

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
            predictionLabel.textContent = "Nguy cơ Cao";
            predictionLabel.classList.add("text-danger");
            predictionDesc.textContent = "Mô hình chỉ ra xác suất ung thư phổi CAO. Khuyến cáo tham vấn y tế ngay lập tức.";
        } else if (percentage >= 40) {
            riskCircle.style.stroke = "var(--warning)";
            predictionLabel.textContent = "Nguy cơ Trung bình";
            predictionLabel.classList.add("text-warning");
            predictionDesc.textContent = "Mô hình chỉ ra các yếu tố nguy cơ trung bình. Đề xuất tầm soát và theo dõi định kỳ.";
        } else {
            riskCircle.style.stroke = "var(--success)";
            predictionLabel.textContent = "Nguy cơ Thấp";
            predictionLabel.classList.add("text-success");
            predictionDesc.textContent = "Mô hình chỉ ra xác suất ung thư phổi thấp dựa trên các chỉ số được cung cấp.";
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
            explanationPlaceholder.textContent = "Không thể tải phần giải thích (SHAP).";
        }
    }

    // Handle CSV Upload UI
    csvFileInput.addEventListener("change", () => {
        if (csvFileInput.files.length > 0) {
            uploadLabelSpan.textContent = csvFileInput.files[0].name;
        } else {
            uploadLabelSpan.textContent = "Nhấp hoặc kéo thả CSV vào đây";
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
        uploadStatus.textContent = "Đang tải lên và xác thực...";

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
                uploadLabelSpan.textContent = "Nhấp hoặc kéo thả CSV vào đây";
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
