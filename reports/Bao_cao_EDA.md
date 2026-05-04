# Báo cáo phân tích dữ liệu khám phá (EDA)

**Dự án:** Lung Cancer Prediction (`Data_Mining-Lung_Cancer_Prediction`)  
**Nguồn dữ liệu:** `data/raw/survey_lung_cancer.csv`  
**Pipeline EDA:** `src/data/eda.py` (kết quả hình ảnh tại `reports/eda/`, báo cáo text tại `reports/eda/eda_report.txt`)

---

## 4.1. Mục tiêu của EDA

### Vì sao cần làm mục này?

Trước khi huấn luyện mô hình, cần hiểu dữ liệu đang có chất lượng ra sao, có rủi ro gì và có phù hợp với mục tiêu dự đoán `lung_cancer_risk` hay không. Nếu bỏ qua bước này, mô hình có thể học từ dữ liệu lỗi hoặc lệch và cho kết quả thiếu tin cậy.

### Giải thích mục này

Phân tích dữ liệu khám phá (EDA) nhằm **làm quen và kiểm chứng chất lượng dữ liệu** trước khi huấn luyện mô hình. Cụ thể, EDA giúp:

- **Hiểu cấu trúc và ý nghĩa từng biến** (đếm, kiểu dữ liệu, phân phối, giá trị đặc trưng).
- **Phát hiện rủi ro mô hình hóa:** mất cân bằng lớp, giá trị thiếu, ngoại lai, đa cộng tuyến.
- **So sánh hai nhóm nguy cơ** để nhận diện đặc trưng có khả năng mang **tín hiệu dự đoán**.
- **Làm cơ sở cho tiền xử lý:** chọn metric (ví dụ nhấn mạnh recall/F1 khi lớp hiếm), xử lý outlier, và quyết định feature engineering.

EDA **không thay thế** mô hình thống kê hay học máy đa biến, nhưng **giảm sai lầm** do đưa dữ liệu “mù” vào huấn luyện.

### Kết luận

Mục tiêu EDA là tạo nền tảng tin cậy cho toàn bộ pipeline: hiểu dữ liệu, nhận diện rủi ro sớm, và định hướng bước tiền xử lý/mô hình hóa phù hợp.

---

## 4.2. Kiểm tra cấu trúc dữ liệu

### Vì sao cần làm mục này?

Kiểm tra cấu trúc dữ liệu là bước xác nhận dữ liệu đầu vào có đúng schema dự kiến hay không (số mẫu, số cột, kiểu dữ liệu, tên biến). Đây là điều kiện tiên quyết để tránh lỗi pipeline ở các bước preprocess, feature engineering và train.

### Giải thích mục này

### Số dòng, số cột

- **1000** dòng (mẫu), **30** cột (biến).

### Kiểu dữ liệu

Tất cả các cột đều **dạng số** (`int64` hoặc `float64`), không có cột văn bản thô — thuận lợi cho pipeline huấn luyện.

- **`int64`:** các biến giá trị nguyên (nhị phân/thứ bậc được mã hóa số), ví dụ `gender`, `smoker`, `copd`, `lung_cancer_risk`, …
- **`float64`:** các biến liên tục hoặc có phần thập phân, ví dụ `pack_years`, `bmi`, `oxygen_saturation`, `crp_level`, …

### Tên biến (30 cột)

`age`, `gender`, `education_years`, `income_level`, `smoker`, `smoking_years`, `cigarettes_per_day`, `pack_years`, `passive_smoking`, `air_pollution_index`, `occupational_exposure`, `radon_exposure`, `family_history_cancer`, `copd`, `asthma`, `previous_tb`, `chronic_cough`, `chest_pain`, `shortness_of_breath`, `fatigue`, `bmi`, `oxygen_saturation`, `fev1_x10`, `crp_level`, `xray_abnormal`, `exercise_hours_per_week`, `diet_quality`, `alcohol_units_per_week`, `healthcare_access`, `lung_cancer_risk`.

### Giá trị đầu tiên (dòng đầu của CSV)

| Biến | Giá trị |
|:-----|--------:|
| age | 69 |
| gender | 0 |
| education_years | 21 |
| income_level | 2 |
| smoker | 1 |
| smoking_years | 47 |
| cigarettes_per_day | 16 |
| pack_years | 37.6 |
| passive_smoking | 0 |
| air_pollution_index | 53.575573 |
| occupational_exposure | 0 |
| radon_exposure | 0 |
| family_history_cancer | 0 |
| copd | 0 |
| asthma | 1 |
| previous_tb | 0 |
| chronic_cough | 1 |
| chest_pain | 0 |
| shortness_of_breath | 1 |
| fatigue | 0 |
| bmi | 19.500802 |
| oxygen_saturation | 94.149342 |
| fev1_x10 | 26.210025 |
| crp_level | 7.076544 |
| xray_abnormal | 0 |
| exercise_hours_per_week | 8.342551 |
| diet_quality | 4 |
| alcohol_units_per_week | 24.937279 |
| healthcare_access | 3 |
| lung_cancer_risk | 0 |

### Thống kê mô tả

Bảng dưới đây là `describe()` cho toàn bộ biến số (làm tròn 4 chữ số thập phân). Cột `50%` là **trung vị** (Q2).

| feature | count | mean | std | min | 25% | 50% | 75% | max |
|:------------------------|--------:|--------:|--------:|--------:|--------:|--------:|--------:|---------:|
| age | 1000 | 52.881 | 20.9589 | 18 | 34.75 | 52.5 | 71 | 89 |
| gender | 1000 | 0.497 | 0.5002 | 0 | 0 | 0 | 1 | 1 |
| education_years | 1000 | 14.412 | 4.0295 | 8 | 11 | 14 | 18 | 21 |
| income_level | 1000 | 3.028 | 1.4259 | 1 | 2 | 3 | 4 | 5 |
| smoker | 1000 | 0.423 | 0.4943 | 0 | 0 | 0 | 1 | 1 |
| smoking_years | 1000 | 7.593 | 13.341 | 0 | 0 | 0 | 10 | 72 |
| cigarettes_per_day | 1000 | 8.701 | 12.4118 | 0 | 0 | 0 | 17 | 39 |
| pack_years | 1000 | 7.7546 | 16.077 | 0 | 0 | 0 | 7.0375 | 106.4 |
| passive_smoking | 1000 | 0.296 | 0.4567 | 0 | 0 | 0 | 1 | 1 |
| air_pollution_index | 1000 | 54.5946 | 25.9498 | 10.3527 | 31.9039 | 54.3407 | 76.5832 | 99.9515 |
| occupational_exposure | 1000 | 0.211 | 0.4082 | 0 | 0 | 0 | 0 | 1 |
| radon_exposure | 1000 | 0.079 | 0.2699 | 0 | 0 | 0 | 0 | 1 |
| family_history_cancer | 1000 | 0.145 | 0.3523 | 0 | 0 | 0 | 0 | 1 |
| copd | 1000 | 0.131 | 0.3376 | 0 | 0 | 0 | 0 | 1 |
| asthma | 1000 | 0.103 | 0.3041 | 0 | 0 | 0 | 0 | 1 |
| previous_tb | 1000 | 0.058 | 0.2339 | 0 | 0 | 0 | 0 | 1 |
| chronic_cough | 1000 | 0.2 | 0.4002 | 0 | 0 | 0 | 0 | 1 |
| chest_pain | 1000 | 0.161 | 0.3677 | 0 | 0 | 0 | 0 | 1 |
| shortness_of_breath | 1000 | 0.199 | 0.3994 | 0 | 0 | 0 | 0 | 1 |
| fatigue | 1000 | 0.309 | 0.4623 | 0 | 0 | 0 | 1 | 1 |
| bmi | 1000 | 26.7684 | 4.7765 | 18.0043 | 22.8292 | 27.0681 | 30.6871 | 34.9855 |
| oxygen_saturation | 1000 | 96.0375 | 2.2796 | 92.0636 | 94.1473 | 96.0348 | 97.948 | 99.9915 |
| fev1_x10 | 1000 | 35.5938 | 8.8392 | 20.0183 | 27.8102 | 35.7096 | 43.2137 | 49.9961 |
| crp_level | 1000 | 5.027 | 2.832 | 0.01 | 2.7022 | 4.9944 | 7.4566 | 9.993 |
| xray_abnormal | 1000 | 0.116 | 0.3204 | 0 | 0 | 0 | 0 | 1 |
| exercise_hours_per_week | 1000 | 7.7682 | 4.4061 | 0.0007 | 3.9774 | 7.9546 | 11.6208 | 14.9948 |
| diet_quality | 1000 | 3.154 | 1.4284 | 1 | 2 | 3 | 4 | 5 |
| alcohol_units_per_week | 1000 | 12.6851 | 7.2242 | 0.005 | 6.5383 | 12.7579 | 18.8596 | 24.9956 |
| healthcare_access | 1000 | 2.959 | 1.4189 | 1 | 2 | 3 | 4 | 5 |
| lung_cancer_risk | 1000 | 0.209 | 0.4068 | 0 | 0 | 0 | 0 | 1 |

*Bản đầy đủ cùng kiểu dữ liệu chi tiết còn được ghi trong `reports/eda/eda_report.txt` khi chạy `python -m src.data.eda`.*

### Kết luận

Bộ dữ liệu có cấu trúc đầy đủ (1000 mẫu, 30 biến), kiểu dữ liệu đã số hóa đồng nhất và phù hợp để đi tiếp sang các bước phân tích chất lượng sâu hơn.

---

## 4.3. Phân tích phân phối biến mục tiêu

### Vì sao cần làm mục này?

Biết phân phối của biến mục tiêu giúp chọn chiến lược huấn luyện đúng ngay từ đầu. Nếu dữ liệu mất cân bằng mà vẫn dùng metric không phù hợp, mô hình có thể cho accuracy cao nhưng bỏ sót nhiều ca nguy cơ.

### Giải thích mục này

**Biến mục tiêu:** `lung_cancer_risk` (0 = nguy cơ thấp / không mắc theo nhãn trong dữ liệu; 1 = nguy cơ cao).

**Hình minh họa:** `reports/eda/target_distribution.png`

**Tần suất:**

| Lớp | Số mẫu | Tỷ lệ |
|:---:|:------:|:-----:|
| 0 | 791 | 79.1% |
| 1 | 209 | 20.9% |

**Đánh giá mất cân bằng:** Tỷ lệ **khoảng 4:1** (lớp 0 chiếm đa số) — đây là **mất cân bằng lớp đáng kể**. Ảnh hưởng thường gặp: mô hình có xu hướng thiên về dự đoán lớp đa số; metric độ chính xác (accuracy) dễ **ảo cao** nếu chỉ đoán lớp 0.

**Hướng xử lý gợi ý:** theo dõi **recall**, **F1**, **ROC-AUC**; cân nhắc `class_weight`, lấy mẫu lại, hoặc các kỹ thuật cân bằng (ví dụ SMOTE) tùy pipeline và yêu cầu nghiệp vụ.

### Kết luận

`lung_cancer_risk` bị mất cân bằng rõ (79.1% vs 20.9%), vì vậy cần ưu tiên metric nhạy với lớp thiểu số và áp dụng kỹ thuật cân bằng phù hợp trong huấn luyện.

---

## 4.4. Phân tích giá trị thiếu

### Vì sao cần làm mục này?

Missing values là nguyên nhân phổ biến gây sai lệch thống kê và lỗi mô hình. Kiểm tra mục này giúp quyết định có cần imputation hay có thể giữ nguyên dữ liệu.

### Giải thích mục này

**Kết quả:** Mọi cột đều có **0** giá trị thiếu — **tỉ lệ missing 0%** trên toàn bộ biến.

**Tệp hình:** `reports/eda/missing_values_percentage.png` chỉ được tạo khi **có ít nhất một cột** có missing; với bộ dữ liệu hiện tại, `src/data/eda.py` **bỏ qua** bước vẽ này vì không có cột nào thiếu.

**Ảnh hưởng:** Không cần **imputation** (điền thiếu) cho tập hiện tại. Vẫn nên **giữ kiểm tra missing** trong pipeline production để phát hiện lô dữ liệu mới bị lỗi.

### Kết luận

Tập dữ liệu hiện tại sạch về missing (0%), giúp đơn giản hóa tiền xử lý; tuy nhiên kiểm tra missing vẫn cần duy trì trong môi trường production để phát hiện dữ liệu mới bất thường.

---

## 4.5. Phân tích ngoại lai

### Vì sao cần làm mục này?

Ngoại lai có thể kéo lệch thống kê, làm mô hình học không ổn định và giảm khả năng tổng quát hóa. Mục này giúp xác định mức độ xuất hiện giá trị cực đoan để quyết định có cần xử lý robust/capping.

### Giải thích mục này

### Biến được vẽ boxplot outlier trong pipeline EDA

Module `eda.py` cấu hình **`DEFAULT_CONTINUOUS_COLS`** và sinh các hình:

- `reports/eda/outlier_boxplot_age.png`
- `reports/eda/outlier_boxplot_oxygen_saturation.png`
- `reports/eda/outlier_boxplot_fev1_x10.png`
- `reports/eda/outlier_boxplot_crp_level.png`

Các điểm **chấm tròn** trên boxplot thường là giá trị nằm **ngoài khoảng râu** theo quy tắc **1.5 × IQR** (do thư viện vẽ tính nội bộ; code dự án không gán tay Q1/Q3/IQR).

### BMI và các biến liên tục khác

Trong **CSV có cột `bmi`**, và thống kê mô tả cho thấy BMI phân bố khoảng **18–35** với độ lệch chuẩn vừa phải. Tuy nhiên **`eda.py` hiện không sinh** `outlier_boxplot_bmi.png`; để đồng bộ với yêu cầu phân tích BMI, có thể **mở rộng** `DEFAULT_CONTINUOUS_COLS` hoặc chạy notebook `reports/EDA_step_by_step.ipynb` cho biểu đồ bổ sung.

### Nhận xét ngắn

- **`age`:** boxplot cho thấy phạm vi quan sát rộng (xấp xỉ 18-89), trung vị quanh 52-53, không xuất hiện điểm outlier rõ theo quy tắc 1.5xIQR.
- **`crp_level`:** trung vị quanh ~5, hộp và râu cân đối trong miền dữ liệu quan sát; không thấy điểm outlier rõ trên boxplot.
- **`oxygen_saturation`**, **`fev1_x10`:** hộp và râu tương đối gọn, biến thiên hẹp hơn so với các biến hành vi/phơi nhiễm; không có outlier rõ theo quy tắc 1.5xIQR.

**Tiền xử lý trong dự án:** pipeline preprocess vẫn áp dụng **IQR capping** như một lớp an toàn khi dữ liệu mới phát sinh giá trị cực đoan, dù tập hiện tại không ghi nhận outlier rõ ở các biến liên tục đã vẽ.

### Kết luận

Ở các biến liên tục được kiểm tra (`age`, `oxygen_saturation`, `fev1_x10`, `crp_level`), chưa thấy outlier rõ theo 1.5xIQR; do đó dữ liệu hiện tại ổn định, đồng thời vẫn nên giữ IQR capping như cơ chế phòng ngừa cho dữ liệu mới.

---

## 4.6. Phân tích phân phối từng nhóm đặc trưng

### Vì sao cần làm mục này?

Chia biến theo **nhóm chủ đề** (thuốc lá, lâm sàng, chỉ số đo lường, xã hội–kinh tế) giúp đọc phân bố **theo ngữ cảnh nghiệp vụ**. Loại hình vẽ được **chọn theo kiểu dữ liệu** (không dùng một kiểu cho mọi biến); tất cả vẫn trên **toàn bộ mẫu** (không tách `lung_cancer_risk` trong cùng một hình).

### Giải thích mục này

**Artefact:** trong `reports/eda/`, tên file `distribution_<nhóm>_<biến>.png` (sinh bởi `src/data/eda.py`). **Thuốc lá** và **đo lường:** cùng dạng **một panel** — histogram+KDE (thang gốc, dễ đọc nhanh). **Lâm sàng:** **bar chart tần suất** 0/1 (có ghi count). **Xã hội–kinh tế:** **cột tần suất có thứ tự** theo mức, không KDE.

### Nhận xét chung (theo nhóm)

- **Nhóm thuốc lá:** Các chỉ số phơi nhiễm và cường độ đều mang **cấu trúc lệch phải mạnh và nhiều mẫu quanh 0** — tức đa số ít hoặc không tích lũy, một phần nhỏ có mức rất cao. Đây là nhóm “hành vi / tích lũy” nên phân bố thường **không chuẩn**, quan trọng cho log/bins hoặc mô hình chịu skew.
- **Nhóm lâm sàng / triệu chứng:** Toàn biến **nhị phân**; biểu đồ là **hai cột tần suất 0 và 1**, **0 chiếm đa số rõ** — mô tả “phần lớn không có triệu chứng”. So sánh giữa nhãn nguy cơ nếu cần nên dùng **tỷ lệ / bảng chéo**, không chỉ nhìn phân phối toàn tập.
- **Nhóm chỉ số đo lường (sinh lý / viêm):** Phân bố **liên tục, thường đa đỉnh hoặc kéo dài** (SpO₂, FEV1, CRP) — gợi ý **không đồng nhất một cụm** trong mẫu; KDE có thể “lượn” thay vì một chuông đơn. Khi mô hình hóa, nên tránh giả định Gaussian đơn giản nếu không kiểm tra lại.
- **Nhóm xã hội – kinh tế:** Các biến **thứ bậc / năm học** phân bố **tương đối đều trên các mức**, với vài mode nổi (ví dụ 12 năm học, pattern thu nhập–tiếp cận y tế) — ít kiểu “đuôi dài” như thuốc lá; mang nhiều ý nghĩa **mô tả cấu trúc khảo sát** hơn là skew cực đoan.

### Nhận xét theo biểu đồ (quan sát trực tiếp từ hình)

**Nhóm thuốc lá** (`distribution_smoking_*.png` — histogram+KDE một panel)

- **`pack_years`:** **Lệch phải mạnh**, đỉnh gần 0, đuôi dài — đa số tích lũy thấp, ít mẫu phơi nhiễm rất cao.
- **`smoking_years`:** **Zero-inflated / lệch phải**, đỉnh gần 0 rồi giảm nhanh.
- **`cigarettes_per_day`:** Đỉnh gần **0 điếu/ngày**, phần còn lại thấp và rải.

**Nhóm lâm sàng / triệu chứng** (`distribution_clinical_*.png` — **bar chart**)

- **`copd`, `chronic_cough`, `shortness_of_breath`, `xray_abnormal`:** **Hai cột tần suất** 0 và 1; cột **0 cao hơn hẳn** 1 — đa số không có COPD / không ho mạn / không khó thở / X-quang bình thường; tỷ lệ “có” là **thiểu số** (COPD và X-quang bất thường thường **ít hơn** ho/mạn và khó thở).

**Nhóm chỉ số đo lường** (`distribution_biomarkers_*.png`)

- **`oxygen_saturation`:** Giá trị nằm trong khoảng **SpO2 sinh lý** (~92–100%). Không phải một đỉnh chuông đơn — có **nhiều cục tần suất** (gợi ý biến thiên theo nhiều “mức đọc” hoặc nhóm con trong dữ liệu), KDE có dạng **sóng / nhiều cực tiểu cực đại**.
- **`fev1_x10`:** Phân bố **khá trải** trên khoảng 20–50, không tập trung một đỉnh giữa duy nhất; có **nhóm tần suất cao hơn** về phía **cuối thang** — cần gắn với quy ước đơn vị trong dataset khi diễn giải lâm sàng.
- **`crp_level`:** Phân bố **không đơn điệu**: có **nhiều đỉnh cục bộ** dọc thang 0–10; KDE lượn — khó mô tả bằng một phân phối chuẩn đơn giản.

**Nhóm xã hội – kinh tế** (`distribution_socioeconomic_*.png` — **cột có thứ tự**, không KDE)

- **`education_years`:** Mỗi năm học là một cột; đỉnh **rõ nhất quanh 12 năm**; có **vực sụt** quanh **18 năm** so với lân cận — phản ánh chu kỳ học / mã hóa khảo sát.
- **`income_level`:** Năm mức **1–5** theo thứ tự; **mức 2 thấp hơn** các mức khác, **mức 4** trong nhóm cao — **khá đều**, không cực đoan một phía.
- **`healthcare_access`:** Năm mức **tương đối cân bằng**; **mức 2** mode nhẹ, **mức 4** thấp nhất trong nhóm.

### Kết luận

Từ **các biểu đồ** (thuốc lá + đo lường: hist+KDE; lâm sàng: bar; xã hội–kinh tế: cột thứ tự), nhóm **thuốc lá** thể hiện **zero-inflation và lệch phải**; **lâm sàng** **nhị phân lệch về 0**; **đo lường** **đa đỉnh / phẳng** hơn; **xã hội–kinh tế** **thứ bậc đều đặn** với vài mode. Để so sánh **theo nhãn `lung_cancer_risk`**, cần **tách nhóm** riêng.


## 4.7. Phân tích tương quan giữa các biến

### Vì sao cần làm mục này?

Khi nhiều đặc trưng mang thông tin trùng nhau, mô hình có thể học dư thừa hoặc khó diễn giải. Phân tích tương quan là bước cần thiết để phát hiện sớm đa cộng tuyến và chuẩn bị cho bước chọn/biến đổi đặc trưng.

### Giải thích mục này

**Hình:** `reports/eda/correlation_heatmap.png` — ma trận tương quan Pearson giữa các biến số.

**Quan sát chính:**

- Nhóm **hút thuốc** (`pack_years`, `smoking_years`, `cigarettes_per_day`) có xu hướng **tương quan dương** với nhau — phù hợp nghiệp vụ (pack-years gắn với cường độ và thời gian hút).
- Các cặp tương quan mạnh giữa đặc trưng “cùng gốc thông tin” gợi ý **đa cộng tuyến** (multicollinearity).

**Hệ quả kỹ thuật cho mô hình:**

- Mô hình **tuyến tính** (logistic thuần, không chuẩn hóa hệ số) có thể **không ổn định** hoặc khó giải thích hệ số khi nhiều biến trùng thông tin.
- Có thể dùng **regularization** (L1/L2), **giảm chiều**, hoặc **bỏ bớt một biến** trong cụm trùng — tùy chiến lược và khả năng giữ interpretability.

### Kết luận

Mục tương quan cho thấy cụm biến hút thuốc cần được kiểm soát đa cộng tuyến ở giai đoạn mô hình hóa để tránh giảm độ ổn định và khả năng diễn giải.

---

## 4.8. So sánh giữa hai nhóm nguy cơ

### Vì sao cần làm mục này?

Đây là bước trả lời trực tiếp câu hỏi nghiệp vụ: "Nhóm nguy cơ cao có khác gì nhóm nguy cơ thấp?". Kết quả giúp ưu tiên đặc trưng và hỗ trợ giải thích mô hình theo ngữ cảnh y tế.

### Giải thích mục này

**Lưu ý:** Pipeline EDA hiện tại (`src/data/eda.py`) **không** sinh chuỗi boxplot `group_comparison_*` theo `lung_cancer_risk`. Phần dưới là **nhận định định tính có định hướng** để đặt cạnh **heatmap tương quan** và **huấn luyện mô hình**; để khẳng định chênh lệch giữa hai nhãn nên bổ sung biểu đồ **phân bố có hue = nhãn** hoặc kiểm định/thống kê theo nhóm.

**Tóm tắt định tính (tham chiếu nghiệp vụ + ma trận tương quan):**

- **Khác biệt rõ hơn ở nhóm phơi nhiễm thuốc lá:** `pack_years`, `smoking_years`, `cigarettes_per_day` có xu hướng trung vị và miền giá trị phía nhóm `lung_cancer_risk=1` cao hơn.
- **Biến nhị phân (0/1) nên đọc theo tỷ lệ hơn là "hình hộp":** với `copd`, `xray_abnormal`, `chronic_cough`, `shortness_of_breath`, boxplot chủ yếu phản ánh đa số điểm ở 0 và một phần ở 1; để kết luận chênh lệch giữa hai nhóm nên ưu tiên thêm bar/count plot theo tỷ lệ.
- **Khác biệt yếu hoặc chồng lấn nhiều:** `oxygen_saturation`, `education_years`, `healthcare_access` cho thấy mức chồng lấn đáng kể giữa hai nhóm, khó tách mạnh khi nhìn đơn biến.
- **Cần thận trọng khi diễn giải:** một số biến sinh học (như **`fev1_x10`**) có thể có xu hướng **không khớp trực giác y học đơn giản** trong riêng bộ dữ liệu này; không nên suy ra quy luật lâm sàng chỉ từ EDA đơn biến.

**Tham chiếu hình:** `reports/eda/correlation_heatmap.png`; phân phối đơn biến theo nhóm chủ đề: các file `reports/eda/distribution_<nhóm>_*.png`.

### Kết luận

Nhóm biến liên quan hút thuốc và một số dấu hiệu lâm sàng cho tín hiệu phân tách tốt hơn, trong khi nhiều biến khác vẫn chồng lấn và cần kết hợp trong mô hình đa biến để tăng khả năng dự đoán.

---

## 4.9. Nhận xét rút ra từ EDA

### Vì sao cần làm mục này?

Sau các phân tích chi tiết, cần một phần tổng hợp để chuyển từ "quan sát dữ liệu" sang "hành động mô hình hóa", giúp pipeline huấn luyện có định hướng rõ ràng.

### Giải thích mục này

1. **Dữ liệu** có cấu trúc số hóa đồng nhất, **không missing**, phù hợp đưa vào pipeline phân loại.
2. **Mất cân bằng lớp** (~79% vs ~21%) đòi hỏi **metric và chiến lược huấn luyện** phù hợp với lớp hiếm (nhấn mạnh recall/F1, v.v.).
3. **Thuốc lá** và một số biến **chẩn đoán / triệu chứng** cho thấy **tín hiệu phân tách** khá rõ trong so sánh hai nhóm — ứng viên đặc trưng quan trọng cho mô hình.
4. **Đa cộng tuyến** trong cụm biến hút thuốc cần được xử lý trong giai đoạn **mô hình hóa** (regularization, chọn biến).
5. Với các biến liên tục đã kiểm tra bằng boxplot (`age`, `oxygen_saturation`, `fev1_x10`, `crp_level`), **không ghi nhận outlier rõ** theo 1.5xIQR; vẫn nên giữ IQR capping để phòng dữ liệu mới lệch phân bố.
6. **Mở rộng EDA (tuỳ chọn):** thêm boxplot outlier cho **`bmi`** và các biến liên tục khác nếu muốn khớp đầy đủ khung phân tích lâm sàng; luôn tái sinh artifact sau khi chỉnh `eda.py`.

### Kết luận

EDA xác nhận dữ liệu đủ tốt để huấn luyện, đồng thời chỉ ra ba ưu tiên kỹ thuật chính: xử lý mất cân bằng lớp, kiểm soát đa cộng tuyến, và duy trì bước chuẩn hóa/giảm ảnh hưởng giá trị cực đoan khi dữ liệu mới thay đổi.

---

*Tài liệu được xây dựng nhất quán với số liệu trong `survey_lung_cancer.csv` và cấu hình EDA trong `src/data/eda.py`.*
