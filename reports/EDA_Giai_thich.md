# 3. Giải thích các chỉ số trong EDA

Tài liệu chỉ **định nghĩa và ý nghĩa các chỉ số** được dùng trong phân tích khám phá dữ liệu. Các giá trị cụ thể (tần suất, trung vị…) lấy từ `reports/eda/eda_report.txt` và báo cáo chính `Bao_cao_EDA.md`.

---

## Chỉ số trong từng bước EDA

| Bước | Chỉ số / đại lượng | Ý nghĩa ngắn |
|:-----|:-------------------|:-------------|
| Cấu trúc | Số dòng \(N\), số cột \(p\) | Quy mô tập dữ liệu. |
| Cấu trúc | Kiểu cột (`int64` / `float64`) | Cách máy đọc số và quyết định vẽ thống kê nào. |
| Biến mục tiêu | Tần suất lớp 0 và 1, tỉ lệ % | Mất cân bằng lớp để chọn metric / cân bằng mẫu. |
| Thiếu dữ liệu | `%` missing theo cột | Cần imputation hay không. |
| Ngoại lai | Ngưỡng râu theo \(1{,}5 \times IQR\) (trên boxplot) | Giá trị xa vùng trung tâm theo định nghĩa IQR. |
| Tương quan | Hệ số tương quan Pearson | Mức độ đồng biến tuyến tính giữa hai biến số (−1 đến 1). |
| Phân phối đặc trưng | Trục X = giá trị chỉ số, trục Y = Count | Tần suất xuất hiện từng khoảng giá trị của chỉ số trên toàn tập. |

---

## Định nghĩa từng chỉ số trong hình `distribution_<nhóm>_*.png`

Loại biểu đồ phụ thuộc nhóm (sinh bởi `eda.py`): **thuốc lá** và **đo lường** — một panel histogram + KDE; **lâm sàng** — bar tần suất 0/1; **xã hội–kinh tế** — cột tần suất theo mức có thứ tự. Tất cả trên **toàn tập** (không tách hai nhãn `lung_cancer_risk` trên cùng một hình).

| File / biến | Chỉ số là gì | Thang / đơn vị trong dữ liệu |
|:------------|:-------------|:----------------------------|
| `pack_years` | Mức tích lũy phơi nhiễm thuốc lá (pack-years). | Liên tục (năm pack). |
| `smoking_years` | Số năm đã hút thuốc. | Nguyên (năm). |
| `cigarettes_per_day` | Trung bình điếu thuốc mỗi ngày. | Liên tục (điếu/ngày). |
| `copd` | Có hay không COPD. | Nhị phân 0/1. |
| `chronic_cough` | Có hay không ho mạn tính. | Nhị phân 0/1. |
| `shortness_of_breath` | Có hay không khó thở. | Nhị phân 0/1. |
| `xray_abnormal` | Có hay không ảnh X-quang bất thường. | Nhị phân 0/1. |
| `oxygen_saturation` | Độ bão hòa oxy máu \(SpO2\). | % (≈ 90–100). |
| `fev1_x10` | Thể tích thở ra trong 1 giây \(FEV1\) (đã scale ×10 trong tên biến). | Liên tục theo đơn vị dataset. |
| `crp_level` | Nồng độ hoặc mức CRP — chỉ báo viêm. | Liên tục (đơn vị theo quy ước dataset). |
| `education_years` | Tổng số năm học. | Nguyên (năm). |
| `income_level` | Mức thu nhập qua điểm thứ bậc. | Thứ bậc \(1\) thấp–\(5\) cao (theo khảo sát). |
| `healthcare_access` | Mức tiếp cận dịch vụ y tế qua điểm thứ bậc. | Thứ bậc \(1\)–\(5\) (theo khảo sát). |

---

## Biến mục tiêu (chỉ số nhãn)

| Biến | Ý nghĩa | Giá trị |
|:-----|:--------|:--------|
| `lung_cancer_risk` | Nguy cơ ung thư phổi theo gán nhãn trong dữ liệu. | 0 = không / thấp, 1 = có / cao (theo bộ khảo sát). |

---

## Ghi chú ngắn về các file khác trong `reports/eda/`

- `target_distribution.png`: chỉ là **đếm** theo hai lớp `lung_cancer_risk`.
- `outlier_boxplot_<tên>.png`: `age`, `oxygen_saturation`, `fev1_x10`, `crp_level` — xem median, quartiles và khoảng râu theo \(1{,}5 \times IQR\).
- `correlation_heatmap.png`: ma trận Pearson giữa mọi cột số; thanh màu thang −1 đến 1.

Chi tiết số và nhận xét tổng hợp: xem `Bao_cao_EDA.md`.
