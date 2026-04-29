# 3. PHAN TICH DU LIEU KHAM PHA (EDA)

Tai lieu nay tong hop ket qua EDA cho bo du lieu trong `data/raw/survey_lung_cancer.csv`, dua tren cac artifact da sinh o `reports/eda/`.

## 1. Kiem tra kich thuoc du lieu va kieu du lieu cua tung bien

- **Kich thuoc du lieu**: `1000` dong, `30` cot.
- **Cau truc bien**:
  - Bien nhi phan (dang `int64`): `gender`, `smoker`, `copd`, `xray_abnormal`, ...
  - Bien lien tuc/so hoc (`float64`): `pack_years`, `oxygen_saturation`, `fev1_x10`, `crp_level`, ...
  - Bien muc tieu (`int64`): `lung_cancer_risk`.

Nhan xet:
- Du lieu da duoc chuan hoa kieu du lieu kha tot cho bai toan phan loai.
- Khong co cot dang text/chuoi bat thuong, nen phu hop de dua vao pipeline tien xu ly va huan luyen.

## 2. Phan tich phan phoi cua bien muc tieu `lung_cancer_risk`

Hinh tham chieu: `reports/eda/target_distribution.png`

So lieu:
- Nhom `0` (nguy co thap/khong mac): `791` mau (`79.1%`)
- Nhom `1` (nguy co cao/mac): `209` mau (`20.9%`)

Nhan xet:
- Du lieu **mat can bang lop** (class imbalance), lop 0 nhieu hon ro ret.
- Khi huan luyen mo hinh, can uu tien metric nhu `recall`, `f1`, `roc-auc` va xem xet ky thuat can bang du lieu (vi du: SMOTE, class_weight).

## 3. Kiem tra missing values theo tung cot

Hinh tham chieu: `reports/eda/missing_values_percentage.png` (neu co)

Ket qua:
- Tat ca cac cot deu co `0` gia tri thieu.
- Ti le missing cua moi cot deu bang `0.0%`.

Nhan xet:
- Khong can xu ly missing trong tap du lieu hien tai.
- Van nen giu buoc kiem tra missing trong pipeline de dam bao an toan khi nhan du lieu moi.

## 4. Phat hien outliers o cac bien lien tuc

Hinh tham chieu:
- `reports/eda/outlier_boxplot_age.png`
- `reports/eda/outlier_boxplot_oxygen_saturation.png`
- `reports/eda/outlier_boxplot_fev1_x10.png`
- `reports/eda/outlier_boxplot_crp_level.png`

Nhan xet tong quan:
- Cac bien lien tuc deu co do phan tan nhat dinh.
- `age` va `crp_level` co kha nang xuat hien diem xa tam phan bo (outlier nhe).
- `oxygen_saturation` va `fev1_x10` phan bo tuong doi tap trung hon, nhung van can kiem tra gioi han sinh hoc.

Y nghia xu ly:
- Outlier co the anh huong den cac mo hinh nhay cam voi thang do.
- Pipeline hien tai da co buoc **IQR capping** trong preprocess de han che anh huong cua cac gia tri cuc doan.

## 5. Phan tich tuong quan giua cac bien so

Hinh tham chieu: `reports/eda/correlation_heatmap.png`

Diem can chu y:
- Nhom bien hut thuoc (`pack_years`, `smoking_years`, `cigarettes_per_day`) co xu huong tuong quan voi nhau.
- Dieu nay phu hop nghiep vu vi `pack_years` duoc tao tu cuong do + thoi gian hut thuoc.

Nhan xet:
- Can can nhac da cong tuyen khi dung cac mo hinh tuyen tinh.
- Co the giam bot bien trung lap hoac dung regularization de han che overfitting.

## 6. So sanh phan phoi bien quan trong giua 2 nhom nguy co

Cac hinh so sanh:
- `reports/eda/group_comparison_pack_years.png`
- `reports/eda/group_comparison_smoking_years.png`
- `reports/eda/group_comparison_cigarettes_per_day.png`
- `reports/eda/group_comparison_copd.png`
- `reports/eda/group_comparison_chronic_cough.png`
- `reports/eda/group_comparison_shortness_of_breath.png`
- `reports/eda/group_comparison_xray_abnormal.png`
- `reports/eda/group_comparison_oxygen_saturation.png`
- `reports/eda/group_comparison_fev1_x10.png`
- `reports/eda/group_comparison_crp_level.png`
- `reports/eda/group_comparison_education_years.png`
- `reports/eda/group_comparison_income_level.png`
- `reports/eda/group_comparison_healthcare_access.png`

Nhan xet tong hop:
- Nhom nguy co cao thuong co cac chi so lien quan den hut thuoc va benh ho hap bat loi hon.
- Cac bien lam sang/chan doan nhu `copd`, `chronic_cough`, `shortness_of_breath`, `xray_abnormal` co kha nang phan tach nhom tot.
- Cac bien sinh hoc (`oxygen_saturation`, `fev1_x10`, `crp_level`) co y nghia lam sang va can duoc uu tien theo doi.
- Nhom yeu to xa hoi (`education_years`, `income_level`, `healthcare_access`) co the tac dong gian tiep den nguy co, can danh gia them trong mo hinh da bien.

---

## Ket luan EDA ngan gon

- Du lieu sach (khong missing), cau truc ro rang, phu hop cho bai toan phan loai.
- Bien muc tieu mat can bang, can uu tien chien luoc toi uu `recall`.
- Nhom bien hut thuoc va trieu chung ho hap cho thay tin hieu du doan manh.
- Nen tiep tuc buoc feature engineering va kiem soat da cong tuyen trong giai doan mo hinh hoa.
