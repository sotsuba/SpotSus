# __ADVERSARIAL VALIDATION__
# __01 - Mục tiêu__

- Xác định covariate shift giữa tập train và test trong dataset IEEE-CIS về Fraud Detection.

- Phát hiện các đặc trưng bị time leakage, từ đó có thể chọn bộ feature đảm bảo tính tổng quát hóa của mô hình.

# __02 - Phương pháp luận__

## 02.01 - Về dữ liệu

- Gộp hai tập train (bỏ feature `isFraud`) và test lại, đồng thời gán nhãn `is_test` để chỉ ra record này là của tập dữ liệu nào.

## 02.02 - Về mô hình

`CatboostClassifier` được chọn để giải quyết các mục tiêu nêu trên vì:

- Chống overfit tốt nhờ kiến trúc Oblivious Trees.

- Bỏ qua được bước điền khuyết dữ liệu; chuẩn hóa dữ liệu.

- Mạnh với dữ liệu có categorical features.

Đầu vào: tập dữ liệu.

Đầu ra: `is_test` (0 cho train, 1 cho test).

## 02.03 - Về metrics

Hiện tại, tôi sử dụng duy nhất một primary metric là `AUC` với kì vọng như sau:

| AUC | Đánh giá |
|-----|----------|
| ~0.5 | Chứng tỏ tập train và test gần như tương đồng với nhau, đảm bảo kết quả mô hình tương đồng giữa train và test. Đây là điều kiện lý tưởng nhất |
| 0.75 <= x < 0.85 | Có data drift đáng kể, cần phải tiếp tục xử lý khi có thể |
| x >= 0.85 | Data drift nghiêm trọng, cần phải giải quyết ngay |

# __03 - Kiểm thử__

## Giai đoạn 1: Baseline trên dữ liệu thô
**Ban đầu:** AUC score ~0.99

**Top Features gây nhiễu:**

- `TransactionDT`: 

- `TransactionID`

**Nhận xét:** Mô hình phân biệt tuyệt đối dựa trên biến thời gian và index.

**Hành động:** Loại bỏ TransactionDT và TransactionID khỏi tập feature.

**Kết quả:** AUC Score giảm xuống còn ~0.85.

## Giai đoạn 2: Tổng quát hóa id_31
**Ban đầu:** AUC score ~0.85

| Top Features gây nhiễu | Mô tả  | Nguyên nhân | Hành động |
| :--- | :--- | :--- | :--- |
| **id_31** | Thông tin `user-agent` (Trình duyệt/Thiết bị). | **Version Shift:** *VD: Train thì có Chrome 63.0, trong khi Test có Chrome 68.0*. | **Generalization:** Loại bỏ yếu tố version, chỉ giữ lại tên định danh gốc của trình duyệt/thiết bị. |
| **D15** | Biến timedelta cho một cái gì đó. | **(Tạm hoãn):** Chưa hiểu ý nghĩa của feature này | Tạm thời loại bỏ khỏi tập feature để tránh rò rỉ thông tin thời gian. Cần nghiên cứu phương pháp xử lý chuyên sâu sau. |

**Kết quả:** AUC Score giảm ~0.85 -> ~0.82. Kết quả mới tạm chấp nhận được.

# __Nguồn tham khảo__

## Blogs:

- [Adversarial validation, part one - FastML](http://fastml.com/adversarial-validation-part-one/)

- [Adversarial validation, part one - FastML](http://fastml.com/adversarial-validation-part-two/)

## Youtube:

- [How to use Adversarial Validation to Help Fix Overfitting - WelcomeAIOverlords](https://www.youtube.com/watch?v=7cUCDRaIZ7I)

- [Introduction to Adversarial Validation In Machine Learning - Underfitted](https://www.youtube.com/watch?v=c5ClgEhAhK0)