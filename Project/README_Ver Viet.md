# 6. Pipeline tổng thể của hệ thống ToxiGuard AI
Trước khi đi vào chi tiết từng bước xử lý dữ liệu hay xây dựng mô hình, mình muốn nhìn toàn bộ hệ thống dưới dạng một pipeline hoàn chỉnh. Điều này giúp dễ hình dung từ một comment sẽ đi qua những bước nào trước khi đưa ra dự đoán cuối cùng.

Về cơ bản, pipeline của Toxic Comment Detection hoạt động theo flow như sau:
<p align="center">
  <img src=https://github.com/AIVIETNAM-AIO-MyNguyen/Warmup03_Debug-Team/blob/main/Collection/6_pipeline.png style="margin: 0 auto; display: block;"><br/>
  <em>Hình 6. Toxic Comment Detection pipeline </em>
</p>

## 6.1 Dữ liệu đầu vào

Bước đầu tiên chính là dữ liệu comment được thu thập từ dataset của cuộc thi Jigsaw Toxic Comment Classification Challenge.

Mỗi dòng dữ liệu chứa một đoạn bình luận và các nhãn tương ứng như:

- toxic
- severe_toxic
- obscene
- threat
- insult
- identity_hate

Điểm đặc biệt của bài toán này là một comment có thể thuộc nhiều nhãn cùng lúc. Ví dụ một câu vừa mang tính xúc phạm, vừa chứa nội dung thù ghét. Vì vậy đây không phải bài toán phân loại thông thường mà là multi-label classification.

## 6.2 Text Preprocessing

Dữ liệu text ngoài thực tế đặc biệt là từ comment thường rất lộn xộn. Người dùng có thể viết hoa toàn bộ câu, spam ký tự đặc biệt, viết tắt hoặc dùng ngôn ngữ thiếu chuẩn mực. Nếu đưa trực tiếp vào model thì hiệu quả dự đoán thường không tốt.

Vì vậy trước khi huấn luyện, nhóm mình thực hiện bước preprocessing để làm sạch dữ liệu, bao gồm:

- Chuyển toàn bộ text về lowercase
- Loại bỏ ký tự đặc biệt
- Xóa các pattern không mang ý nghĩa đánh giá như: URL, HTML, username
- Tokenize câu thành từng word riêng biệt.
- Loại bỏ stopwords

Bước này giúp model tập trung vào phần nội dung mang nhiều ý nghĩa nhất thay vì bị nhiễu bởi các ký tự dư thừa.

## 6.3 Vectorization

Trong project này, nhóm mình sử dụng TF-IDF Vectorization để biểu diễn mỗi comment dưới dạng vector số dựa trên mức độ quan trọng của từng từ trong tập dữ liệu.

Những từ xuất hiện phổ biến ở hầu hết comment sẽ có trọng số thấp hơn, trong khi các từ mang tính công kích thường sẽ có trọng số cao hơn.

## 6.4 Classification Model

Sau bước vectorization, dữ liệu sẽ được đưa vào mô hình machine learning để học các pattern liên quan đến toxic behavior.

Trong project này:

- Naive Bayes được dùng làm baseline model
- Logistic Regression được chọn làm mô hình chính

Ở bước cuối cùng, model sẽ trả về xác suất cho từng nhãn độc hại.

Dựa trên threshold được thiết lập, hệ thống sẽ quyết định comment thuộc những nhãn nào.