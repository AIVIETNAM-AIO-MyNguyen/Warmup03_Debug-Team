# 1. ToxiGuard AI và bài toán Toxic Comment Detection trong cộng đồng học tập trực tuyến

Khi lớp học dần chuyển lên môi trường trực tuyến, phần bình luận không còn chỉ là nơi đặt câu hỏi hay trao đổi kiến thức. Nó đã trở thành một phần của trải nghiệm học tập số, nơi phản ánh trực tiếp văn hóa giao tiếp của cộng đồng học tập. Tuy nhiên, song song với sự phát triển của các nền tảng học online là sự xuất hiện ngày càng nhiều của các bình luận mang tính công kích, xúc phạm hoặc gây thù ghét. Những nội dung này không chỉ làm giảm chất lượng thảo luận mà còn ảnh hưởng tiêu cực đến tâm lý và động lực học tập của người dùng.

Trong các diễn đàn học tập, chỉ cần một vài bình luận mang tính toxic cũng có thể khiến môi trường trao đổi trở nên căng thẳng và thiếu an toàn. Người học có thể ngại đặt câu hỏi, ngại tham gia thảo luận hoặc thậm chí rời bỏ cộng đồng vì cảm giác bị tấn công hoặc chế giễu. Đối với các nền tảng có số lượng người dùng lớn, việc kiểm duyệt thủ công toàn bộ bình luận gần như không khả thi vì tốn nhiều thời gian và nguồn lực. Điều này đặt ra nhu cầu về các hệ thống có khả năng hỗ trợ phát hiện nội dung độc hại một cách nhanh chóng và tự động hơn.

Từ thực tế đó, nhóm xây dựng ToxiGuard AI như một hệ thống hỗ trợ kiểm duyệt bình luận độc hại trong môi trường học tập trực tuyến. Mục tiêu của hệ thống không phải thay thế hoàn toàn con người trong việc kiểm duyệt, mà đóng vai trò như một lớp hỗ trợ giúp phát hiện sớm các nội dung có nguy cơ gây hại, từ đó giúp quản trị viên phản hồi nhanh và hiệu quả hơn, đồng thời duy trì môi trường học tập tích cực cho người dùng.

Để xây dựng một hệ thống như vậy, bài toán cốt lõi cần giải quyết là Toxic Comment Detection — tức phát hiện các bình luận độc hại bằng trí tuệ nhân tạo. Toxic comment có thể được hiểu là các bình luận mang tính xúc phạm, công kích, đe dọa, thù ghét hoặc gây ảnh hưởng tiêu cực đến người khác trong môi trường trực tuyến. Đây là một bài toán thuộc lĩnh vực Natural Language Processing (NLP), nơi máy tính được huấn luyện để hiểu và phân tích ngôn ngữ con người. Cụ thể hơn, toxic comment detection thường được xem là một bài toán text classification, trong đó mô hình AI sẽ đọc nội dung bình luận và dự đoán xem bình luận đó có chứa yếu tố độc hại hay không.

Tuy nhiên, “độc hại” không phải lúc nào cũng chỉ có một dạng duy nhất. Trong nhiều bộ dữ liệu hiện đại, một bình luận có thể được gán nhiều nhãn khác nhau tùy theo mức độ và kiểu công kích. Trong đó, “toxic” thường được xem là nhãn tổng quát cho các nội dung gây hại hoặc tiêu cực. “Insult” đại diện cho các nội dung mang tính xúc phạm cá nhân, “threat” liên quan đến đe dọa hoặc gây sợ hãi, còn “obscene” dùng để chỉ các bình luận chứa ngôn từ tục tĩu hoặc phản cảm. Ngoài ra, một bình luận hoàn toàn có thể đồng thời thuộc nhiều nhóm nhãn khác nhau. Ví dụ, một câu chứa cả lời lăng mạ lẫn ngôn từ tục tĩu có thể vừa được gán nhãn insult vừa thuộc nhóm obscene.

Ví dụ, câu “You are stupid” có thể được xem là một bình luận mang tính xúc phạm trực tiếp. Trong khi đó, những câu mang tính mỉa mai hoặc công kích gián tiếp như “Wow, what an intelligent answer…” lại khó nhận diện hơn vì ý nghĩa thực sự phụ thuộc vào ngữ cảnh và sắc thái diễn đạt. Đây cũng là lý do vì sao toxic comment detection không đơn thuần là lọc từ cấm, mà cần đến các mô hình học máy có khả năng học được ngữ nghĩa của ngôn ngữ tự nhiên.

Trong bối cảnh giáo dục trực tuyến phát triển mạnh mẽ, việc ứng dụng AI vào kiểm duyệt nội dung không chỉ mang ý nghĩa công nghệ mà còn góp phần xây dựng một môi trường học tập tích cực, an toàn và tôn trọng lẫn nhau. ToxiGuard AI được phát triển với định hướng đó: hỗ trợ cộng đồng học tập duy trì chất lượng giao tiếp mà không làm gián đoạn trải nghiệm trao đổi kiến thức của người dùng.

# 2. Vì sao bài toán Toxic Comment Detection khó?

Mặc dù toxic comment detection là một trong những bài toán phổ biến trong NLP hiện nay, việc xây dựng một hệ thống có khả năng nhận diện chính xác bình luận độc hại trên thực tế lại không hề đơn giản. Nguyên nhân đến từ chính sự phức tạp của ngôn ngữ tự nhiên và cách con người giao tiếp trên môi trường trực tuyến.

Khó khăn đầu tiên nằm ở tính đa nghĩa của ngôn ngữ. Một câu nói có thể mang ý nghĩa hoàn toàn khác nhau tùy vào ngữ cảnh sử dụng. Ví dụ, một số từ ngữ tưởng chừng mang tính công kích có thể chỉ là cách nói đùa giữa bạn bè, trong khi những câu nghe có vẻ lịch sự đôi khi lại chứa hàm ý mỉa mai hoặc xúc phạm. Điều này khiến AI khó xác định chính xác đâu là toxic nếu chỉ dựa trên bề mặt câu chữ.
Ngoài ra, nhiều bình luận độc hại không mang tính tấn công trực tiếp mà được thể hiện dưới dạng sarcasm (mỉa mai), công kích ngầm hoặc thao túng cảm xúc. Con người có thể nhận ra sắc thái này nhờ kinh nghiệm giao tiếp và hiểu ngữ cảnh xã hội, nhưng đối với mô hình AI, đây là một thách thức lớn. Nếu không được huấn luyện đủ tốt, hệ thống có thể bỏ sót các bình luận nguy hiểm hoặc ngược lại gắn nhãn sai cho những bình luận bình thường.

Một vấn đề khác là dữ liệu thường bị mất cân bằng. Trong thực tế, số lượng bình luận bình thường luôn nhiều hơn rất nhiều so với bình luận toxic. Điều này khiến mô hình dễ học theo xu hướng “đa số” và ưu tiên dự đoán non-toxic để đạt accuracy cao. Tuy nhiên, một hệ thống kiểm duyệt hiệu quả không thể chỉ dựa vào accuracy, vì điều quan trọng hơn là khả năng phát hiện đúng các nội dung nguy hiểm.
Bên cạnh đó, toxic comment detection còn là bài toán multi-label classification. Một bình luận có thể đồng thời thuộc nhiều nhóm nhãn khác nhau. Ví dụ, một câu chứa lời lăng mạ và từ ngữ tục tĩu có thể vừa được gán nhãn insult vừa thuộc nhóm obscene. Điều này khiến mô hình phải học cách nhận diện nhiều sắc thái độc hại cùng lúc thay vì chỉ phân loại đơn giản thành “toxic” hoặc “non-toxic”.

Ngôn ngữ trên Internet cũng thay đổi liên tục. Người dùng thường viết tắt, cố tình sai chính tả hoặc sử dụng ký tự đặc biệt để né hệ thống kiểm duyệt. Ví dụ, các từ ngữ xúc phạm có thể được biến đổi thành nhiều dạng khác nhau như “i.d.i.o.t”, “1d10t” hoặc thay thế bằng emoji và meme. Nếu hệ thống không được cập nhật thường xuyên, khả năng nhận diện sẽ giảm đáng kể theo thời gian.

Chính vì những lý do đó, toxic comment detection không chỉ là bài toán kỹ thuật đơn thuần mà còn liên quan đến ngôn ngữ học, hành vi người dùng và bối cảnh giao tiếp xã hội. Một mô hình AI tốt không phải là mô hình “xóa sạch” mọi bình luận tiêu cực, mà là mô hình có khả năng hỗ trợ con người đưa ra quyết định kiểm duyệt chính xác và công bằng hơn.

# 4. Giới thiệu bộ dữ liệu

Môi trường học tập trực tuyến chỉ thực sự hiệu quả khi người học cảm thấy an toàn và được tôn trọng. Tuy nhiên, việc kiểm duyệt thủ công hàng ngàn thảo luận mỗi ngày là thử thách quá lớn đối với các nền tảng giáo dục. Để giải quyết bài toán này, chúng tôi đã lựa chọn bộ dữ liệu Jigsaw Toxic Comment Classification làm nền tảng huấn luyện cho ứng dụng phát hiện bình luận độc hại của cả nhóm.

## 4.1. Tổng quan về bộ dữ liệu

Bộ dữ liệu Jigsaw Toxic Comment Classification Challenge được cung cấp trên nền tảng Kaggle bởi Jigsaw, một công ty con thuộc tập đoàn Alphabet. Bản chất của tập dữ liệu này bao gồm khoảng 159.000 bình luận thực tế được thu thập từ các trang thảo luận của Wikipedia. Và trên hết, các bình luận này đều đã được người dùng gán dãn là độc hại hoặc không độc hại. Mục tiêu cốt lõi của Jigsaw khi công bố kho dữ liệu này là khuyến khích cộng đồng công nghệ xây dựng các mô hình xử lý ngôn ngữ tự nhiên có khả năng nhận diện và phân loại chính xác các sắc thái tiêu cực trong văn bản trực tuyến.

## 4.2. Cấu trúc các cột dữ liệu

Bộ dữ liệu được tổ chức theo dạng bảng, gồm 8 cột. Cột đầu tiên là `id`, đóng vai trò là mã định danh duy nhất cho từng mẫu dữ liệu. Cột quan trọng nhất chính là `comment_text`, chứa toàn bộ nội dung văn bản thô của bình luận cần đưa vào mô hình phân tích. Các cột còn lại bao gồm `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, và `identity_hate`. Đây là các cột nhãn nhị phân nhận giá trị `0` hoặc `1`, đại diện cho sự vắng mặt hoặc xuất hiện của từng loại hành vi độc hại tương ứng.

Trong đó:

- Toxic (Độc hại): Đại diện cho những phát ngôn thô lỗ, thiếu tôn trọng hoặc mang tính kích động ở mức độ cơ bản.
- Severe Toxic (Độc hại nghiêm trọng): Là cấp độ leo thang của nhãn trước, bao gồm các ngôn từ thù hận cực đoan và những đòn tấn công ác ý có chủ đích.
- Obscene (Tục tĩu): Tập trung vào việc sử dụng các từ ngữ chửi thề, thô tục hoặc các thuật ngữ không phù hợp với chuẩn mực văn hóa.
- Threat (Đe dọa): Ghi nhận các phát ngôn chứa hành vi bạo lực hoặc dọa dẫm gây tổn hại trực tiếp đến an toàn thân thể của người khác.
- Insult (Xúc phạm): Nhắm vào các hành vi lăng mạ, bôi nhọ danh dự hoặc hạ bệ uy tín của một cá nhân cụ thể trong cuộc thảo luận.
- Identity Hate (Thù ghét danh tính): Là những lời tấn công, kỳ thị dựa trên các đặc điểm cốt lõi như chủng tộc, tôn giáo, giới tính hoặc xu hướng tính dục.

## 4.3. Lí do chọn bộ dữ liệu

Chúng tôi quyết định tin tưởng bộ dữ liệu này vì nó mang lại độ bao phủ rất cao với 6 sắc thái độc hại được phân tách rõ ràng. Trong môi trường học thuật, học viên không chỉ cần tránh các lời chửi thề thô tục, mà còn cần được bảo vệ khỏi sự xúc phạm cá nhân hay thù ghét danh tính khi tranh luận.

Bên cạnh đó, do dữ liệu được lấy từ Wikipedia nên cấu trúc câu từ có sự tương đồng lớn với môi trường giáo dục, nơi người dùng thường viết các đoạn văn dài để trao đổi kiến thức thay vì chỉ dùng các câu khẩu ngữ ngắn như trên mạng xã hội thông thường. Việc sử dụng một bộ dữ liệu chuẩn hóa toàn cầu như Jigsaw sẽ giúp mô hình đạt được độ chính xác và độ bền vững cao nhất khi triển khai vào thực tế.

# 5. Phân tích Khám phá Dữ liệu (EDA)

Sau khi hiểu rõ cấu trúc lý thuyết, bước tiếp theo không thể thiếu là trực quan hóa dữ liệu để tìm ra các quy luật ẩn giấu. Việc phân tích này giúp chúng ta định hình chiến lược tiền xử lý dữ liệu và lựa chọn kiến trúc mô hình phù hợp nhất cho ứng dụng.

## 5.1. Tỷ lệ bình luận độc hại và lành mạnh

<p align="center">
  <img src=https://github.com/AIVIETNAM-AIO-MyNguyen/Warmup03_Debug-Team/blob/main/Collection/5_1_toxic_vs_nontoxic.png style="margin: 0 auto; display: block;"><br/>
  <em>Hình 5.1. Tỷ lệ bình luận độc hại và lành mạnh </em>
</p>

Biểu đồ đầu tiên phản ánh bức tranh toàn cảnh về sự cân bằng của tập dữ liệu thông qua tỷ lệ giữa bình luận độc hại (toxic) và bình luận lành mạnh (non-toxic). Kết quả trực quan hóa cho thấy một sự chênh lệch cực kỳ lớn khi nhóm bình luận lành mạnh chiếm đến khoảng 90% tổng số dữ liệu, trong khi nhóm chứa yếu tố độc hại chỉ chiếm khoảng 10%.

Sự mất cân bằng nghiêm trọng này là một đặc tính thực tế của các mạng xã hội nhưng lại là thách thức lớn cho AI. Nếu giữ nguyên tỷ lệ này để huấn luyện, mô hình sẽ có xu hướng đoán mọi bình luận đều là lành mạnh để đạt độ chính xác cao trên lý thuyết. Do đó, chúng ta bắt buộc phải áp dụng các kỹ thuật cân bằng dữ liệu như lấy mẫu lại (sampling) hoặc điều chỉnh trọng số hàm mất mát (loss weight) trong quá trình huấn luyện.

## 5.2. Sự phân bố của sáu nhãn tiêu cực

<p align="center">
  <img src=https://github.com/AIVIETNAM-AIO-MyNguyen/Warmup03_Debug-Team/blob/main/Collection/5_2_label_distribution.png style="margin: 0 auto; display: block;"><br/>
  <em>Hình 5.2. Tần suất xuất hiện của 6 nhãn độc hại </em>
</p>

Đi sâu hơn vào 10% bình luận tiêu cực, biểu đồ cột phân bố sáu nhãn sẽ bóc tách chi tiết tần suất xuất hiện của từng loại hành vi thù ghét. Nhãn `toxic` có số lượng vượt trội hoàn toàn so với năm nhãn còn lại, tiếp theo là nhãn `insult` (xúc phạm) và `obscene` (tục tĩu) với số lượng khá tương đồng nhau. Ngược lại, ba nhãn bao gồm `severe_toxic`, `identity_hate`, và `threat` có tần suất xuất hiện cực kỳ thấp, tạo thành các nhóm dữ liệu thiểu số trong tập dữ liệu.

Sự phân bố không đồng đều này chỉ ra rằng phần lớn hành vi vi phạm trên không gian mạng dừng lại ở mức độ thô lỗ hoặc xúc phạm lẫn nhau. Đối với ứng dụng học trực tuyến, việc nhãn `insult` xuất hiện nhiều cảnh báo chúng ta cần tập trung cao độ vào việc ngăn chặn hành vi hạ bệ, công kích cá nhân giữa các học viên nhằm bảo vệ không gian tranh luận lành mạnh.

## 5.3. Đặc điểm độ dài của các bình luận

<p align="center">
  <img src=https://github.com/AIVIETNAM-AIO-MyNguyen/Warmup03_Debug-Team/blob/main/Collection/5_3_comment_length_distribution.png style="margin: 0 auto; display: block;"><br/>
  <em>Hình 5.3. Phân phối độ dài bình luận (Giới hạn dưới 400 từ) </em>
</p>

Biểu đồ phân phối mật độ (Histogram) về độ dài ký tự và số lượng từ trong mỗi bình luận mang lại những góc nhìn kỹ thuật quan trọng. Hầu hết các bình luận tập trung dày đặc ở phân khúc ngắn, dao động từ vài chục đến dưới hai trăm từ. Tuy nhiên, biểu đồ cũng xuất hiện một chiếc "đuôi dài" kéo về phía bên phải, đại diện cho những bài viết có độ dài đột biến.

Đặc điểm này ảnh hưởng trực tiếp đến việc cấu hình tham số `max_length` khi số hóa văn bản (tokenization). Nếu chọn giới hạn quá ngắn, mô hình sẽ cắt bỏ nhiều ngữ cảnh quan trọng của các bài viết dài. Nếu chọn giới hạn quá dài, hệ thống sẽ lãng phí tài nguyên tính toán để xử lý các khoảng trống vô nghĩa (padding tokens) của các bình luận ngắn, làm chậm tốc độ phản hồi của ứng dụng kiểm duyệt thời gian thực.

## 5.4. Mối quan hệ tương quan giữa các nhãn

<p align="center">
  <img src=https://github.com/AIVIETNAM-AIO-MyNguyen/Warmup03_Debug-Team/blob/main/Collection/5_4_labels_correlation_heatmap.png style="margin: 0 auto; display: block;"><br/>
  <em>Hình 5.4. Biểu đồ ma trận tương quan giữa 6 nhãn độc hại </em>
</p>

Biểu đồ ma trận nhiệt (Heatmap) thể hiện hệ số tương quan Pearson giữa sáu nhãn độc hại mang lại cái nhìn sâu sắc về hành vi ngôn từ. Chỉ số tương quan mạnh nhất xuất hiện giữa hai cặp nhãn: `toxic` với `insult`, và `obscene` với `insult`. Ngược lại, nhãn threat hầu như không có sự tương quan đáng kể nào với các nhãn khác, đứng hoàn toàn độc lập trong ma trận.

Mối liên hệ hữu cơ này chứng minh rằng một người khi đã sử dụng từ ngữ tục tĩu (obscene) thì tỷ lệ rất cao là họ đang nhằm mục đích lăng mạ (insult) ai đó. Về mặt kỹ thuật, sự tương quan cao giữa các nhãn củng cố quyết định sử dụng mô hình phân loại đa nhãn (Multi-label), cho phép một bình luận kích hoạt đồng thời nhiều nhãn thay vì ép buộc mô hình phải chọn một nhãn duy nhất.

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

# Tài Liệu Tham Khảo

Jigsaw Toxic Comment Classification Challenge

Natural Language Processing Specialization – DeepLearning.AI

DeepLearning.AI. (n.d.). Natural Language Processing Specialization. Coursera. https://www.deeplearning.ai/courses/natural-language-processing-specialization/

Kaggle. (2018). Jigsaw Toxic Comment Classification Challenge. https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

Jurafsky, D., & Martin, J. H. (2023). Speech and Language Processing (3rd ed. draft). Stanford University. https://web.stanford.edu/~jurafsky/slp3/

Schmidt, A., & Wiegand, M. (2017). A survey on hate speech detection using natural language processing. Proceedings of the Fifth International Workshop on Natural Language Processing for Social Media, 1–10.
