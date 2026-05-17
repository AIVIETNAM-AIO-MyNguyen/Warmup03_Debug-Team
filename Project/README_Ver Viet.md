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

- **Toxic (Độc hại):** Đại diện cho những phát ngôn thô lỗ, thiếu tôn trọng hoặc mang tính kích động ở mức độ cơ bản.
- **Severe Toxic (Độc hại nghiêm trọng):** Là cấp độ leo thang của nhãn trước, bao gồm các ngôn từ thù hận cực đoan và những đòn tấn công ác ý có chủ đích.
- **Obscene (Tục tĩu):** Tập trung vào việc sử dụng các từ ngữ chửi thề, thô tục hoặc các thuật ngữ không phù hợp với chuẩn mực văn hóa.
- **Threat (Đe dọa):** Ghi nhận các phát ngôn chứa hành vi bạo lực hoặc dọa dẫm gây tổn hại trực tiếp đến an toàn thân thể của người khác.
- **Insult (Xúc phạm):** Nhắm vào các hành vi lăng mạ, bôi nhọ danh dự hoặc hạ bệ uy tín của một cá nhân cụ thể trong cuộc thảo luận.
- **Identity Hate (Thù ghét danh tính):** Là những lời tấn công, kỳ thị dựa trên các đặc điểm cốt lõi như chủng tộc, tôn giáo, giới tính hoặc xu hướng tính dục.

## 4.3. Lí do chọn bộ dữ liệu

Chúng tôi quyết định tin tưởng bộ dữ liệu này vì nó mang lại độ bao phủ rất cao với 6 sắc thái độc hại được phân tách rõ ràng. Trong môi trường học thuật, học viên không chỉ cần tránh các lời chửi thề thô tục, mà còn cần được bảo vệ khỏi sự xúc phạm cá nhân hay thù ghét danh tính khi tranh luận.

Bên cạnh đó, do dữ liệu được lấy từ Wikipedia nên cấu trúc câu từ có sự tương đồng lớn với môi trường giáo dục, nơi người dùng thường viết các đoạn văn dài để trao đổi kiến thức thay vì chỉ dùng các câu khẩu ngữ ngắn như trên mạng xã hội thông thường. Việc sử dụng một bộ dữ liệu chuẩn hóa toàn cầu như Jigsaw sẽ giúp mô hình đạt được độ chính xác và độ bền vững cao nhất khi triển khai vào thực tế.

# 5. Phân tích Khám phá Dữ liệu (EDA)

Sau khi hiểu rõ cấu trúc lý thuyết, bước tiếp theo không thể thiếu là trực quan hóa dữ liệu để tìm ra các quy luật ẩn giấu. Việc phân tích này giúp chúng ta định hình chiến lược tiền xử lý dữ liệu và lựa chọn kiến trúc mô hình phù hợp nhất cho ứng dụng.

## 5.1. Tỷ lệ bình luận độc hại và lành mạnh

<p align="center">
  <img src=https://github.com/AIVIETNAM-AIO-MyNguyen/Warmup03_Debug-Team/blob/main/Collection/5_1_toxic_vs_nontoxic.png style="margin: 0 auto; display: block;"><br/>
  <em>Hình 5.1. Tỷ lệ bình luận độc hại và lành mạnh</em>
</p>

Biểu đồ đầu tiên phản ánh bức tranh toàn cảnh về sự cân bằng của tập dữ liệu thông qua tỷ lệ giữa bình luận độc hại (toxic) và bình luận lành mạnh (non-toxic). Kết quả trực quan hóa cho thấy một sự chênh lệch cực kỳ lớn khi nhóm bình luận lành mạnh chiếm đến khoảng 90% tổng số dữ liệu, trong khi nhóm chứa yếu tố độc hại chỉ chiếm khoảng 10%.

Sự mất cân bằng nghiêm trọng này là một đặc tính thực tế của các mạng xã hội nhưng lại là thách thức lớn cho AI. Nếu giữ nguyên tỷ lệ này để huấn luyện, mô hình sẽ có xu hướng đoán mọi bình luận đều là lành mạnh để đạt độ chính xác cao trên lý thuyết. Do đó, chúng ta bắt buộc phải áp dụng các kỹ thuật cân bằng dữ liệu như lấy mẫu lại (sampling) hoặc điều chỉnh trọng số hàm mất mát (loss weight) trong quá trình huấn luyện.

## 5.2. Sự phân bố của sáu nhãn tiêu cực

<p align="center">
  <img src=https://github.com/AIVIETNAM-AIO-MyNguyen/Warmup03_Debug-Team/blob/main/Collection/5_2_label_distribution.png style="margin: 0 auto; display: block;"><br/>
  <em>Hình 5.2. Tần suất xuất hiện của 6 nhãn độc hại</em>
</p>

Đi sâu hơn vào 10% bình luận tiêu cực, biểu đồ cột phân bố sáu nhãn sẽ bóc tách chi tiết tần suất xuất hiện của từng loại hành vi thù ghét. Nhãn `toxic` có số lượng vượt trội hoàn toàn so với năm nhãn còn lại, tiếp theo là nhãn `insult` (xúc phạm) và `obscene` (tục tĩu) với số lượng khá tương đồng nhau. Ngược lại, ba nhãn bao gồm `severe_toxic`, `identity_hate`, và `threat` có tần suất xuất hiện cực kỳ thấp, tạo thành các nhóm dữ liệu thiểu số trong tập dữ liệu.

Sự phân bố không đồng đều này chỉ ra rằng phần lớn hành vi vi phạm trên không gian mạng dừng lại ở mức độ thô lỗ hoặc xúc phạm lẫn nhau. Đối với ứng dụng học trực tuyến, việc nhãn `insult` xuất hiện nhiều cảnh báo chúng ta cần tập trung cao độ vào việc ngăn chặn hành vi hạ bệ, công kích cá nhân giữa các học viên nhằm bảo vệ không gian tranh luận lành mạnh.

## 5.3. Đặc điểm độ dài của các bình luận

<p align="center">
  <img src=https://github.com/AIVIETNAM-AIO-MyNguyen/Warmup03_Debug-Team/blob/main/Collection/5_3_comment_length_distribution.png style="margin: 0 auto; display: block;"><br/>
  <em>Hình 5.3. Phân phối độ dài bình luận (Giới hạn dưới 400 từ)</em>
</p>

Biểu đồ phân phối mật độ (Histogram) về độ dài ký tự và số lượng từ trong mỗi bình luận mang lại những góc nhìn kỹ thuật quan trọng. Hầu hết các bình luận tập trung dày đặc ở phân khúc ngắn, dao động từ vài chục đến dưới hai trăm từ. Tuy nhiên, biểu đồ cũng xuất hiện một chiếc "đuôi dài" kéo về phía bên phải, đại diện cho những bài viết có độ dài đột biến.

Đặc điểm này ảnh hưởng trực tiếp đến việc cấu hình tham số `max_length` khi số hóa văn bản (tokenization). Nếu chọn giới hạn quá ngắn, mô hình sẽ cắt bỏ nhiều ngữ cảnh quan trọng của các bài viết dài. Nếu chọn giới hạn quá dài, hệ thống sẽ lãng phí tài nguyên tính toán để xử lý các khoảng trống vô nghĩa (padding tokens) của các bình luận ngắn, làm chậm tốc độ phản hồi của ứng dụng kiểm duyệt thời gian thực.

## 5.4. Mối quan hệ tương quan giữa các nhãn

<p align="center">
  <img src=https://github.com/AIVIETNAM-AIO-MyNguyen/Warmup03_Debug-Team/blob/main/Collection/5_4_labels_correlation_heatmap.png style="margin: 0 auto; display: block;"><br/>
  <em>Hình 5.4. Biểu đồ ma trận tương quan giữa 6 nhãn độc hại</em>
</p>

Biểu đồ ma trận nhiệt (Heatmap) thể hiện hệ số tương quan Pearson giữa sáu nhãn độc hại mang lại cái nhìn sâu sắc về hành vi ngôn từ. Chỉ số tương quan mạnh nhất xuất hiện giữa hai cặp nhãn: `toxic` với `insult`, và `obscene` với `insult`. Ngược lại, nhãn threat hầu như không có sự tương quan đáng kể nào với các nhãn khác, đứng hoàn toàn độc lập trong ma trận.

Mối liên hệ hữu cơ này chứng minh rằng một người khi đã sử dụng từ ngữ tục tĩu (obscene) thì tỷ lệ rất cao là họ đang nhằm mục đích lăng mạ (insult) ai đó. Về mặt kỹ thuật, sự tương quan cao giữa các nhãn củng cố quyết định sử dụng mô hình phân loại đa nhãn (Multi-label), cho phép một bình luận kích hoạt đồng thời nhiều nhãn thay vì ép buộc mô hình phải chọn một nhãn duy nhất.

## 5.5. Các thông tin dư thừa

Bộ dữ liệu Jigsaw Toxic Comment Classification Challenge tổng hợp các bình luận trên Wikipedia nên không thể tránh khỏi việc chúng có thể chứa một số thông tin nhạy cảm, thiên vị hoặc dư thừa. Cụ thể, một số thông tin như địa chỉ IP hay tên người dùng có thể tiết lộ danh tính thật của ai đó; hay các đường dẫn nội bộ của Wikipedia có dạng như `Wikipedia:...`, `Help:...`, `File:...` không có ý nghĩa trong các ngữ cảnh khác ngoại trừ trên Wikipedia; hoặc các URL hay thẻ HTML cũng không mang ý nghĩa cảm xúc hay độc hại, mà chúng chỉ làm tăng từ vựng rác cho mô hình.

Việc nhận diện và loại bỏ chúng có thể sẽ giúp mô hình tránh bị overfitting và tăng độ hiệu quả của quá trình huấn luyện. Sau đây là một số các thông tin nhạy cảm và dưa thừa có trong tập dữ liệu:

- **Escape Sequence:** các kí tự chỉ mang tính chất định dạng như xuống dòng, tab
- **Wiki link:** các đường dẫn nội bộ của Wikipedia, chỉ có nghĩa trên Wikipedia
- **URL:** các đường dẫn http không mang giá trị cảm xúc hay độc hại
- **Hashtag:** một thẻ siêu dữ liệu dùng để nhóm, phân loại chủ đề được nhắc đến
- **Email:** địa chỉ email của người dùng, tiết lộ thông tin của người dùng đó
- **Địa chỉ IP:** địa chỉ IP, tương tự như email đây cũng là thông tin cá nhân cần phải bảo mật
- **Mention:** các đoạn bình luận có dạng `@username` dùng để trả lời, đề cập đến một cá nhân nào đó
- **HTML tag:** các thẻ định dạng sử dụng HTML, không có giá trị cảm xúc

<p align="center">
  <img src=https://github.com/AIVIETNAM-AIO-MyNguyen/Warmup03_Debug-Team/blob/main/Collection/5_5_redundant_information_distribution.png style="margin: 0 auto; display: block;"><br/>
  <em>Hình 5.5. Biểu đồ tần suất xuất hiện của các thông tin nhạy cảm, dư thừa</em>
</p>

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

# 7. Tiền xử lý văn bản và Vectorization

Trong các bài toán NLP, dữ liệu text thường không thể đưa trực tiếp vào mô hình machine learning hay deep learning. Với bài toán toxic comment detection, comment ngoài thực tế thường chứa rất nhiều ký tự nhiễu, viết tắt hoặc cách diễn đạt không chuẩn.

Vì vậy trước khi huấn luyện model, nhóm mình thực hiện hai bước quan trọng:

- **Text Preprocessing**
- **Vectorization**

Hai bước này giúp chuyển đổi dữ liệu từ ngôn ngữ tự nhiên sang dạng số để AI có thể hiểu và xử lý hiệu quả hơn.

## 7.1 Preprocessing

**Vì sao cần preprocessing?**

Dữ liệu comment trên internet thường không đồng nhất và chứa khá nhiều nhiễu. Người dùng có thể:

- Viết hoa toàn bộ câu
- Spam ký tự hoặc lặp chữ liên tục
- Chèn username, link hoặc địa chỉ IP
- Dùng từ viết tắt như "you're", "can't"
- Viết sai chính tả hoặc không đúng ngữ pháp

Những yếu tố này khiến dữ liệu trở nên khó xử lý hơn đối với mô hình machine learning. Nếu giữ nguyên dữ liệu gốc, model rất dễ học phải các pattern không thực sự mang ý nghĩa, từ đó làm giảm khả năng tổng quát hóa trên dữ liệu mới.

Vì vậy, trước khi vector hóa, mình xây dựng một hàm clean() để chuẩn hóa comment về cùng một định dạng, giúp dữ liệu sạch và nhất quán hơn cho quá trình huấn luyện.

**Các bước preprocessing được sử dụng**

***Chuyển text về lowercase***

Đầu tiên, toàn bộ comment được chuyển về chữ thường:

```python
comment = comment.lower()
```

Điều này giúp "HATE" và "hate" được xem là cùng một từ thay vì hai token khác nhau.

***Loại bỏ ký tự xuống dòng***
```python
comment = re.sub("\\n"," ",comment)
```

Một số comment chứa ký tự xuống dòng \n, có thể làm dữ liệu bị phân mảnh không cần thiết. Vì vậy nhóm mình thay chúng bằng khoảng trắng.

***Xóa địa chỉ IP và pattern dư thừa***

```python
comment=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"," ",comment)
```

Một số comment có thể chứa IP address hoặc các pattern không liên quan đến nội dung toxic. Những dữ liệu này thường không mang nhiều ý nghĩa về mặt cảm xúc nên được loại bỏ để giảm nhiễu.

***Xóa username***

```python
comment=re.sub("\[\[.*\]"," ",comment)
```

Tên người dùng thường không giúp ích cho việc xác định toxic behavior. Nếu giữ lại, model có thể học nhầm vào username thay vì nội dung comment thực sự.

***Chuẩn hóa từ viết tắt***

```python
words=[APPO[word] if word in APPO else word for word in words]
```

Nhiều comment sử dụng dạng viết tắt như:

"you're" → "you are"
"can't" → "can not"

Việc chuẩn hóa giúp model hiểu dữ liệu đồng nhất hơn.

***Loại bỏ stopwords***

```python
words = [w for w in words if not w in eng_stopwords]
```

Các từ như "the", "is", "and" xuất hiện rất thường xuyên nhưng không mang nhiều giá trị phân loại, nên được loại bỏ để model tập trung vào những từ quan trọng hơn.


## 7.2 Vectorization

***Vì sao cần vectorization?***

Sau bước preprocessing, dữ liệu vẫn đang ở dạng text. Tuy nhiên machine learning model không thể hiểu trực tiếp ngôn ngữ tự nhiên.

Vì vậy cần chuyển comment sang dạng vector số trước khi đưa vào mô hình học máy.

Trong bài toán này, mình sử dụng kỹ thuật TF-IDF Vectorization.

<p align="center">
  <img src=https://github.com/AIVIETNAM-AIO-MyNguyen/Warmup03_Debug-Team/blob/main/Collection/7_1_tfidf.png style="margin: 0 auto; display: block;"><br/>
  <em>Hình 7.1 TF-IDF</em>
</p>

**TF-IDF hoạt động như thế nào?**

Ý tưởng của TF-IDF khá trực quan:

Một từ xuất hiện nhiều trong một comment sẽ quan trọng hơn

Nhưng nếu từ đó xuất hiện ở gần như mọi comment thì mức độ quan trọng sẽ giảm xuống

Ví dụ:

Các từ như:

- "the"
- "is"
- "you"

xuất hiện rất phổ biến nên không giúp phân biệt toxic hay non-toxic.

Ngược lại, những từ như:

- "idiot"
- "trash"
- "hate"

thường mang nhiều tín hiệu toxic hơn nên sẽ có trọng số cao hơn.


TF-IDF giúp mô hình tập trung vào những từ mang tính phân biệt thay vì chỉ đếm tần suất xuất hiện đơn thuần.

Trong project này, mình sử dụng TfidfVectorizer như sau:

```python
vec = TfidfVectorizer(
    ngram_range=(1,2),
    min_df=3,
    max_df=0.9,
    strip_accents='unicode',
    use_idf=True,
    smooth_idf=True,
    sublinear_tf=True
)
```

Một số tham số đáng chú ý:

- ngram_range=(1,2)
Sử dụng cả unigram và bigram để model có thể học các cụm từ như "stupid idiot" thay vì chỉ từng từ riêng lẻ.
- min_df=3
Loại bỏ những từ xuất hiện quá ít để giảm nhiễu.
- max_df=0.9
Loại bỏ những từ xuất hiện quá phổ biến trong dataset.
- sublinear_tf=True
Giảm ảnh hưởng của những từ lặp lại quá nhiều trong cùng một comment.

Chuyển dữ liệu thành vector

Sau khi fit TF-IDF trên tập train:

```python
trn_term_doc = vec.fit_transform(train['clean_comment'])
test_term_doc = vec.transform(test['clean_comment'])
```

Mỗi comment sẽ được biểu diễn thành một vector số với nhiều đặc trưng khác nhau.

Đây chính là dữ liệu đầu vào cho các mô hình machine learning ở bước tiếp theo.

# 8. Xây dựng mô hình
Phần này phân tích quá trình xây dựng mô hình cơ sở (baseline model) và mô hình chính (main model). Với mô hình cơ sở, mô hình Naive Bayes cùng với TF-IDF được lựa chọn, với mô hình chính, ta sử dụng mô hình Logistic Regression.

## 8.1. Naive Bayes

## 8.2. Logistic Regression

## 8.3. One vs Rest Classifier

## 8.4. Xây dựng mô hình

### 8.4.1. Mô hình cơ sở
Như đã đề cập ở các phần trước, ta sử dụng Naive Bayes cho mô hình cơ sở (baseline model). Với mô hình cơ sở, ta sẽ xử lí dữ liệu ở mức độ cơ bản như:

- Loại bỏ các dữ liệu trống
- Chuyển các kí tự từ chữ in hoa thành chữ thường
- Loại bỏ các kí tự không có ý nghĩa trong văn bản (các dấu câu ngẫu nhiên, địa chỉ web)

```python
# Get dataset information and check for null values
df_train.info()
df_train.isnull().sum()
```

```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 159571 entries, 0 to 159570
Data columns (total 9 columns):
 #   Column         Non-Null Count   Dtype 
---  ------         --------------   ----- 
 0   id             159571 non-null  object
 1   comment_text   159571 non-null  object
 2   toxic          159571 non-null  int64 
 3   severe_toxic   159571 non-null  int64 
 4   obscene        159571 non-null  int64 
 5   threat         159571 non-null  int64 
 6   insult         159571 non-null  int64 
 7   identity_hate  159571 non-null  int64 
 8   clean_text     159571 non-null  object
dtypes: int64(6), object(3)
```

Sau khi kiểm tra, ta thấy dataset không có dữ liệu trống. Do đó ta tiến hành chuyển đổi kí tự in hoa và loại bỏ kí tự không có ý nghĩa.

```python
# Clean out https, urls, extra spaces, tabs, newlines
def clean_text(text):

    # lowercase
    text = text.lower()

    # remove urls
    text = re.sub(r"http\S+|www\S+", "", text)

    # remove extra spaces/newlines/tabs
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Apply cleaning
df_train['clean_text'] = df_train['comment_text'].apply(clean_text)
```

Sau khi clean text, ta tiến hành phân chia dữ liệu training và validation. Bên cạnh đó, dữ liệu cũng được mã hóa bằng TF-IDF với các thông số:

- stop_words='english'
- ngram_range=(1,2)
- min_df=3
- max_df=0.9

```python
# Train, test split
from sklearn.model_selection import train_test_split

X = df_train['clean_text']
y = df_train[label_cols]

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Initialize TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

word_vectorizer  = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    ngram_range=(1,2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True
)

# Fit
X_train_word = word_vectorizer.fit_transform(X_train)
X_valid_word = word_vectorizer.transform(X_valid)
```

Bộ dữ liệu Jigsaw thuộc dữ liệu đa nhãn, mỗi bình luận đều có thể thuộc nhiều nhãn khác nhau (vừa toxic, vừa insult) do đó ta huấn luyện mô hình cho từng nhãn ở mỗi vòng lặp. Tại bước này, mô hình ComplementNB được sử dụng.

```python
import pandas as pd
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report, f1_score

results = []

models = {}

for label in label_cols:

    print(f"\nTraining model for: {label}")

    # Current label
    y_train_label = y_train[label]
    y_valid_label = y_valid[label]

    # Model
    model = ComplementNB(alpha=0.5)

    # Train
    model.fit(X_train_word, y_train_label)

    # Predict
    y_pred = model.predict(X_valid_word)

    # Classification report as dictionary
    report = classification_report(
        y_valid_label,
        y_pred,
        output_dict=True
    )

    # Extract metrics for positive class (class 1)
    precision = report['1']['precision']
    recall = report['1']['recall']
    f1 = report['1']['f1-score']
    support = report['1']['support']
    accuracy = report['accuracy']

    # Save results
    results.append({
        'label': label,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'support': support
    })

    # Save model
    models[label] = model

results_df = pd.DataFrame(results)
results_df = results_df.round(4)

results_df
```
Sau quá trình huấn luyện, ta thu được bảng kết quả:

|  | label | precision | recall | f1_score | accuracy | support |
| :---------: | :-------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| 0 | toxic        | 0.7015 | 0.6643 | 0.6824 | 0.9408 | 3056.0 |
| 1 | severe_toxic | 0.3384 | 0.4143 | 0.3725 | 0.9860 | 321.0 |
| 2 | obscene      | 0.6691 | 0.6507 | 0.6598 | 0.9639 | 1715.0 |
| 3 | threat       | 0.1100 | 0.1486 | 0.1264 | 0.9952 | 74.0 |
| 4 | insult       | 0.5985 | 0.5818 | 0.5900 | 0.9591 | 1614.0 |
| 5 | identity_hate| 0.2550 | 0.2177 | 0.2349 | 0.9869 | 294.0 |

**Nhận xét về bảng kết quả:**

Với nhãn toxic:

- Có số lượng samples lớn nhất, với precision = 0.70, recall = 0.66, f1 = 0.68
- Khi dự đoán bình luận là toxic, có 70% bình luận thật sự là toxic, 66% bình luận được dự đoán là toxic.
- Tuy nhiên, f1-score thấp cho thấy vẫn còn nhiều bình luận toxic chưa được dự đoán đúng.  


Với nhãn obsence:

- Đây là nhãn có số lượng samples cao thứ nhì, với precision = 0.66, recall = 0.65, f1 = 0.65
- Với nhãn obsence, các từ ngữ mang tính thô tục có xu hướng xuất hiện nhiều và riêng lẻ, dễ phân biệt, do đó mô hình đưa ra các chỉ số tương đối tốt
- Tuy nhiên, f1-score chưa cao cho thấy mô hình dự đoán chưa tốt cho nhãn obsence, một số từ ngữ tục tĩu được viết dưới định dạng đặc biệt như "f*ck", "f u c k", "fuuuuuck", "f#ck" gây khó khăn cho quá trình mã hóa dữ liệu. Do đó, ta cần xử lí tốt hơn ở phần này cho mô hình chính.

Với nhãn Insult:

- Số lượng samples cao thứ ba, precision = 0.59, recall = 0.58, f1 = 0.59
- Vì tính chất của insult là dựa vào ngữ cảnh (ví dụ: "you are clueless"), do đó rất khó để xác định một bình luận thật sự có ý xúc phạm hay không. Do đó ta cần xử lí tốt hơn để mô hình có thể hiểu được ngữ cảnh.


Với nhãn severe_toxic, identity_hate, threat:

- Các nhãn này đều có số lượng samples thấp, ảnh hưởng trực tiếp đến chỉ số precision, recall, f1-score.
- Với bản chất là bộ dữ liệu mất cân bằng giữa các class, đây là hiện tượng không thể tránh khỏi.

Điểm chung giữa các nhãn cho thấy accuracy đều rất cao, dao động từ 0.94 đến 0.98. Tuy nhiên ở bài toán này, ta không thể dựa vào accuracy:

- Bộ dữ liệu có tính mất cân bằng lớn, phần lớn bình luận đều là non-toxic, khi đó chỉ số accuracy cao (ví dụ: 0.97) chỉ phản ánh rằng mô hình dự đoán hầu hết các bình luận là non-toxic, nhưng không thể dự đoán được bình luận mang tình chất toxic nào.
- Để việc đánh giá mô hình hiệu quả hơn, ta nên sử dụng các metric khác như precision, recall, f1-score, ROC AUC

### 8.4.2. Mô hình chính
Như đã đề cập ở phần trước, để huấn luyện một mô hình có kết quả tốt hơn, ta tiến hành tiền xử lí dữ liệu và sử dụng Logistic Regression.

```python
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
COMMENT = 'comment_text'
train["comment_text"].fillna("unknown", inplace=True)
test["comment_text"].fillna("unknown", inplace=True)

tokenizer=TweetTokenizer()

nltk.download('stopwords')
nltk.download('wordnet')

eng_stopwords = set(stopwords.words("english"))
```

Để cải thiện dữ liệu so với baseline model, ta sử dụng thêm thư viện hỗ trợ như nltk (một thư viện để xử lí ngôn ngữ tự nhiên):

- Package stopwords dùng để loại bỏ các từ dừng như "a, an, the" vì các từ này thường mang không nhiều ý nghĩa cho phần phân loại.
- Pacakge WordNet là một cơ sở dữ liệu từ vựng khổng lồ của tiếng Anh được phát triển bởi Đại học Princeton. Ta sẽ sử dụng thuật toán Lemmatizer của WordNet để đưa các biến thể của một từ về dạng gốc của nó (ví dụ: running -> run). Qúa trình này đảm bảo mô hình xem các biến thể của một từ là đối tượng duy nhất mà không phải là các từ độc lập.
- TweetTokenizer là công cụ tokenizer đặc biệt dùng cho dữ liệu thuộc lĩnh vực social media. Bộ dữ liệu Jigsaw bao gồm các bình luận từ wikipedia, do đó việc sử dụng TweetTokenizer góp phần tối ưu giai đoạn xử lí ngôn ngữ.

Ở bước xử lí dữ liệu, ta tiến hành: 
- Tách các từ viết tắt thành từ đầy đủ trước khi đưa vào bộ tokenizer.
- Chuyển đổi các từ dạng viết hoa thành viết thường.
- Loại bỏ các dấu câu không liên quan, tên miền web. 

```python
APPO = {
"aren't" : "are not",
"can't" : "cannot",
"couldn't" : "could not",
"didn't" : "did not",
"doesn't" : "does not",
"don't" : "do not",
"hadn't" : "had not",
"hasn't" : "has not",
"haven't" : "have not",
"he'd" : "he would",
"he'll" : "he will",
"he's" : "he is",
"i'd" : "I would",
"i'd" : "I had",
"i'll" : "I will",
"i'm" : "I am",
"isn't" : "is not",
"it's" : "it is",
"it'll":"it will",
"i've" : "I have",
"let's" : "let us",
"mightn't" : "might not",
"mustn't" : "must not",
"shan't" : "shall not",
"she'd" : "she would",
"she'll" : "she will",
"she's" : "she is",
"shouldn't" : "should not",
"that's" : "that is",
"there's" : "there is",
"they'd" : "they would",
"they'll" : "they will",
"they're" : "they are",
"they've" : "they have",
"we'd" : "we would",
"we're" : "we are",
"weren't" : "were not",
"we've" : "we have",
"what'll" : "what will",
"what're" : "what are",
"what's" : "what is",
"what've" : "what have",
"where's" : "where is",
"who'd" : "who would",
"who'll" : "who will",
"who're" : "who are",
"who's" : "who is",
"who've" : "who have",
"won't" : "will not",
"wouldn't" : "would not",
"you'd" : "you would",
"you'll" : "you will",
"you're" : "you are",
"you've" : "you have",
"'re": " are",
"wasn't": "was not",
"we'll":" will",
"didn't": "did not",
"tryin'":"trying"
}
```

```python
import re
from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()

def clean(comment):
    """
    This function receives comments and returns clean word-list
    """
    #Convert to lower case , so that Hi and hi are the same
    comment=comment.lower()
    # remove \n
    comment=re.sub("\\n"," ",comment)
    # remove leaky elements like ip,user
    comment=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"," ",comment)
    #removing usernames
    comment=re.sub("\[\[.*\]"," ",comment)

    #Split the sentences into words
    words=tokenizer.tokenize(comment)
    # (')aphostophe  replacement (ie)   you're --> you are
    # ( basic dictionary lookup : master dictionary present in a hidden block of code)
    words=[APPO[word] if word in APPO else word for word in words]
    words=[lem.lemmatize(word, "v") for word in words]
    words = [w for w in words if not w in eng_stopwords]

    clean_sent=" ".join(words)

    return(clean_sent)

# Apply cleaning process to the train set
train['clean_comment'] = train['comment_text'].apply(clean)
test['clean_comment'] = test['comment_text'].apply(clean)
```

Sau quá trình xử lí dữ liệu, ta biến đổi dữ liệu thông qua véc tơ hóa. Hàm TfidfVectorizer với các tham số:
- ngram_range=(1,2): Trích xuất từ đơn và cụm hai từ trong văn bản. Việc trích xuất 2 từ giúp phân tích ý nghĩa rõ hơn (ví dụ: "you idiot" sẽ mang hàm ý chỉ trích hơn so với "you" và "idiot" riêng lẻ)
- min_df=3: Được hiểu như tần số xuất hiện của token, các token xuất hiện từ dưới ba lần sẽ được bỏ qua, điều này giúp giảm nhiễu như các từ sai chính tả, từ rác xuất hiện trong văn bản.
- max_df=0.9: Loại bỏ các token xuất hiện nhiều hơn 90%, các token này thường là các từ lặp như "the", "is", "you".
- strip_accents='unicode': Chuẩn hóa các từ về chuẩn unicode.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
n = train.shape[0]
vec = TfidfVectorizer(ngram_range=(1,2), 
                      min_df=3, 
                      max_df=0.9, 
                      strip_accents='unicode', 
                      use_idf=True,
                      smooth_idf=True, 
                      sublinear_tf=True 
                      )
trn_term_doc = vec.fit_transform(train['clean_comment'])
test_term_doc = vec.transform(test['clean_comment'])
```

Ta huấn luyện mô hình với input đã chuẩn hóa ở bước trước. Với mô hình chính thức, ta sử dụng mô hình Logistic Regression kết hợp với phương pháp nâng trọng số cho các đặc trưng từ TF-IDF.

Naive Bayes weighting là phương pháp nâng trọng số cho các đặc trưng từ TF-IDF. Bằng việc đặt thêm trọng số cho các đặc trưng từ TF-IDF, các từ có xu hướng độc hại sẽ được nhấn mạnh hơn, điều này giúp cải thiện hiệu suất phân loại cho mô hình. Phương pháp này được biểu diễn thông qua hàm sau:

```python
from sklearn.linear_model import LogisticRegression

def pr(y_i, y):
    p = x[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def get_mdl(y):
    y = y.values
    r = np.log(pr(1,y) / pr(0,y))
    m = LogisticRegression(C=4, dual=False)
    x_nb = x.multiply(r)
    return m.fit(x_nb, y), r
```

**Ý tưởng của thuật toán:**
```python
p = x[y==y_i].sum(0)
```
- Xét một lớp bất kì (ví dụ: lớp toxic y_i==1), tính tần số xuất hiện (.sum(0)) của các bình luận mang tính toxic (x[y==y_i]). Ta thu được tần số xuất hiện của một từ trong các bình luận toxic: "stupid" -	high, "idiot" -	high, "nice" -	low.
- Ta cộng 1 cho tử và mẫu để tránh trường hợp không xác định (kết quả trả về là một phân số có mẫu số là 0 và log(0))

```python
r = np.log(pr(1,y) / pr(0,y))
```
- Log-count ratio (hay Naive Bayes log-count ratio): tỷ lệ giữa tần suất xuất hiện của một từ trong một nhóm so với nhóm khác, sau đó được đưa qua hàm logarit. Thuật toán định lượng mức độ quan trọng của từ đối với từng phân lớp. Thuật toán chỉ ra "Từ đang xét mang ý toxic hay non-toxic nhiều hơn?".
- Ví dụ: từ "idiot" có tần số xuất hiện là 500, từ "nice" có tần số xuất hiện là 10, khi đó log(50) là số dương, nhận định rằng từ đó mang nghĩa toxic. Ngược lại, nếu tần số xuất hiện của "nice" là 1000, tần số của "idiot" là 5, từ "nice", kết quả trả về log(0.005) là số âm, nhận định rằng từ đó mang nghĩa non-toxic.
- Việc áp dụng véc tơ trọng số giúp ta có thể tạo ra các đặc trưng mạnh hơn, từ đó hỗ trợ quá trình phân loại của mô hình.

Ta tiến hành huấn luyện mô hình với từng nhãn cụ thể:

```python 
x = trn_term_doc
test_x = test_term_doc

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
preds = np.zeros((len(test), len(label_cols)))

for i, j in enumerate(label_cols):
    print('fit', j)
    m,r = get_mdl(train[j])
    preds[:,i] = m.predict_proba(test_x.multiply(r))[:,1]
```

Sau khi huấn luyện, ta lưu file train để tiến hành dự đoán:

```python
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = label_cols)], axis=1)
submission.to_csv('submission1.csv', index=False)

import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

# 1. Read prediction file and test file
sub = pd.read_csv('submission1.csv')
# labels = pd.read_csv('/content/drive/MyDrive/AIO_WarmUp3_Conquer/test.csv/test.csv')
labels = y_test # Use y_test which contains the true labels

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# 2. Filter out class with label == -1
mask = labels['toxic'] != -1
y_true_df = labels[mask]
y_pred_df = sub[mask]

# Lấy values (ma trận numpy) để tính toán
y_true = y_true_df[label_cols].values
y_pred_probs = y_pred_df[label_cols].values

# 3. Transform probability to binary value for F1-score
threshold = 0.5
y_pred_binary = (y_pred_probs > threshold).astype(int)

# 4. Calculate F1 Score and print result
print("=== F1 SCORE (Threshold = 0.5) ===")
f1_scores = []
for i, col in enumerate(label_cols):
    score = f1_score(y_true[:, i], y_pred_binary[:, i])
    f1_scores.append(score)
    print(f"{col:<15}: {score:.4f}")

macro_f1 = np.mean(f1_scores)
print("-" * 25)
print(f"Macro F1 Score : {macro_f1:.4f}\n")
```

Với hướng tiếp cận mới, ta có thể thấy mô hình cho ra kết quả tốt hơn so với mô hình cơ sở. Trừ nhãn toxic, F1 score cải thiện đáng kể cho 5 nhãn còn lại. Bên cạnh F1 score, ta sẽ dùng thêm ROC AUC để đánh giá hiệu suất mô hình:

**(phần này giải thích về ROC AUC)**

```python
print("=== ROC AUC Evaluation ===")
auc_scores = []
for i, col in enumerate(label_cols):
    score = roc_auc_score(y_true[:, i], y_pred_probs[:, i])
    auc_scores.append(score)
    print(f"{col:<15}: {score:.4f}")

mean_auc = np.mean(auc_scores)
print("-" * 25)
print(f"Mean ROC AUC   : {mean_auc:.4f}")
```

Kết quả ROC AUC:
| Label        | Value    |
| :------:     | :------: |
| toxic        | 0.9646   |
| severe_toxic | 0.9829   |
| obscene      | 0.9744   |
| threat       | 0.9854   |
| insult       | 0.9666   |
| identity_hate| 0.9753   |
| Mean ROC AUC | 0.9749   |


# Phần 11: Hạn chế của AI và Vấn đề Đạo đức

Mặc dù ToxiGuard AI cho thấy kết quả khả quan trong việc phân loại bình luận, chúng ta cần nhìn nhận thực tế rằng không có mô hình học máy nào là hoàn hảo. Khi áp dụng AI vào việc kiểm duyệt nội dung, đặc biệt là trong môi trường giáo dục, có những hạn chế và vấn đề đạo đức quan trọng cần được xem xét kỹ lưỡng:

- **Thiếu khả năng hiểu ngữ cảnh sâu sắc:** Mô hình hiện tại chủ yếu phân tích dựa trên từ vựng và cấu trúc câu (thông qua TF-IDF). Nó gặp khó khăn trong việc nắm bắt toàn bộ ngữ cảnh của cuộc hội thoại. Một từ ngữ có thể mang tính chất xúc phạm trong bối cảnh này, nhưng lại là từ lóng đùa giỡn giữa những người bạn trong bối cảnh khác.
- **Vấn đề thiên kiến (Bias):** Mô hình AI học từ dữ liệu chúng ta cung cấp. Nếu bộ dữ liệu huấn luyện (như Jigsaw dataset) chứa đựng những thiên kiến ngầm (ví dụ: thường xuyên gán nhãn "độc hại" cho các bình luận chứa một số từ khóa nhất định thuộc về một nhóm người yếu thế), mô hình sẽ có xu hướng "học" và lặp lại những thiên kiến đó, dẫn đến việc kiểm duyệt không công bằng.
- **Điểm mù với sự châm biếm (Sarcasm):** Nhận diện sự mỉa mai, châm biếm là một trong những thử thách khó nhất của Xử lý Ngôn ngữ Tự nhiên. Một bình luận với lời lẽ lịch sự nhưng mang hàm ý công kích sâu cay rất dễ "qua mặt" hệ thống.
- **Sự cần thiết của "Human-in-the-loop" (Sự can thiệp của con người):** Vì những hạn chế trên, một nguyên tắc quan trọng về đạo đức AI là không bao giờ để hệ thống tự động xóa bỏ hoàn toàn bình luận chỉ dựa trên phán đoán của thuật toán. ToxiGuard AI được thiết kế như một công cụ hỗ trợ (assisting tool) để gắn cờ (flag) cảnh báo mức độ rủi ro, giúp người quản trị (moderators/giáo viên) tiết kiệm thời gian lọc nội dung. Quyết định cuối cùng về việc ẩn, xóa bình luận hay nhắc nhở học viên vẫn cần sự can thiệp và đánh giá của con người để đảm bảo tính công bằng và tôn trọng quyền tự do ngôn luận.

# Phần 12: Tổng kết và Hướng phát triển tương lai

Dự án ToxiGuard AI đã chứng minh khả năng áp dụng các kỹ thuật Xử lý Ngôn ngữ Tự nhiên (NLP) và Học máy cơ bản để giải quyết bài toán phức tạp: Phân loại nhiều nhãn (Multi-label Classification) cho các bình luận độc hại. Từ việc làm sạch dữ liệu, trích xuất đặc trưng bằng TF-IDF, cho đến việc xây dựng mô hình Logistic Regression với chiến lược One-vs-Rest, chúng ta đã có một hệ thống nền tảng hoạt động hiệu quả.

**Giá trị mang lại:**

- **Về mặt học thuật:** Dự án là một quy trình hoàn chỉnh minh họa cách tiếp cận bài toán Text Classification từ đầu đến cuối, giúp củng cố kiến thức về tiền xử lý văn bản, lựa chọn mô hình và các chỉ số đánh giá.
- **Về mặt thực tiễn:** ToxiGuard AI mở ra tiềm năng ứng dụng thực tế trong việc làm sạch không gian học tập trực tuyến, tạo ra một môi trường an toàn, văn minh hơn để học viên tự do trao đổi kiến thức mà không lo ngại về bạo lực mạng.

**Hướng phát triển tương lai:**

Để ToxiGuard AI trở nên mạnh mẽ và hữu ích hơn trong thực tế, nhóm dự án đề xuất các hướng phát triển tiếp theo:

- **Hỗ trợ tiếng Việt:** Bước tiến quan trọng nhất là thu thập và xây dựng bộ dữ liệu bình luận tiếng Việt để huấn luyện mô hình, giúp ToxiGuard AI ứng dụng trực tiếp được vào cộng đồng học tập tại Việt Nam.
- **Ứng dụng các mô hình Deep Learning tiên tiến:** Chuyển đổi từ các mô hình truyền thống (Logistic Regression) sang các mô hình ngôn ngữ lớn mạnh mẽ hơn như BERT (hoặc PhoBERT cho tiếng Việt) để cải thiện đáng kể khả năng hiểu ngữ cảnh và nhận diện độ châm biếm.
- **Hệ thống phản hồi từ người dùng (Human Feedback):** Xây dựng cơ chế cho phép người quản trị sửa lỗi dự đoán của AI. Mô hình sẽ liên tục học hỏi từ những sửa đổi này (Active Learning) để ngày càng chính xác hơn.
- **Tích hợp trực tiếp vào Hệ thống Quản lý Học tập (LMS):** Phát triển API hoặc Plugin để cắm trực tiếp ToxiGuard AI vào các nền tảng học trực tuyến như Moodle, Canvas, hay các diễn đàn học tập, tự động quét và cảnh báo bình luận theo thời gian thực.

# Tài Liệu Tham Khảo

Jigsaw Toxic Comment Classification Challenge

Natural Language Processing Specialization – DeepLearning.AI

DeepLearning.AI. (n.d.). Natural Language Processing Specialization. Coursera. https://www.deeplearning.ai/courses/natural-language-processing-specialization/

Kaggle. (2018). Jigsaw Toxic Comment Classification Challenge. https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge

Jurafsky, D., & Martin, J. H. (2023). Speech and Language Processing (3rd ed. draft). Stanford University. https://web.stanford.edu/~jurafsky/slp3/

Schmidt, A., & Wiegand, M. (2017). A survey on hate speech detection using natural language processing. Proceedings of the Fifth International Workshop on Natural Language Processing for Social Media, 1–10.
