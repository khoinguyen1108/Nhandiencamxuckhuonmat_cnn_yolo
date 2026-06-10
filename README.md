PHẦN 1: CÀI ĐẶT THƯ VIỆN
Các thư viện cần thiết để triển khai dự án bao gồm các framework học sâu, xử lý ảnh và xây dựng giao diện. Cụ thể:
- PyTorch: Framework chính để xây dựng và thiết lập quá trình học sâu cho mô hình CNN.
- Ultralytics: Cung cấp các công cụ và kiến trúc mạng học sâu cần thiết cho mạng YOLO.
- OpenCV: Xử lý ảnh số, cắt, phân giải mảng ma trận điểm ảnh và thực hiện phép biến đổi hình học (Face Alignment).
- Numpy và Pandas: Xử lý mảng dữ liệu tính toán và phân tích cấu trúc dữ liệu đa chiều.
- Gradio: Xây dựng giao diện web trực quan hóa kết quả đầu ra.
Cú pháp cài đặt thông qua trình quản lý gói pip:
pip install torch torchvision torchaudio
pip install ultralytics opencv-python numpy pandas gradio

PHẦN 2: CHUẨN BỊ DỮ LIỆU
Quá trình chuẩn bị dữ liệu đóng vai trò quyết định mức độ tổng quát hóa của mô hình học sâu.
- Nguồn dữ liệu: Dữ liệu được thu thập từ các kho lưu trữ dữ liệu học máy như Kaggle. Truy cập trang web Kaggle, nhập từ khóa tìm kiếm liên quan đến nhận diện cảm xúc khuôn mặt (ví dụ: FER2013), tải tệp dữ liệu nguyên gốc và giải nén vào thư mục dự án cục bộ.
- Cấu trúc thư mục: Dữ liệu hình ảnh cần được phân loại và định tuyến theo cấu trúc phân nhánh. Cụ thể là chia thành các tập huấn luyện (train), tập xác thực (validation) và tập kiểm thử (test). Trong mỗi tập, phân chia thêm các thư mục con đại diện cho từng lớp nhãn cảm xúc cụ thể.
- Chọn thiết bị xử lý: Quá trình thiết lập sẽ yêu cầu mã nguồn tự động kiểm tra phần cứng. Nếu bo mạch đồ họa (VGA) hỗ trợ kiến trúc tính toán song song CUDA, thiết bị sẽ tự động chuyển đổi sang sử dụng GPU để gia tốc thuật toán tính toán ma trận, ngược lại sẽ thiết lập sử dụng CPU.

PHẦN 3: MÔ HÌNH CNN VÀ THUẬT TOÁN FACE ALIGNMENT
1. Quá trình huấn luyện CNN:
Mô hình Convolutional Neural Network (CNN) tập trung vào việc trích xuất các dải đặc trưng không gian đa chiều từ hình ảnh. Kiến trúc liên kết nhiều lớp tích chập (Convolutional Layers) qua các bộ lọc (filters) nhằm thu thập cấu trúc, góc cạnh và biểu cảm. Lớp gộp (Pooling) giảm kích thước không gian ma trận, trước khi truyền vào các lớp kết nối đầy đủ (Fully Connected Layers). Hàm mất mát (Loss Function) Cross-Entropy được dùng để đo lường độ sai lệch giữa dự đoán và nhãn thực tế. Thuật toán lan truyền ngược kết hợp thuật toán tối ưu hóa (Optimizer) thực hiện cập nhật ma trận trọng số trong các vòng lặp (epochs).
2. Áp dụng thuật toán Face Alignment:
- Lý do áp dụng: Hình ảnh khuôn mặt đầu vào thường có xu hướng bị nghiêng, gây ra sai lệch khi CNN đối chiếu ma trận điểm ảnh. 
- Công thức tính toán: Xác định điểm tọa độ của hai mắt trên trục Descartes hai chiều: Mắt trái (x1, y1) và mắt phải (x2, y2).
Khoảng cách chênh lệch hai trục tọa độ: 
Delta_x = x2 - x1
Delta_y = y2 - y1
Góc nghiêng lệch (Theta) được tính bằng hàm lượng giác arctan:
Theta = arctan(Delta_y / Delta_x)
Tính toán tâm điểm xoay nằm giữa hai mắt: (center_x, center_y).
Thiết lập ma trận biến đổi Affine (M):
M = [cos(Theta)  -sin(Theta)  t_x]
    [sin(Theta)   cos(Theta)  t_y]
- Áp dụng vào thực tế: Một hàm biến đổi Warp Affine được gọi từ thư viện xử lý ảnh. Hình ảnh vùng mặt gốc sẽ được nhân với ma trận M. Kết quả đầu ra là vùng mặt được hiệu chỉnh đưa về góc xoay ngang tiêu chuẩn với độ lệch xấp xỉ không độ. Nhờ đó, đặc trưng về mắt và miệng luôn cố định theo vị trí ngang.

PHẦN 4: MÔ HÌNH YOLO 
1. Lý thuyết cơ bản:
Mô hình YOLO hoạt động dựa trên cơ chế phân chia hình ảnh thành một mạng lưới tọa độ S x S (grid cells). Tại mỗi phân vùng lưới, mô hình tính toán phương trình hồi quy trực tiếp ra tọa độ các hộp giới hạn (Bounding Boxes), đánh giá độ tin cậy (Confidence Score) cho sự xuất hiện của chủ thể và xác suất phân lớp. Toàn bộ hình ảnh chỉ cần đi qua mạng lưới nơ-ron một chiều lan truyền tiến duy nhất, tối ưu tuyệt đối cho yêu cầu thời gian thực.
2. Tác dụng của YOLO đối với hệ thống:
Mô hình YOLO trong ngữ cảnh dự án này vận hành dưới tư cách một hệ thống thị giác máy tính toàn diện, không rập khuôn trong việc tìm kiếm con người tổng thể. Nó chịu trách nhiệm lọc và hướng cơ chế chú ý (Attention) vào phân vùng không gian chứa thông tin trọng tâm. Thay vì dùng các phương pháp xác định vị trí cũ tốn kém tài nguyên tính toán, mô hình lượng hóa độ sâu của bức ảnh để bóc tách riêng phần khuôn mặt bất chấp bối cảnh phức tạp phía sau, mang lại định vị mục tiêu chính xác ở tần số quét tính bằng mili-giây.
3. Huấn luyện và áp dụng thuật toán Face Alignment với YOLO:
- Quá trình huấn luyện sử dụng tệp trọng số có sẵn để tinh chỉnh (Fine-tuning) qua tập dữ liệu nhãn đã được đánh dấu vùng bounding box. Thuật toán thay đổi các tham số trong kiến trúc mạng dựa trên hàm mất mát tổng hợp từ mất mát khoanh vùng và phân lớp.
- Áp dụng Face Alignment: YOLO cung cấp tọa độ x_min, y_min, x_max, y_max vùng khuôn mặt. Dữ liệu này được trích xuất thành vùng phân tích. Trên diện tích vùng ảnh này, thuật toán dò tìm điểm mốc (Landmarks) phát hiện hai con mắt và trực tiếp áp dụng chuỗi ma trận Affine được lý giải ở Phần 3 để chuẩn hóa góc nhìn trước khi tiến hành những bước xử lý kế tiếp.

PHẦN 5: KẾT HỢP CNN VÀ YOLO
Sự kết hợp này tách một bài toán nhận diện lớn ra thành hai khâu hoạt động riêng biệt nhưng bổ trợ chặt chẽ cho nhau.
1. Vai trò cụ thể:
- YOLO đóng vai trò là "Module định vị và khử nhiễu" (Localization & Denoising Module). Giải quyết hoàn toàn vấn đề nhiễu môi trường và thay đổi bối cảnh bằng cách giới hạn hoàn toàn thuật toán vào một hình vuông bao quanh trọng tâm.
- CNN đóng vai trò là "Module phân lớp sâu" (Deep Classifier Module). Hoàn toàn không bận tâm đến vị trí khuôn mặt nằm ở đâu, chỉ tập trung giải mã các đặc điểm kết cấu vi mô (độ nhăn, độ giãn cơ mặt) để quy về thông số nhận diện.
2. Quá trình tính toán chuỗi dữ liệu:
Khi một bức ảnh hoặc một khung hình video đầu vào đi vào hệ thống:
Bước 1: Hình ảnh kích thước lớn qua YOLO xử lý, xuất ra tọa độ Bounding Box chứa khuôn mặt.
Bước 2: Phép toán nội suy ma trận trích xuất cắt rời đoạn lưới ma trận ảnh từ vùng Bounding Box đó. 
Bước 3: Vùng mặt chưa căn chỉnh chạy qua hàm tính góc lệnh Theta và ma trận xoay M, ép về tỷ lệ thẳng trục (Face Alignment).
Bước 4: Kết quả cuối cùng là một ảnh kích thước nhỏ chuẩn hóa được nạp thẳng vào đầu vào của lớp Convolutional đầu tiên của mạng CNN.
Bước 5: CNN tính toán qua toàn bộ các lớp tích chập và gộp, xuất ra vector Logits, đi qua hàm Softmax trả về bảng danh sách tỷ lệ phần trăm phân bố trên từng nhãn cảm xúc cụ thể.

PHẦN 7: XÂY DỰNG GIAO DIỆN TỪ GRADIO
Gradio là một thư viện hỗ trợ biến mã nguồn máy học thành các bộ khung ứng dụng Web trực quan mà không yêu cầu kiến thức phức tạp về thiết kế hệ thống Front-End.
Cách xây dựng:
- Khởi tạo khối hàm trung tâm (Core Function). Hàm này có thiết kế nhận dữ liệu đầu vào là mảng byte hình ảnh, kích hoạt tuần tự các mô hình đã lưu từ việc gọi YOLO định vị, căn chỉnh góc mặt, cho đến kết luận từ CNN. 
- Tại phần thân chương trình, đối tượng Blocks của Gradio được sử dụng để xây dựng bố cục đa thẻ (Tabs). 
- Thẻ thứ nhất (Dự Đoán Cảm Xúc): Cấu hình module tải file hình ảnh, truyền tín hiệu nhận diện và hiển thị khung ảnh kết quả phân lớp cảm xúc.
- Thẻ thứ hai (Biểu Đồ Huấn Luyện): Cấu hình module tĩnh đọc các tệp hình ảnh lưu trữ tiến trình học (Loss, mAP, Accuracy) của YOLO và CNN, cho phép người theo dõi đối chiếu trực tiếp thông số hiệu suất ngay trên giao diện mà không cần truy xuất dữ liệu gốc.
- Hàm launch() được cấu hình ở dòng cuối để thiết lập máy chủ cục bộ theo thời gian thực.

PHẦN 8: HƯỚNG DẪN HUẤN LUYỆN VÀ SỬ DỤNG
1. Hướng dẫn huấn luyện:
- Mở cửa sổ giao diện dòng lệnh (Terminal/Command Prompt).
- Chuyển thư mục đường dẫn về vị trí gốc của dự án.
- Kích hoạt trình thông dịch ngôn ngữ có chứa các bộ thư viện phân tích.
- Thực thi mã kịch bản huấn luyện của hệ thống YOLO bằng việc chỉ định tệp thông số YAML cấu hình dữ liệu và file trọng số sơ khởi.
- Kế tiếp, chạy mã kịch bản huấn luyện CNN, mã nguồn tự động lặp qua thư mục tập dữ liệu khuôn mặt, chạy tiến trình lan truyền ngược tối ưu hàm mất mát. File định dạng dạng .pt hoặc .pth chứa trọng số nhẹ nhất, cho độ suy luận tốt nhất ở tập Validation sẽ được xuất ra vào thư mục lưu trữ trọng số.
2. Hướng dẫn sử dụng:
- Tại thư mục gốc dự án, dùng lệnh ngôn ngữ để thực thi file khởi chạy của Gradio.
- Máy chủ khởi tạo và in ra một chuỗi địa chỉ giao thức cục bộ (thường là http://127.0.0.1:7860).
- Sao chép đường dẫn này và dán vào thanh tìm kiếm trên trình duyệt web.
- Giao diện mở ra, tải hình ảnh có sự xuất hiện của khuôn mặt hoặc cấp quyền kích hoạt webcam, sau đó thực thi phân tích để hiển thị kết quả cảm xúc trên màn hình theo dòng thời gian thực.

PHẦN 9: TRỰC QUAN HÓA BIỂU ĐỒ HUẤN LUYỆN (LOSS & EVALUATION)
Để đánh giá trực quan hiệu suất của cả hệ thống trong suốt quá trình huấn luyện, các biểu đồ về hàm mất mát (Loss) và các chỉ số đánh giá (Evaluation) được xuất bản hoàn toàn tự động. Hệ thống hỗ trợ xem các biểu đồ này trực tiếp tại Thẻ "Biểu Đồ Huấn Luyện" trên giao diện Gradio.
1. Biểu đồ đánh giá mô hình YOLO:
- Mọi kết quả từ tiến trình huấn luyện được cấu trúc và lưu trữ tự động tại thư mục "runs/detect/train/" (hoặc thư mục tương đương của Ultralytics).
- Tệp "results.png" hiển thị trực quan toàn cảnh biểu đồ dao động của Box Loss, Classification Loss, DFL Loss, cũng như sự gia tăng của các chỉ số mAP50 và mAP50-95 qua từng vòng lặp (Epoch).
- Ma trận nhầm lẫn (confusion_matrix.png) và đường cong F1-Confidence (F1_curve.png) giúp đối chiếu chi tiết khả năng nhận diện trên tập kiểm định (Validation), xác định rõ mô hình đang gặp khó khăn ở nhãn lớp nào.
2. Biểu đồ đánh giá mô hình CNN:
- Tại kịch bản mã nguồn huấn luyện CNN, biến số Training Loss, Validation Loss và Accuracy (độ chính xác) được mảng hóa ở cuối mỗi chu kỳ.
- Thư viện vẽ biểu đồ (Matplotlib) được tích hợp để phác họa các điểm dữ liệu này thành đường xu hướng. Mã nguồn đã được cấu hình tự động lưu kết quả đồ thị thành tệp "cnn_loss_accuracy_plot.png".
- Việc phân tích điểm giao cắt và xu hướng tách rời giữa đường Training Loss và Validation Loss mang ý nghĩa học thuật cao, phục vụ việc phát hiện hiện tượng học quá khớp (Overfitting) hoặc học chưa đủ (Underfitting).
