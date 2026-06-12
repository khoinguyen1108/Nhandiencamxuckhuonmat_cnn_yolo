import json
import re

nb_path = 'NhanDienCamXucKhuonMat_CNN_Yolo-1.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Markdown replacements
# We'll identify cells by checking their content strings.

md_cnn_text = """### Phần 4: Kiến trúc mô hình CNN (Trích xuất đặc trưng cảm xúc)

#### 1. Kiến trúc Depthwise Separable Convolution
Hệ thống sử dụng cơ chế Depthwise Separable Convolution nhằm tối ưu hóa số lượng phép tính mà vẫn duy trì khả năng trích xuất đặc trưng. 
Giả sử đầu vào có $C_{in}$ kênh, đầu ra có $C_{out}$ kênh và kích thước bộ lọc là $K \\times K$ (với $K=3$).
- Số lượng phép nhân (Multiply-Accumulate Operations - MACs) của lớp Convolution tiêu chuẩn tỷ lệ với: $K^2 \\times C_{in} \\times C_{out}$.
- Số lượng phép nhân của Depthwise Separable Convolution tỷ lệ với: $K^2 \\times C_{in} + C_{in} \\times C_{out}$.

Tỷ lệ nén (Compression Ratio) đạt được là:
$$ \\frac{\\text{Chi phí Depthwise Separable}}{\\text{Chi phí Conv Tiêu chuẩn}} = \\frac{1}{C_{out}} + \\frac{1}{K^2} \\approx \\frac{1}{9} $$

Việc ứng dụng cấu trúc này giúp giảm gần 9 lần khối lượng tính toán, cho phép mô hình hoạt động ổn định trong các ứng dụng thời gian thực (Real-time).

#### 2. Nén không gian và Cơ chế chống hiện tượng Overfitting
Ma trận ảnh xám đầu vào dưới dạng Tensor 4 chiều $\\mathbf{X} \\in \\mathbb{R}^{B \\times 1 \\times 112 \\times 112}$ (với $B$ là Batch Size) lần lượt đi qua 4 khối Block. Thông qua lớp `MaxPool2d(2, 2)`, độ phân giải không gian giảm dần và độ sâu kênh tăng dần: $112 \\times 112 \\rightarrow 56 \\times 56 \\rightarrow 28 \\times 28 \\rightarrow 14 \\times 14 \\rightarrow 7 \\times 7$.

Đến lớp cuối cùng, Tensor đạt kích thước:
$$ \\mathbf{T}_{final} \\in \\mathbb{R}^{B \\times 256 \\times 7 \\times 7} $$

Thay vì sử dụng phép biến đổi `Flatten` tạo ra một vector khổng lồ ($12,544$ chiều) gây quá tải lượng tham số, mạng được tích hợp **Global Average Pooling (GAP)** thông qua lớp `AdaptiveAvgPool2d((1, 1))`. Thuật toán này tính trung bình cộng toàn bộ ma trận $7 \\times 7$ cho từng kênh:
$$ \\mathbf{v}_c = \\frac{1}{H \\times W} \\sum_{i=1}^{H} \\sum_{j=1}^{W} \\mathbf{T}_{c, i, j} $$

Tensor lập tức được thu gọn thành $\\mathbb{R}^{B \\times 256 \\times 1 \\times 1}$. Trọng số ở lớp Linear được giữ ở mức cực thấp ($256 \\times 8 = 2048$ tham số). Kết hợp cùng `Dropout(0.5)`, mạng CNN hạn chế tối đa nguy cơ học vẹt (Overfitting) dựa trên dữ liệu huấn luyện.
"""

md_yolo_text = """### Phần 5: Mô hình YOLO (Nhận diện vị trí khuôn mặt)

#### 1. Trích xuất đặc trưng Đa tỷ lệ (PANet - Path Aggregation Network)
Để xử lý vấn đề chênh lệch kích thước đối tượng khi khuôn mặt ở khoảng cách khác nhau so với camera, kiến trúc sử dụng cơ chế PANet. Mạng tiến hành trích xuất đặc trưng ở 3 cấp độ phân giải khác nhau (Feature Maps: P3, P4, P5), sau đó thực hiện quá trình truyền thông tin hai chiều (Top-down và Bottom-up). Kỹ thuật này giúp liên kết dữ liệu ngữ cảnh diện rộng với chi tiết vi mô.

#### 2. Cơ chế Dự đoán Anchor-Free & Task-Aligned Assigner
Khác với các thế hệ trước dựa vào các hộp định sẵn (Anchors), kiến trúc hiện tại ứng dụng phương pháp dự đoán không cần Anchor (Anchor-Free). Tâm của mỗi ô lưới (Grid Center) đóng vai trò là một điểm dự đoán độc lập. Tại điểm $(x, y)$, mô hình xuất ra một Vector Hồi quy bao gồm 4 giá trị khoảng cách tới 4 cạnh của Bounding Box tương ứng $(l, t, r, b)$ (Trái, Trên, Phải, Dưới):
$$ \\text{Box} = [x - l, \\quad y - t, \\quad x + r, \\quad y + b] $$

Quá trình huấn luyện sử dụng hàm **CIoU Loss** (Complete Intersection over Union) để đo lường độ sai lệch giữa Hộp dự đoán ($B_{pred}$) và Hộp chuẩn ($B_{gt}$) trên 3 phương diện: Độ chồng lấn, Khoảng cách tâm và Tỷ lệ khung hình:
$$ \\text{Loss}_{CIoU} = 1 - \\text{IoU} + \\frac{\\rho^2(b_{pred}, b_{gt})}{c^2} + \\alpha v $$

#### 3. Thuật toán lọc Non-Maximum Suppression (NMS)
Khi mô hình phân tích qua các ô lưới, quá trình nhận diện thường xuất hiện nhiều Bounding Box bao quanh cùng một đối tượng (Ghost Detections). Thuật toán NMS xử lý hiện tượng này thông qua các bước toán học:
1.  Loại bỏ các Bounding Box có điểm tin cậy (Confidence Score) thấp hơn ngưỡng quy định (ví dụ: $conf < 0.5$).
2.  Lựa chọn Bounding Box có điểm cao nhất làm hộp tham chiếu (Hộp chuẩn).
3.  Tính chỉ số IoU (Intersection over Union) giữa Hộp chuẩn và các hộp lân cận. Nếu chỉ số IoU vượt quá ngưỡng (ví dụ: $iou > 0.45$), hệ thống sẽ triệt tiêu các hộp đó, đảm bảo tính duy nhất của đối tượng nhận diện.
$$ \\text{IoU}(A, B) = \\frac{\\text{Area}(A \\cap B)}{\\text{Area}(A \\cup B)} $$
"""

md_opt_text = """### Phần 6: Huấn luyện Mô hình & Các Kỹ thuật Tối ưu (Training & Optimizations)
Phần này mô tả cấu trúc tối ưu hóa để đảm bảo hệ thống duy trì hiệu năng cao (SOTA) trong môi trường phát hiện thời gian thực (Real-time).

#### 1. Tối ưu Kiến trúc Mạng (CNN Tuning)
- **Tăng dung lượng mô hình (Capacity):** Mạng tích hợp 4 khối `DepthwiseSeparableConv` với sự gia tăng kích thước kênh theo cấp số nhân (16 $\\rightarrow$ 32 $\\rightarrow$ 64 $\\rightarrow$ 128 $\\rightarrow$ 256). Thiết kế này cung cấp đủ chiều sâu không gian tham số để trích xuất các đặc trưng phức tạp như nếp nhăn và góc miệng, trong khi vẫn đảm bảo cấu trúc siêu nhẹ.
- **Tăng cường Dữ liệu (Data Augmentation):** Áp dụng ngẫu nhiên các phép biến đổi không gian hình học (như xoay ngang, điều chỉnh độ phơi sáng) trong giai đoạn tiền xử lý, cải thiện tính tổng quát hóa mô hình đối với các yếu tố ngoại cảnh.

#### 2. Kỹ thuật Kết hợp Không gian (Pipeline Integration)
Quy trình trích xuất ROI (Region of Interest) bằng thuật toán phát hiện và điều phối nạp vào mạng CNN đòi hỏi quá trình hiệu chuẩn nghiêm ngặt:
- **Tinh chỉnh tỷ lệ khung hình (Padding):** Các khung Bounding Box mặc định thường bị hạn chế ở phần trung tâm ngũ quan. Cấu trúc nội tại tự động nới rộng khung hình thêm khoảng $5\\%$ dọc theo các trục. Phương pháp này đảm bảo thu giữ toàn vẹn bối cảnh vi sinh thái biểu cảm (chân mày, cơ mặt) mà không tiếp nhận các nhiễu phông nền.
- **Căn chỉnh Hình học (Face Alignment):** Để khắc phục độ nhạy sai lệch do góc xoay (Rotation Variance) đối với mạng CNN. Ảnh được căn chỉnh toán học như sau:
  * Trích xuất tạo độ tâm mắt trái $E_L(x_1, y_1)$ và mắt phải $E_R(x_2, y_2)$.
  * Xây dựng Vector khoảng cách $\\Delta x = x_2 - x_1$ và $\\Delta y = y_2 - y_1$.
  * Áp dụng hàm lượng giác ngược `atan2` nhằm xác định góc xoay $\\theta$ của quỹ đạo khuôn mặt so với trục ngang: 
  $$ \\theta = \\arctan2(\\Delta y, \\Delta x) \\times \\frac{180}{\\pi} $$
  * Thực thi Phép biến đổi Affine (Affine Transform) với tâm quay $x_c = \\frac{x_1 + x_2}{2}, y_c = \\frac{y_1 + y_2}{2}$ thông qua ma trận quay $M$:
  $$ M = \\begin{bmatrix} \\cos\\theta & \\sin\\theta \\\\ -\\sin\\theta & \\cos\\theta \\end{bmatrix} $$

#### 3. Kỹ thuật Làm mịn Bộ đệm (Majority Voting Smoothing)
Dưới tần số quét của Camera 30 FPS, độ không chắc chắn (Uncertainty) nội tại của mạng CNN gây ra hiện tượng dao động xác suất nhỏ trên các khung hình tĩnh.
- **Giải pháp:** Tích hợp bộ đệm Queue lưu trữ $N=5$ kết quả phân lớp chuỗi thời gian gần nhất.
- **Cơ chế hoạt động:** Thay vì xuất kết quả đơn lẻ, hàm bầu chọn số đông (Majority Vote) được kích hoạt nhằm triệt tiêu các mẫu nhiễu độc lập lẻ tẻ (Outliers), tạo ra sự mượt mà khi quan sát nhãn cảm xúc theo thời gian.
"""

md_infer_text = """### Phần 7: Luồng suy luận (Inference Pipeline)
Giai đoạn Inference là cơ chế vận hành chính, quy định lộ trình dữ liệu không xác định chạy qua các mô hình trí tuệ nhân tạo.

**1. Chế độ Camera Thời gian thực (Real-time Stream):**
- Sử dụng mô-đun VideoCapture trích xuất từng khung hình (Frame) dưới dạng ma trận mảng nội suy.
- Hình ảnh được định tuyến qua thuật toán dò tìm đối tượng. Vùng nhận diện (Khuôn mặt) sau khi cắt sẽ trải qua Pipeline tiền xử lý và cung cấp trực tiếp vào mạng CNN, trả về vector xác suất phân loại cảm xúc.
- **Thuật toán làm mịn chuỗi (Smoothing Queue):** Cảm xúc có tần suất xuất hiện cực đại (Mode) trong bộ đệm $N=5$ khung hình liên tiếp sẽ được gán nhãn cuối cùng, đảm bảo sự ổn định tín hiệu ngõ ra.

**2. Chế độ Suy luận Ảnh tĩnh (Image Upload):**
- Khung ảnh tĩnh dạng ma trận RGB sẽ qua bộ lọc nhận diện đa đối tượng nhằm khoanh vùng tất cả các biến thể có khả năng là khuôn mặt tồn tại trên ảnh.
- Quá trình chuyển đổi cấu trúc (Resize, Normalize) được áp đặt đồng nhất theo phân phối trọng số của tập dữ liệu huấn luyện, trước khi qua mạng CNN.
- Hệ thống áp dụng giao diện Bounding Box để chồng lấp ma trận hiển thị (Overlay) và trả về kết quả hình thái học cuối.
"""

def replace_markdown(cell, prefix, new_text):
    if cell['cell_type'] == 'markdown':
        content = "".join(cell['source'])
        if content.startswith(prefix):
            # re-split source into list of strings with newlines
            lines = [line + '\n' for line in new_text.split('\n')]
            lines[-1] = lines[-1].rstrip('\n') # remove last newline
            cell['source'] = lines
            return True
    return False

# Code modifications
def modify_code_lines(cell):
    if cell['cell_type'] == 'code':
        source = cell['source']
        modified = False
        for i in range(len(source)):
            # 1. Update YOLO inference params
            if 'results = self.detector.track(frame' in source[i]:
                source[i] = source[i].replace('persist=True, verbose=False', 'persist=True, conf=0.5, iou=0.45, verbose=False')
                modified = True
            elif 'results = self.detector(image, verbose=False)' in source[i]:
                source[i] = source[i].replace('verbose=False', 'conf=0.5, iou=0.45, verbose=False')
                modified = True
            
            # 2. Update Haar Cascade groupRectangles
            elif 'faces, weights = cv2.groupRectangles' in source[i] and 'groupThreshold=0' in source[i]:
                source[i] = source[i].replace('groupThreshold=0', 'groupThreshold=1')
                modified = True
            
            # 3. Update padding 0.15 -> 0.05
            elif 'pad_w, pad_h = int(w * 0.15), int(h * 0.15)' in source[i]:
                source[i] = source[i].replace('0.15', '0.05')
                modified = True
                
        return modified
    return False

for cell in nb['cells']:
    # Replace markdown cells
    if replace_markdown(cell, '### Phần 5: Mô hình YOLO', md_yolo_text): pass
    elif replace_markdown(cell, '### Phần 6: Huấn luyện Mô hình', md_opt_text): pass
    elif replace_markdown(cell, '### Phần 6: Luồng suy luận', md_infer_text): pass
    # It seems 'Phần 4' was merged with CNN code in the view, let's replace by looking for "Lợi ích Toán học" or similar
    elif replace_markdown(cell, '**Lợi ích Toán học:**', md_cnn_text): pass
    elif replace_markdown(cell, '#### 2. Nén Không gian và Chống học vẹt', md_cnn_text): pass
    
    # Let's write a generic scanner for CNN markdown
    elif cell['cell_type'] == 'markdown' and 'Depthwise Separable' in "".join(cell['source']) and 'Lợi ích' in "".join(cell['source']):
        replace_markdown(cell, "".join(cell['source']), md_cnn_text)
        
    modify_code_lines(cell)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Update complete")
