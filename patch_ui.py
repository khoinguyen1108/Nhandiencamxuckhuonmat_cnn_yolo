import json

nb_path = 'NhanDienCamXucKhuonMat_CNN_Yolo-1.ipynb'

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i in range(len(source)):
            # 1. Revert groupThreshold
            if 'faces, weights = cv2.groupRectangles' in source[i] and 'groupThreshold=1' in source[i]:
                source[i] = source[i].replace('groupThreshold=1', 'groupThreshold=0')
            
            # 2. Add gr.Info to flag_image
            if 'print(f"[INFO] Đã cắm cờ' in source[i]:
                # Insert gr.Info right after print
                indent = source[i][:len(source[i]) - len(source[i].lstrip())]
                if "gr.Info" not in source[i+1]:
                    source.insert(i+1, indent + "gr.Info(\"Đã lưu ảnh lỗi thành công vào thư mục flagged_vn!\")\n")

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Patch applied successfully.")
