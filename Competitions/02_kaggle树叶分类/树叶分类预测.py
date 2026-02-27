import torch
import os
import pandas as pd
from PIL import Image
from torchvision import transforms

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载整个模型
model = torch.load(r'C:\Users\19902\Desktop\PythonProject\save\model.pth', weights_only=False)
model.to(device)
model.eval() # 进入预测模式

# 必须和训练代码中的 test_data_trans 保持一致
data_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 假设你的图片放在这个目录下
img_root = r'C:\Users\19902\Desktop\data'
test_csv_path = r'C:\Users\19902\Desktop\data\test.csv'

test_df = pd.read_csv(test_csv_path)
predictions = []

# 这里假设你已经有了 class_to_idx 的映射关系
# 如果没有，需要手动定义一个列表，顺序必须和训练时 ImageFolder 扫描的顺序一致
# 建议查看你之前保存的 id_code.csv
id_code = pd.read_csv(r'C:\Users\19902\Desktop\data\id_code.csv')
idx_to_class = dict(zip(id_code['id'], id_code['label']))

print("正在预测...")
with torch.no_grad():
    for img_path in test_df['image']:
        full_path = os.path.join(img_root, img_path)
        img = Image.open(full_path).convert('RGB')
        img_tensor = data_transform(img).unsqueeze(0).to(device)

        output = model(img_tensor)
        pred_idx = output.argmax(dim=1).item()
        predictions.append(idx_to_class[pred_idx])

# 保存最终预测结果
test_df['label'] = predictions
test_df.to_csv(r'C:\Users\19902\Desktop\PythonProject\save\submission.csv', index=False)
print("Done! 结果已保存到 save/submission.csv")