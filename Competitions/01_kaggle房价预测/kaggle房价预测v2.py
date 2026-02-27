import torch
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
# 环境配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv('train_dataset.csv')
test_df = pd.read_csv('test_dataset.csv')

# 提取训练集特征和标签
# 训练集特征：去掉最后一列 PRICE
x_train_raw = train_df.drop(columns=['PRICE']).values
y_train_raw = train_df['PRICE'].values

# 提取测试集特征
# 测试集特征：去掉 ID
x_test_raw = test_df.drop(columns=['ID']).values

print(f"训练特征维度: {x_train_raw.shape}")  # 应该是 (xxx, 13)
print(f"测试特征维度: {x_test_raw.shape}")  # 应该是 (xxx, 13)

# 标准化
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_raw)
x_test_scaled = scaler.transform(x_test_raw)

# 转换为张量
train_xt = torch.from_numpy(x_train_scaled.astype(np.float32))
train_yt = torch.from_numpy(y_train_raw.astype(np.float32))
test_xt = torch.from_numpy(x_test_scaled.astype(np.float32)).to(device)

# 数据加载
train_ds = Data.TensorDataset(train_xt, train_yt)
train_loader = Data.DataLoader(dataset=train_ds, batch_size=32, shuffle=True)


# 模型
class HouseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.fc(x)


model = HouseModel().to(device)


# 训练
def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 换成 Adam 通常收敛更快
    loss_fun = nn.MSELoss()

    model.train()
    for epoch in range(100):
        for x, y in train_loader:
            out = model(x.to(device))
            loss = loss_fun(out.squeeze(), y.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    train()

    # 开始预测
    model.eval()
    with torch.no_grad():
        predictions = model(test_xt).cpu().numpy().flatten()

    # 生成提交文件 (Kaggle 格式)
    # --- 严格按照 SampleSubmission.csv 格式生成结果 ---

    # 1. 读取测试集，获取 ID 列（确保它是 id_1, id_2 这种格式）
    # 如果 test_df 里原本就有 ID 列，直接引用；如果没有，需要根据 sample 生成
    sample_df = pd.read_csv('SampleSubmission.csv')

    # 2. 构建 DataFrame
    # 注意：列名必须严格对应 'ID' 和 'value'
    submission = pd.DataFrame({
        'ID': sample_df['ID'],  # 直接使用示例文件里的 ID，保证顺序和格式 100% 正确
        'value': predictions  # 填入你的模型预测结果
    })

    # 3. 保存文件
    submission.to_csv('submission.csv', index=False)

    print("✅ 成功！已生成符合 Kaggle 要求的 submission.csv")
    print("预览前 3 行：")
    print(submission.head(3))

    # 简单可视化
    plt.plot(predictions[:50], label='预测房价')
    plt.legend()
    plt.show()