import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ================= 1. 数据预处理与加载 =================
# 作用：将MNIST图像转为Tensor并归一化到[0,1]，标签自动转为整数
transform = transforms.Compose([
    transforms.ToTensor(),                                       # 将PIL图转为Tensor (1,28,28)，并除以255
    transforms.Normalize((0.1307,), (0.3081,))                   # MNIST标准均值方差，稳定训练
])

# 补全：下载MNIST训练集与测试集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


batch_size = 64              #建议为2的幂次，
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

# ================= 2. 手写 MLP 模型 =================
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 补全：定义层数与维度。MNIST图像是28*28=784维向量输入
        self.fc1 = nn.Linear(784, 256)   
        self.fc2 = nn.Linear(256, 32)    #建议为2的幂次
        self.fc3 = nn.Linear(32, 10)     # 10分类（数字0-9）
        self.relu = nn.ReLU()               # 非线性激活：筛选有效特征
        
    def forward(self, x):
        # 作用：将 (B,1,28,28) 的图像展平为 (B,784) 的向量，B为batch_size
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))  # 第一层线性变换+激活
        x = self.relu(self.fc2(x))  # 第二层
        x = self.fc3(x)             # 输出层：不加激活，后续CrossEntropyLoss内置Softmax
        return x

model = MLP()

# ================= 3. 损失函数与优化器 =================
criterion = nn.CrossEntropyLoss()  # 作用：计算预测分布与真实标签的交叉熵
optimizer = optim.Adam(model.parameters(), lr= 0.001 )  # Adam自适应学习率优化

# ================= 4. 训练循环 =================
epochs = 10  

for epoch in range(epochs):
    model.train()  # 开启训练模式（影响Dropout/BN等行为，此处预留习惯）
    total_loss = 0
    for images, labels in train_loader:
        # 前向传播：数据流经模型得到预测 logits
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播三步固定模板：清零旧梯度 -> 计算新梯度 -> 更新参数
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}")

# ================= 5. 测试评估 =================
model.eval()  # 评估模式
correct = 0
total = 0
with torch.no_grad():  # 关闭梯度计算，节省显存加速推理
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # 取概率最大索引作为预测类别
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")