import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ================= 1. 数据预处理（预留数据增强位） =================
# 作用：训练集可加入轻微扰动提升泛化；测试集必须固定，只做归一化
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)


batch_size = ____
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)

# ================= 2. 带 Dropout/BN 的 MLP =================
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 补全：自行设定隐藏层维度，例如 [512, 256]
        self.layer1 = nn.Linear(784, ____)
        self.bn1    = nn.BatchNorm1d(____)  # 作用：对隐藏层输出做归一化，稳定分布加速收敛
        self.dropout1 = nn.Dropout(0.2)      # 作用：训练时随机丢弃20%神经元，防止过拟合
        
        self.layer2 = nn.Linear(____, ____)
        self.bn2    = nn.BatchNorm1d(____)
        self.dropout2 = nn.Dropout(0.2)
        
        self.layer3 = nn.Linear(____, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        
        x = self.layer1(x)
        x = self.bn1(x)      # 先归一化
        x = self.relu(x)     # 再激活
        x = self.dropout1(x) # 再丢弃（仅在 model.train() 时生效）
        
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.layer3(x)   # 输出层：无BN、无Dropout、无激活
        return x

model = MLP()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)  # 将模型参数搬至GPU（若可用）

# ================= 3. 优化器与调度器 =================
optimizer = optim.Adam(model.parameters(), lr=0.001)
                                                 # 作用：每过3个epoch将学习率乘0.5，后期精细打磨参数
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=_______)

criterion = nn.CrossEntropyLoss()

# ================= 4. 训练与验证双循环 =================
epochs = ____

for epoch in range(epochs):
    # ---- 训练阶段 ----
    model.train()  # 关键：开启Dropout与BN的学习模式
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)  # 数据搬至同设备
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # ---- 验证阶段（每个epoch结束时评估测试集）----
    model.eval()   # 关键：关闭Dropout，BN使用累积的均值方差
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    scheduler.step()  # 更新学习率
    
    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss/len(test_loader):.4f} | "
          f"Val Acc: {100*correct/total:.2f}%")

print("Training complete.")