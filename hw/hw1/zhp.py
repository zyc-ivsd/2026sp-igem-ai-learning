import torch
import torch.nn as nn


# 模拟一张 5x5 的单通道图片（batch=1, channel=1, h=5, w=5）
input_tensor = torch.tensor([
    [1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# 连续两个 unsqueeze：第0维加batch → [1,5,5]；再第0维加channel → [1,1,5,5]

print(f"输入形状: {input_tensor.shape}")      #shape 希望同学自己先在脑子里算一遍，在跟打印结果做对比
print(f"输入:\n{input_tensor.squeeze()}")                # squeeze去掉大小为1的维度，方便打印

# ========== 创建一个卷积层 ==========
# in_channels=1:  输入是1通道（灰度图）
# out_channels=1: 输出1个通道（1个卷积核 → 1张特征图）
# kernel_size=3:  卷积核大小 3x3（滑动窗口的大小）
# stride=1:       每次滑1步
# padding=0:      不在边缘补0

conv = nn.Conv2d(
    in_channels=1,      
    out_channels=1,     
    kernel_size=3,     
    stride=1,           
    padding=0          
)

# 为了让输出稳定可预测，手动设置卷积核权重（教学用）
with torch.no_grad():
    conv.weight.fill_(1.0 / 9)   # 每个权重 = 1/9（相当于取平均值）
    conv.bias.zero_()            # 偏置设为0

# ========== 前向传播 ==========
output = conv(input_tensor)
print(f"\n卷积后形状: {output.shape}")  




#//PART2========================================================
import torch
import torch.nn as nn

# ========== 模拟一张特征图 ==========
feature_map = torch.tensor([
    [1, 3, 2, 1],
    [4, 6, 5, 2],
    [3, 2, 1, 0],
    [5, 4, 3, 1]
], dtype=torch.float32).unsqueeze(0).unsqueeze(0)


print(f"池化前形状: {feature_map.shape}")  
print(f"池化前:\n{feature_map.squeeze()}")

# ========== 创建 MaxPool2d ==========
# kernel_size=2: 窗口大小 2x2
# stride=2:      每次滑2步（不重叠）
pool = nn.MaxPool2d(kernel_size=2, stride=2)  

output = pool(feature_map)
print(f"\n池化后形状: {output.shape}")  
print(f"池化后:\n{output.squeeze()}")

# ========== 池化对形状的影响 ==========
"""
MaxPool2d(kernel_size=2, stride=2) 的效果：
  - 通道数 (C): 不变
  - 高 (H):     变为原来的 1/2
  - 宽 (W):     变为原来的 1/2

输入 [N, C, H, W] → 输出 [N, C, H/2, W/2]
"""

# ========== 验证：连续卷积+池化后的形状变化 ==========
print("\n=== 卷积 + 池化的完整流程 ===")

x = torch.randn(1, 1, 28, 28)  # 1张 28x28 的灰度图
print(f"输入:              {x.shape}")                       #shape 希望同学自己先在脑子里算一遍，在跟打印结果做对比

conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
x = conv1(x)
print(f"Conv(1→16, pad=1): {x.shape}")  

pool1 = nn.MaxPool2d(2, 2)
x = pool1(x)
print(f"MaxPool(2x2):      {x.shape}")  

conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
x = conv2(x)
print(f"Conv(16→32, pad=1):{x.shape}")  

pool2 = nn.MaxPool2d(2, 2)
x = pool2(x)
print(f"MaxPool(2x2):      {x.shape}")  



#PART3=====================================================================

import torch
import torch.nn as nn



# 输入：2张 RGB 图片（3通道）
x = torch.randn(2, 3, 32, 32)  # [N=2, C=3, H=32, W=32]
print(f"输入:              {x.shape}  ← 2张彩色图片")

# 第1层卷积：3通道 → 16通道（16个卷积核，每个扫描3通道输入）
conv_1 = nn.Conv2d(in_channels=3, out_channels=16,    
                  kernel_size=3, padding=1)
x = conv_1(x)
print(f"Conv(3→16):        {x.shape}  ← 16张特征图(每个卷积核输出1张)")

# 池化：高宽减半
pool = nn.MaxPool2d(2, 2)
x = pool(x)
print(f"MaxPool:           {x.shape}  ← 高宽减半，通道不变")

# 第2层卷积：16通道 → 32通道（32个卷积核，每个扫描16通道输入）
conv_2 = nn.Conv2d(in_channels=16, out_channels=32,   
                  kernel_size=3, padding=1)
x = conv_2(x)
print(f"Conv(16→32):       {x.shape}  ← 32张特征图")

x = pool(x)
print(f"MaxPool:           {x.shape}")

# 第3层卷积：32通道 → 64通道
conv_3 = nn.Conv2d(in_channels=32, out_channels=64,    
                  kernel_size=3, padding=1)
x = conv_3(x)
print(f"Conv(32→64):       {x.shape}  ← 64张特征图")

x = pool(x)
print(f"MaxPool:           {x.shape}")

# ========== 统计参数量 ==========
total = sum(p.weight.numel() + p.bias.numel() for p in [conv_1, conv_2, conv_3])
print(f"\n三个卷积层总参数: {total}")

"""
思考题：
Q1: Conv2d(3, 16, 3) 这层有多少个参数？
    答：每个卷积核有 3×3×3 = 27 个权重 + 1 个偏置 = 28
       16 个卷积核 → 16 × 28 = 448 个参数

Q2: 为什么通道数通常设计为逐渐增多（3→16→32→64）？
    答：浅层提取简单特征（边缘、颜色），需要的通道少；
       深层提取复杂特征（组合特征），需要更多通道来描述。
       同时空间尺寸在减小（32→16→8→4），增加通道可以补偿信息容量。
"""


#PART4:==================================================================
import torch
import torch.nn as nn

class MyCNN(nn.Module):
    """
    一个简单的 CNN,输入 [N, 3, 32, 32](CIFAR-10 尺寸)，输出 [N, 10](10个类别)。

    网络结构(Conv → ReLU → Pool)* 3 → Flatten → Linear → Linear
    """

    def __init__(self, num_classes=10):
        super(MyCNN, self).__init__()

        # ===== 特征提取层（Conv + ReLU + Pool）=====
        self.features = nn.Sequential(
            # Block 1: 3通道(RGB) → 16通道, 32x32 → 16x16
            nn.Conv2d(in_channels=3, out_channels=16,   
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),    

            # Block 2: 16 → 32, 16x16 → 8x8
            nn.Conv2d(in_channels=16, out_channels=32,   
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            # Block 3: 32 → 64, 8x8 → 4x4
            nn.Conv2d(in_channels=32, out_channels=64,   
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # ===== 分类层 =====
        self.classifier = nn.Sequential(
            # Flatten 后：64 * 4 * 4 = 1024
            nn.Linear(in_features=1024, out_features=128), 
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10)   
        )

    def forward(self, x):
        x = self.features(x)              
        x = x.view(x.size(0), -1)         # Flatten: [N, 64, 4, 4] → [N, 1024]
        x = self.classifier(x)            
        return x


# ========== 验证数据流 ==========

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyCNN(num_classes=10).to(device)

# 模拟 CIFAR-10 输入：4张 32x32 的彩色图
x = torch.randn(4, 3, 32, 32).to(device)

print("=== 完整数据流形状追踪 ===\n")                         #shape 希望同学自己先在脑子里算一遍，在跟打印结果做对比
print(f"输入:                   {x.shape}")          

# 逐层追踪
f = model.features
x1 = f[0:3](x)    # Conv + ReLU + Pool
print(f"Block1 (Conv3→16+Pool): {x1.shape}")       

x2 = f[3:6](x1)   # Conv + ReLU + Pool
print(f"Block2 (Conv16→32+Pool):{x2.shape}")        

x3 = f[6:9](x2)   # Conv + ReLU + Pool
print(f"Block3 (Conv32→64+Pool):{x3.shape}")       

xf = x3.view(x3.size(0), -1)
print(f"Flatten:                {xf.shape}")       

c1 = model.classifier[0](xf)
print(f"Linear(1024→128):       {c1.shape}")        

output = model(x)
print(f"\n最终输出:               {output.shape}")    
print(f"预测类别:               {output.argmax(dim=1).tolist()}")




#PART5 SOP ===================================================================
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images)
            pred_labels = pred.argmax(dim=1)
            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)
    return correct / total


# ========== 主流程（补全）==========

# 1. 数据加载（CIFAR-10）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 RGB均值
                         std=[0.2023, 0.1994, 0.2010])
])

train_data = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)    
test_data = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)     

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)    
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)     

# 2. 创建模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyCNN(num_classes=10).to(device)    

# 3. 损失函数和优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. 训练循环
epochs = 5
for epoch in range(epochs):
    train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device)
    test_acc = evaluate(model, test_loader, device)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Acc: {test_acc*100:.2f}%")

# 5. 保存模型
torch.save(model.state_dict(), "mycnn_cifar10.pth")
