import torch

# ========== 第1部分：创建张量 ==========

# 1.1 从 Python 列表创建张量
data = [1, 2, 3, 4, 5]
t1 = torch.tensor(data)
print(f"t1 = {t1}")
print(f"t1 的形状 (shape): {t1.shape}")      # 输出 torch.Size([5])
print(f"t1 的数据类型 (dtype): {t1.dtype}")  # 默认 int64
print(f"t1 的维度数: {t1.ndim}")             # 输出 1



# 1.2 创建全零张量（常用于初始化偏置或占位）
zeros = torch.zeros(3, 4)   # 3行4列的全0矩阵
print(f"\nzeros 的形状: {zeros.shape}")       # 应输出 torch.Size([3, 4])
print(f"zeros:\n{zeros}")



# 1.3 创建全一张量
ones = torch.ones(2, 3)
print(f"\nones 的形状: {ones.shape}")          # 应输出 torch.Size([2, 3])



# 1.4 创建随机张量（元素服从 0~1 均匀分布）
# 在深度学习中，随机张量常用于初始化模型参数或模拟输入数据
rand = torch.rand(2, 2)
print(f"\nrand:\n{rand}")



# 1.5 创建随机整数张量（元素在 [low, high) 范围内随机取整）
# 模拟一批类别标签：5张图片，每张图片属于 0~9 中的某一类
labels = torch.randint(low=0, high=10, size=(5,))
print(f"\n随机标签: {labels}")
print(f"labels 的形状: {labels.shape}")         # 应输出 torch.Size([5])

#====================================================================================================

# ========== 第2部分：查看张量的关键属性 ==========

# 创建一张 "模拟图片" 来理解张量的属性
# 想象这是 3 张 28×28 的灰度图（如手写数字图片）
image_batch = torch.rand(3, 1, 28, 28)
# shape [3, 1, 28, 28] 的含义：
#   第0维 3  = 3张图片（batch size，批次大小）
#   第1维 1  = 1个颜色通道（灰度图=1，彩色图=3）
#   第2维 28 = 图片高度 28 像素
#   第3维 28 = 图片宽度 28 像素

print("\n========== 张量属性查看 ==========")
print(f"image_batch 的形状: {image_batch.shape}")         # torch.Size([3, 1, 28, 28])
print(f"image_batch 的数据类型: {image_batch.dtype}")      # torch.float32
print(f"image_batch 存储的设备: {image_batch.device}")      # cpu（或 cuda:0）
print(f"image_batch 的维度数: {image_batch.ndim}")          # 4


# ========== 第3部分：索引和切片（从张量中取数据）==========

t = torch.tensor([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

print("\n========== 索引与切片 ==========")
print(f"第0行: {t[0]}")              # tensor([1, 2, 3]) —— 取第0行
print(f"第1列: {t[:, 1]}")           # tensor([2, 5, 8]) —— 取第1列（:表示所有行）
print(f"第1行第2列: {t[1, 2]}")      # tensor(6) —— 取单个元素
print(f"前2行前2列:\n{t[:2, :2]}")  # tensor([[1, 2],
                                     #         [4, 5]])



# ========== view()：重塑张量形状 ==========

# 2.1 基本 reshape
a = torch.arange(12)          # 创建一个含 0~11 共12个元素的张量 [12]
print(f"a 的原始形状: {a.shape}")         # torch.Size([12])
print(f"a 的内容: {a}")




# 把 12 个元素变成 3×4 的矩阵
b = a.view(3, 4)              # 填空1：括号内填目标形状
print(f"\nb = a.view(3, 4):")
print(f"b 的形状: {b.shape}")             # torch.Size([3, 4])
print(f"b:\n{b}")



# 2.2 用 -1 让 PyTorch 自动计算
# -1 的意思是"这一维的大小你帮我算"，前提是总元素数能对上
c = a.view(2, -1)             # 填空2：2行，列数自动算 (12/2=6)
print(f"\nc = a.view(2, -1):")
print(f"c 的形状: {c.shape}")             # torch.Size([2, 6])



d = a.view(-1, 3)             # 填空3：3列，行数自动算 (12/3=4)
print(f"\nd = a.view(-1, 3):")
print(f"d 的形状: {d.shape}")             # torch.Size([4, 3])





# ========== unsqueeze()：增加一个维度 ==========
# 作用：在指定位置插入一个大小为1的维度
# 常见场景：给单张图片增加 batch 维度，使其能输入神经网络

print("\n========== unsqueeze()：增加维度 ==========")

# 假设有一张灰度图 (1通道, 28高, 28宽) —— 3维张量
single_image = torch.rand(1, 28, 28)
print(f"单张图片形状: {single_image.shape}")       # torch.Size([1, 28, 28])



# 神经网络期望的输入格式是 [N, C, H, W]，N 是 batch 大小（图片张数）
# 现在只有1张图，需要在第0维插入一个大小为1的 batch 维度
batched_image = single_image.unsqueeze(dim = 0)       # 在第0维插入
print(f"增加batch维度后: {batched_image.shape}")    # torch.Size([1, 1, 28, 28])
# 现在格式是 [1张图, 1通道, 28高, 28宽]，可以输入网络了！




# 再练习一个：在最后一维增加维度
vec = torch.tensor([1, 2, 3, 4])
print(f"\n向量形状: {vec.shape}")                    # torch.Size([4])
col_vec = vec.unsqueeze(dim = 1)                       # 在第1维插入
print(f"变成列向量: {col_vec.shape}")                # torch.Size([4, 1])




# ========== squeeze()：去掉大小为1的维度 ==========

print("\n========== squeeze()：去掉维度 ==========")

# 假设网络输出了一个形状为 [1, 10] 的结果（1张图片，10个类别的分数）
model_output = torch.rand(1, 10)
print(f"模型输出形状: {model_output.shape}")         # torch.Size([1, 10])

# 如果只有一张图，想去掉 batch 维度得到 [10]
output = model_output.squeeze(dim = 1)                 #去掉第0维的大小为1的维度
print(f"去掉batch后: {output.shape}")                # torch.Size([10])




# ========== 综合练习：模拟神经网络中的形状变化 ==========

print("\n========== 综合：模拟一次前向传播的形状变化 ==========")

# Step 0: 原始输入是 8 张 28×28 的灰度图
x = torch.rand(8, 1, 28, 28)     # [N=8, C=1, H=28, W=28]
print(f"Step 0 输入:       {x.shape}")



# Step 1: Flatten 展平 —— 把每张图的像素排成一行
# 每张图有 1*28*28 = 784 个像素
x = x.view(8 , -1)                 # 8行，列数自动算 (8*1*28*28 / 8 = 784)
print(f"Step 1 展平后:      {x.shape}")    # torch.Size([8, 784])




# Step 2: 假设经过一层线性变换，输出10个类别
# nn.Linear(784, 10) 会把 [8, 784] 变成 [8, 10]
# 这里我们用随机权重模拟一下
weight = torch.rand(784, 10)
bias = torch.rand(10)
x = x @ weight + bias             # 矩阵乘法 + 偏置
print(f"Step 2 线性变换后:   {x.shape}")    # torch.Size([8, 10])



# Step 3: 取分数最高的类别作为预测结果
pred = x.argmax(dim = 1)            # 在第1维上取最大值的索引
print(f"Step 3 预测结果:     {pred.shape}")  # torch.Size([8])
print(f"预测标签: {pred}")                     # 8个整数，每个在 0~9 之间                       


# ===============================================================================================

#section 3:加载数据集：=======================

import torch
from torch.utils.data import DataLoader          # 填空1：导入 DataLoader
from torchvision import datasets                 # 填空2：导入 datasets
from torchvision.transforms import ToTensor      # 填空3：导入 ToTensor 变换

# ========== 第1步：加载 FashionMNIST 数据集 ==========
# FashionMNIST 是一个服装分类数据集，共10个类别：
# T恤、裤子、套头衫、裙子、外套、凉鞋、衬衫、运动鞋、包、短靴
# 每张图是 28×28 像素的灰度图

# 训练集
train_data = datasets.FashionMNIST(
    root="data",                # 数据存到当前目录的 data/ 文件夹下
    train=True,                 # 填空4：True 表示加载训练集
    download=True,              # 如果本地没有，自动从网上下载
    transform=ToTensor()        # 把 PIL 图片转成张量，像素值 [0,255] → [0.0, 1.0]
)

# 测试集
test_data = datasets.FashionMNIST(
    root="data",
    train=False,                # 填空5：False 表示加载测试集
    download=True,
    transform=ToTensor()
)

print(f"训练集图片数量: {len(train_data)}")     # 应输出 60000
print(f"测试集图片数量: {len(test_data)}")       # 应输出 10000

# 看看一张图片长什么样
image, label = train_data[0]   # 取第0张图片（返回值：(图片张量, 标签)）
print(f"\n第0张图片:")
print(f"  图片形状: {image.shape}")              # torch.Size([1, 28, 28])
# shape 含义：[通道数=1(灰度), 高度=28, 宽度=28]
print(f"  标签: {label}")                        # 整数，0~9之间


# ========== 第2步：用 DataLoader 分批加载数据 ==========
# DataLoader 的作用：
#   1. 自动把数据集分成小批次（batch）
#   2. 每个批次返回 (图片张量, 标签张量)
#   3. 可以设置是否打乱顺序（shuffle）

batch_size = 64  # 每批次64张图片

train_loader = DataLoader(
    dataset=train_data,         # 填空6：传入训练数据集
    batch_size=batch_size,      # 每批64张
    shuffle=True                # 填空7：True 表示打乱顺序（训练时一般打乱）
)

test_loader = DataLoader(
    dataset=test_data,          # 填空8：传入测试数据集
    batch_size=batch_size,
    shuffle=False               # 测试时不打乱，方便复现结果
)

# ========== 第3步：遍历 DataLoader，查看数据形状 ==========

# train_loader 是一个可迭代对象，每次迭代返回一个 batch 的数据
for images, labels in train_loader:
    print(f"\n一个 batch 的数据:")
    print(f"  images 形状: {images.shape}")
    # 应输出: torch.Size([64, 1, 28, 28])
    # 含义: [64张图, 1通道(灰度), 28高, 28宽]

    print(f"  labels 形状: {labels.shape}")
    # 应输出: torch.Size([64])
    # 含义: 64个整数标签

    print(f"  labels 数据类型: {labels.dtype}")   # torch.int64
    print(f"  images 像素范围: [{images.min():.2f}, {images.max():.2f}]")
    # 约 [0.00, 1.00]，因为 ToTensor 做了归一化
    break  # 只看第一个 batch

# ========== 第4步：计算数据集信息 ==========

print("\n========== 数据集统计 ==========")
print(f"训练集 batch 数量: {len(train_loader)}")
# = 60000 / 64 = 938 (向上取整)

print(f"测试集 batch 数量: {len(test_loader)}")
# = 10000 / 64 = 157 (向上取整)

# DataLoader 可以像列表一样访问其属性
print(f"训练集总样本数: {len(train_loader.dataset)}")   # 60000
print(f"测试集总样本数: {len(test_loader.dataset)}")     # 10000


#section4:定义一个神经网络：=========================

import torch
from torch import nn

# ========== 定义网络结构 ==========

class MyNet(nn.Module):                          # 填空1：继承 nn.Module
    def __init__(self):
        super(MyNet, self).__init__()            # 填空2：调用父类构造函数

        # 展平层：把 [N, 1, 28, 28] 变成 [N, 784]
        # start_dim=1 表示从第1维开始展平，保留第0维（batch维度）
        self.flatten = nn.Flatten(start_dim=1)

        # 全连接网络：Sequential 把多个层按顺序串起来
        self.network = nn.Sequential(
            # 第一层：784 → 256
            # 输入特征数 784 = 1 * 28 * 28（展平后的图片大小）
            nn.Linear(in_features=784, out_features=256),
            nn.ReLU(),                           # 激活函数，增加非线性

            # 第二层：256 → 64
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),

            # 输出层：64 → 10（对应10个服装类别）
            nn.Linear(in_features=64, out_features=10)
            # 注意：最后一层一般不加激活函数（后面接 CrossEntropyLoss 会内部处理）
        )

    def forward(self, x):
        """描述数据在网络中的流动路径"""
        x = self.flatten(x)         # 填空3：展平
        x = self.network(x)         # 填空4：通过全连接网络
        return x


# ========== 创建模型实例 ==========

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = MyNet().to(device)          # 填空5：创建模型并移到指定设备
print(model)

# ========== 验证输入输出形状 ==========

# 模拟一个 batch 的输入：16张 28×28 的灰度图
dummy_images = torch.rand(16, 1, 28, 28).to(device)
print(f"\n输入形状:  {dummy_images.shape}")     # torch.Size([16, 1, 28, 28])

output = model(dummy_images)
print(f"输出形状:  {output.shape}")             # torch.Size([16, 10])
# 输出含义：16张图片，每张图对应10个类别的原始分数（logits）

# 获取每张图片的预测类别
predicted = output.argmax(dim=1)                # 填空6：在第1维取最大值的索引
print(f"预测形状:  {predicted.shape}")           # torch.Size([16])
print(f"预测结果:  {predicted}")                  # 16个0~9之间的整数


# ========== （思考题）画出本网络的数据流形状图 ==========
"""
请根据下面的提示，在注释中补全每一步的形状：

输入图片:          [16,   1,  28,  28]
                       ↓
Flatten(start_dim=1)   [16,  ______]      ← 填空7:1*28*28 = ?
                       ↓
Linear(784 → 256)      [16,  256]
                       ↓
ReLU()                 [16,  256]         ← ReLU不改变形状
                       ↓
Linear(256 → 64)       [16,  ______]      ← 填空8
                       ↓
ReLU()                 [16,  64]
                       ↓
Linear(64 → 10)        [16,  ______]      ← 填空9
                       ↓
argmax(dim=1)          [16]                ← 填空10:最终的形状
"""


# section5:训练模型的sop：===============================================================


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# ========== 第0步：准备数据、模型、损失函数、优化器 ==========

# 0.1 加载数据（同题目三）
train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# 0.2 创建模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MyNet().to(device)

# 0.3 定义损失函数
# CrossEntropyLoss = LogSoftmax + NLLLoss，用于多分类问题
# 输入：模型的原始输出 logits [N, num_classes]
# 目标：标签 [N]，值为类别索引（0, 1, 2, ...）
loss_fn = nn.CrossEntropyLoss()

# 0.4 定义优化器
# SGD = 随机梯度下降，负责根据梯度更新模型参数
# lr = learning rate（学习率），控制每次更新的步长
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# model.parameters() 返回模型中所有需要训练的参数（权重和偏置）

print("准备工作完成！四个要素：")
print(f"  - 数据: {len(train_data)} 张训练图, {len(test_data)} 张测试图")
print(f"  - 模型: {type(model).__name__}")
print(f"  - 损失函数: {type(loss_fn).__name__}")
print(f"  - 优化器: {type(optimizer).__name__}, lr={optimizer.defaults['lr']}")


# ========== 第1步：训练函数 ==========

def train(model, dataloader, loss_fn, optimizer):
    model.train()                    # 填空1：设为训练模式
    total_loss = 0

    for images, labels in dataloader:
        # 把数据移到 GPU（如果有的话）
        images = images.to(device)
        labels = labels.to(device)

        # ---- SOP 第①步：前向传播（算预测和损失）----
        pred = model(images)                      # 填空2：模型做预测
        loss = loss_fn(pred, labels)              # 填空3：计算损失

        # ---- SOP 第②步：反向传播（算梯度）----
        loss.backward()                           # 填空4：反向传播

        # ---- SOP 第③步：更新参数（优化器根据梯度调整权重）----
        optimizer.step()                          # 填空5：更新参数

        # ---- SOP 第④步：清空梯度（防止梯度累积）----
        optimizer.zero_grad()                     # 填空6：清空梯度

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ========== 第2步：测试/评估函数 ==========

def test(model, dataloader, loss_fn):
    model.eval()                     # 填空7：设为评估模式
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():            # 填空8：不计算梯度（节省内存，加速）
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images)
            total_loss += loss_fn(pred, labels).item()

            # 计算正确率
            pred_labels = pred.argmax(dim=1)       # 填空9：取预测类别
            correct += (pred_labels == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    return accuracy, total_loss / len(dataloader)


# ========== 第3步：主循环（跑多个 epoch）==========

epochs = 5
print(f"\n开始训练，共 {epochs} 个 epoch...")

for epoch in range(epochs):
    train_loss = train(model, train_loader, loss_fn, optimizer)
    test_acc, test_loss = test(model, test_loader, loss_fn)

    print(f"Epoch {epoch+1:>2d}/{epochs} | "
          f"Train Loss: {train_loss:.4f} | "
          f"Test Loss: {test_loss:.4f} | "
          f"Test Accuracy: {test_acc:.2f}%")

print("\n训练完成！")

# ========== 第4步：保存模型 ==========

torch.save(model.state_dict(), "my_model.pth")    # 填空10：保存模型参数
print("模型已保存到 my_model.pth")