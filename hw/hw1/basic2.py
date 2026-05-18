import torch

# 1. 创建一个形状为 (4, 3) 的随机张量，元素服从标准正态分布（均值为0，标准差为1）
tensor_a = torch.randn(________, ________)
print(f"tensor_a 的形状: {tensor_a.________}")  # 应输出 torch.Size([4, 3])
print(f"tensor_a 的数据类型: {tensor_a.________}")  # 应输出 torch.float32
print(f"tensor_a 的维度数: {tensor_a.________}")  # 应输出 2



# 2. 张量形状变换 —— view 操作（不改变底层数据，只改变视图）
# 将 (4, 3) 重塑为 (2, 6)
tensor_b = tensor_a.________(________, ________)
print(f"tensor_b 的形状: {tensor_b.shape}")  # 应输出 torch.Size([2, 6])



# 3. 使用 -1 让 PyTorch 自动推断维度大小
# 将 (4, 3) 重塑为 (3, ?)，PyTorch 会自动计算 ? 的值
tensor_c = tensor_a.view(________, ________)
print(f"tensor_c 的形状: {tensor_c.shape}")  # 应输出 torch.Size([3, 4])



# 4. 增加一个维度（unsqueeze），常用于将单张图片扩展为 batch 维度
# 模拟一张 28x28 的单通道灰度图像
tensor_d = torch.randn(________, ________)  # 创建形状 (1, 28, 28) 的张量（C, H, W）
print(f"tensor_d 的形状: {tensor_d.shape}")  # 应输出 torch.Size([1, 28, 28])



# 使用 unsqueeze 在第0维增加 batch 维度，变成 (N, C, H, W) 格式
tensor_e = tensor_d.unsqueeze(dim=________)
print(f"tensor_e 的形状: {tensor_e.shape}")  # 应输出 torch.Size([1, 1, 28, 28])



# 5. 移除维度（squeeze）
tensor_f = tensor_e.squeeze(dim=________)
print(f"tensor_f 的形状: {tensor_f.shape}")  # 应输出 torch.Size([1, 28, 28])



# 6. Flatten 操作 —— 将多维张量展平为一维（神经网络中从卷积到全连接的关键变换）
# 将形状 (N, C, H, W) 的数据展平为 (N, C*H*W)
flatten_layer = torch.nn.________()
tensor_g = flatten_layer(________)
print(f"tensor_g 的形状: {tensor_g.shape}")  # 应输出 torch.Size([1, 784])


# ========================section2:使用内置数据集和dataloader ========================

from torch.utils.data import ________
from torchvision import datasets
from torchvision.transforms import ________

# 1. 定义数据转换：将 PIL 图像转换为张量（像素值从 [0,255] 归一化到 [0.0, 1.0]）
transform = ________()

# 2. 下载并加载训练数据集
# FashionMNIST 是 10 类服装分类数据集，每张图像是 28x28 的灰度图
training_data = datasets.FashionMNIST(
    root=________,           # 数据集存储路径
    train=________,          # 是否为训练集
    download=________,       # 如果本地没有，是否自动下载
    transform=________       # 应用的图像变换
)

# 3. 下载并加载测试数据集
test_data = datasets.FashionMNIST(
    root="data",
    train=________,          # 测试集这里应该填什么？
    download=True,
    transform=________
)

# 4. 创建 DataLoader
# 训练集通常需要 shuffle=True（打乱顺序），测试集通常 shuffle=False
batch_size = 64
train_dataloader = DataLoader(
    dataset=________,        # 传入数据集对象
    batch_size=________,     # 每批次的样本数量
    shuffle=________         # 是否在每个 epoch 开始时打乱数据
)

test_dataloader = DataLoader(
    dataset=test_data,
    batch_size=batch_size,
    shuffle=________         # 测试集是否需要打乱？为什么？
)

# 5. 遍历 DataLoader，查看数据的形状
for X, y in test_dataloader:
    print(f"X 的形状 [N, C, H, W]: {X.shape}")
    # 期望输出: torch.Size([64, 1, 28, 28])
    # 其中 N=64（batch_size）, C=1（灰度通道）, H=28, W=28

    print(f"y 的形状: {y.shape}")  # 期望输出: torch.Size([64])
    print(f"y 的数据类型: {y.dtype}")  # 期望输出: torch.int64
    break  # 只查看第一批数据

# 6. 计算数据集的大小
print(f"训练集样本总数: {len(________.dataset)}")  # 应输出 60000
print(f"测试集 batch 数量: {len(________)}")  # 应输出 10000/64 向上取整



#========================section3:构建神经网络模型======================

from torch import ________
import torch.nn as nn

# 确定计算设备（优先使用 GPU）
device = torch.device("cuda" if torch.cuda.________ else "cpu")
print(f"使用设备: {device}")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).________()  # 调用父类构造函数

        # 展平层：将输入的 (N, 1, 28, 28) 展平为 (N, 784)
        # start_dim=1 表示从第1维开始展平，保留第0维（batch维度）
        self.flatten = nn.________(start_dim=________)

        # 特征提取网络：Sequential 将多个层按顺序组合
        self.linear_relu_stack = nn.________(
            # 第一层全连接：784 → 512
            # 输入特征数 784 = 1 * 28 * 28（展平后的图像大小）
            nn.________(in_features=________, out_features=________),
            nn.________(),  # 非线性激活函数

            # 第二层全连接：512 → 256
            nn.Linear(in_features=________, out_features=________),
            nn.________(),  # 再次激活

            # 输出层：256 → 10
            # 输出10个类别对应 FashionMNIST 的10种服装类别
            nn.Linear(in_features=________, out_features=________)
        )

    def forward(self, x):
        """定义数据前向传播的路径"""
        # 第一步：展平
        x = self.________(x)  # 形状变化: (N, 1, 28, 28) → (N, 784)

        # 第二步：通过全连接网络
        logits = self.________(x)  # 形状变化: (N, 784) → (N, 10)

        return logits

# 实例化模型并移至计算设备
model = NeuralNetwork().________(device)
print(model)

# ===== 验证模型输出的形状 =====
# 创建一个模拟的 batch 输入数据
dummy_input = torch.randn(________, ________, ________, ________).to(device)
# dummy_input 形状: (batch_size, channels, height, width) = (8, 1, 28, 28)

output = model(________)
print(f"\n输入形状:  {dummy_input.shape}")   # torch.Size([8, 1, 28, 28])
print(f"输出形状:  {output.shape}")         # torch.Size([8, 10])
print(f"输出含义:  batch中每张图片对应10个类别的未归一化分数（logits）")



#==============================section4:循环训练sop================



def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    """
    执行一个 epoch 的训练。

    标准训练 SOP(5步曲):
    1. 前向传播:model(X) → pred → loss_fn(pred, y) → loss
    2. 反向传播:loss.backward() 计算梯度
    3. 更新参数:optimizer.step() 使用梯度更新权重
    4. 清空梯度:optimizer.zero_grad() 防止梯度累积
    5. (可选)打印损失，监控训练进度
    """
    model.________()  # 将模型设为训练模式（启用 Dropout、BatchNorm 等训练行为）

    total_loss = 0
    num_batches = len(dataloader)

    for batch_idx, (X, y) in enumerate(dataloader):
        # 将数据移到计算设备（GPU/CPU）上
        X = X.to(________)
        y = y.to(________)

        # ========== 第1步：前向传播（计算预测和损失）==========
        # 模型对输入数据进行预测
        pred = ________(________)   # 输出形状: [batch_size, 10]

        # 计算预测值与真实标签之间的损失
        loss = ________(________, ________)

        # ========== 第2步：反向传播（计算梯度）==========
        loss.________()

        # ========== 第3步：更新模型参数==========
        ________.step()

        # ========== 第4步：清空梯度（重要！防止梯度累积）==========
        ________.________()

        # 累加损失用于计算平均损失
        total_loss += loss.________()  # 用 .item() 取出 Python 数值

        # 每100个 batch 打印一次进度
        if batch_idx % 100 == 0:
            current_samples = batch_idx * len(X)
            total_samples = len(dataloader.________)
            print(f"  Batch {batch_idx:>3d} | Loss: {loss.item():>7f} "
                  f"[{current_samples:>5d}/{total_samples:>5d}]")

    avg_loss = total_loss / num_batches
    return avg_loss


# ===== 验证训练函数所需组件 =====
# 以下代码展示了使用 train_one_epoch 前需要准备的4个关键组件

# 组件1：模型
model = NeuralNetwork().to(device)

# 组件2：损失函数
# 交叉熵损失 = LogSoftmax + NLLLoss，适合多分类任务
# 输入：原始 logits [N, 10]，目标标签 [N]（类别索引）
criterion = nn.________()

# 组件3：优化器
# SGD：随机梯度下降，lr 是学习率，控制参数更新的步长
optimizer = torch.optim.________(model.parameters(), lr=________)

# 组件4：数据加载器（前面已创建）
# train_dataloader

print("训练函数准备完毕!4个关键组件:")
print(f"  - 模型: {type(model).__name__}")
print(f"  - 损失函数: {type(criterion).__name__}")
print(f"  - 优化器: {type(optimizer).__name__}")
print(f"  - 数据加载器: 每批 {train_dataloader.batch_size} 张图")




#===========================section5:测试评估模型的基本sop ==============================
import torch

def test_model(dataloader, model, loss_fn, device):
    """
    评估模型在测试集上的表现。

    评估阶段的关键特点（与训练的区别）：
    1. 不需要计算梯度（节省显存，加速推理）
    2. 不需要更新参数
    3. 需要统计整体准确率和平均损失
    """
    model.________()  # 将模型设为评估模式（关闭 Dropout、BatchNorm 使用运行统计）

    size = len(dataloader.________)  # 测试集总样本数
    num_batches = len(dataloader)     # batch 数量
    test_loss = 0
    correct = 0

    # 使用 torch.no_grad() 上下文管理器，禁用梯度计算
    with torch.________():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            # 前向传播得到预测结果
            pred = ________(________)  # 形状: [batch_size, 10]

            # 累加损失
            test_loss += loss_fn(pred, y).________  # 注意这里不要加括号，取 tensor

            # 计算正确预测的数量
            # pred.argmax(1) 返回每张图预测概率最大的类别索引
            # 形状: [batch_size]
            predicted_labels = pred.________(dim=________)

            # 将预测结果与真实标签比较，统计正确数量
            correct += (predicted_labels == y).type(torch.________).sum().________

    # 计算平均值
    test_loss /= num_batches
    correct /= size
    accuracy = 100 * correct

    print(f"Test Results:\n"
          f"  Accuracy: {accuracy:>0.1f}%\n"
          f"  Avg Loss: {test_loss:>8f}")

    return accuracy, test_loss

# ===== 思考：训练模式 vs 评估模式的区别 =====
# 1. model.train() 会启用：
#    - Dropout 层的随机丢弃
#    - BatchNorm 层使用当前 batch 的均值和方差
# 2. model.eval() 会启用：
#    - Dropout 层关闭（所有神经元参与计算）
#    - BatchNorm 层使用训练阶段累积的全局均值和方差
#
# Q: 如果在测试时忘记调用 model.eval()，会发生什么？
# A: ________________________________________________________