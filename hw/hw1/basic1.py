import torch

# ========== 第1部分：创建张量 ==========

# 1.1 从 Python 列表创建张量
data = [1, 2, 3, 4, 5]
t1 = torch.tensor(data)
print(f"t1 = {t1}")
print(f"t1 的形状 (shape): {__your_code__}")      # 填空1：填属性名，输出 torch.Size([5])
print(f"t1 的数据类型 (dtype): {t1.dtype}")  # 默认 int64
print(f"t1 的维度数: {_your_code_}")             # 填空2：填属性名，输出 1



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



d = a.view(___your_code____)             # 填空3：3列，行数自动算 (12/3=4)
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
batched_image = single_image.unsqueeze(____your_code_____)       # 填空4：在第0维插入
print(f"增加batch维度后: {batched_image.shape}")    # torch.Size([1, 1, 28, 28])
# 现在格式是 [1张图, 1通道, 28高, 28宽]，可以输入网络了！




# 再练习一个：在最后一维增加维度
vec = torch.tensor([1, 2, 3, 4])
print(f"\n向量形状: {vec.shape}")                    # torch.Size([4])
col_vec = vec.unsqueeze(___your_code____)                       # 填空5：在第1维插入
print(f"变成列向量: {col_vec.shape}")                # torch.Size([4, 1])




# ========== squeeze()：去掉大小为1的维度 ==========

print("\n========== squeeze()：去掉维度 ==========")

# 假设网络输出了一个形状为 [1, 10] 的结果（1张图片，10个类别的分数）
model_output = torch.rand(1, 10)
print(f"模型输出形状: {model_output.shape}")         # torch.Size([1, 10])

# 如果只有一张图，想去掉 batch 维度得到 [10]
output = model_output.squeeze(___your_code___)                 # 填空6：去掉第0维的大小为1的维度
print(f"去掉batch后: {output.shape}")                # torch.Size([10])




# ========== 综合练习：模拟神经网络中的形状变化 ==========

print("\n========== 综合：模拟一次前向传播的形状变化 ==========")

# Step 0: 原始输入是 8 张 28×28 的灰度图
x = torch.rand(8, 1, 28, 28)     # [N=8, C=1, H=28, W=28]
print(f"Step 0 输入:       {x.shape}")



# Step 1: Flatten 展平 —— 把每张图的像素排成一行
# 每张图有 1*28*28 = 784 个像素
x = x.view(____your_code_____)                 # 填空7：8行，列数自动算 (8*1*28*28 / 8 = 784)
print(f"Step 1 展平后:      {x.shape}")    # torch.Size([8, 784])




# Step 2: 假设经过一层线性变换，输出10个类别
# nn.Linear(784, 10) 会把 [8, 784] 变成 [8, 10]
# 这里我们用随机权重模拟一下
weight = torch.rand(784, 10)
bias = torch.rand(10)
x = x @ weight + bias             # 矩阵乘法 + 偏置
print(f"Step 2 线性变换后:   {x.shape}")    # torch.Size([8, 10])



# Step 3: 取分数最高的类别作为预测结果
pred = x.argmax(___your_code____)            # 填空8：在第1维上取最大值的索引
print(f"Step 3 预测结果:     {pred.shape}")  # torch.Size([8])
print(f"预测标签: {pred}")                     # 8个整数，每个在 0~9 之间                       