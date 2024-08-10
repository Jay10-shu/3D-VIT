①	数据集说明
瘤周3mm的训练集：202例；测试集：81例
②	实验设置
损失函数是CE交叉熵
实验是在配备 NVIDIA A6000 GPU 的 Ubuntu 22.04.4 系统上使用 Python 3.10 环境中的 PyTorch 实现的。由于内存的限制和效率的需求，3D图像被下采样到 64 × 64 × 64 的分辨率。3D-Vit被作为主干网络，其中采用随机梯度下降（SGD）优化器进行训练，学习率是0.001。此外，在多层感知器 (MLP) 中应用了平均池化操作。补丁大小配置为 8 × 8 × 8，批处理大小为 16，MLP 在 2048 维内运行。
注：3D ViT模型的来源：https://github.com/lucidrains/vit-pytorch
