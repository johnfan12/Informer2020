# Informer MindSpore 版本

这是原始 PyTorch 版本的 Informer 模型迁移到 MindSpore 2.6 的版本。

## 主要变更

### 1. 依赖项变更
- 将 PyTorch 替换为 MindSpore 2.6
- 更新 `requirements_mindspore.txt` 文件

### 2. 核心组件迁移

#### 2.1 神经网络层
- `torch.nn.Module` → `mindspore.nn.Cell`
- `torch.nn.Linear` → `mindspore.nn.Dense`
- `torch.nn.Conv1d` → `mindspore.nn.Conv1d`
- `torch.nn.LayerNorm` → `mindspore.nn.LayerNorm`
- `torch.nn.Dropout` → `mindspore.nn.Dropout`
- `torch.nn.Embedding` → `mindspore.nn.Embedding`

#### 2.2 激活函数
- `torch.nn.functional.relu` → `mindspore.ops.relu`
- `torch.nn.functional.gelu` → `mindspore.ops.gelu`

#### 2.3 操作函数
- `torch.matmul` → `mindspore.ops.matmul`
- `torch.einsum` → `mindspore.ops.matmul` (手动实现)
- `torch.softmax` → `mindspore.ops.softmax`
- `torch.cat` → `mindspore.ops.concat`
- `torch.topk` → `mindspore.ops.topk`

#### 2.4 张量操作
- `torch.Tensor` → `mindspore.Tensor`
- `torch.zeros` → `mindspore.ops.zeros`
- `torch.ones` → `mindspore.ops.ones`
- `torch.arange` → `mindspore.ops.arange`

### 3. 模型结构变更

#### 3.1 文件结构
创建了对应的 MindSpore 版本文件：
- `models/embed_mindspore.py` - 嵌入层
- `models/attn_mindspore.py` - 注意力机制
- `models/encoder_mindspore.py` - 编码器
- `models/decoder_mindspore.py` - 解码器
- `models/model_mindspore.py` - 主模型
- `exp/exp_informer_mindspore.py` - 实验类
- `main_informer_mindspore.py` - 主程序

#### 3.2 前向传播方法
- `forward()` → `construct()`

### 4. 训练相关变更

#### 4.1 优化器
- `torch.optim.Adam` → `mindspore.nn.Adam`

#### 4.2 损失函数
- `torch.nn.MSELoss` → `mindspore.nn.MSELoss`

#### 4.3 数据加载器
- `torch.utils.data.DataLoader` → `mindspore.dataset.GeneratorDataset`

#### 4.4 设备管理
- `torch.device` → `mindspore.context.set_context`
- `model.to(device)` → 通过 context 设置设备

#### 4.5 模型保存和加载
- `torch.save/torch.load` → `mindspore.save_checkpoint/mindspore.load_checkpoint`

### 5. 特殊处理

#### 5.1 高级索引
PyTorch 的高级索引在 MindSpore 中需要使用 `gather` 或 `gather_nd` 操作替代。

#### 5.2 einsum 操作
MindSpore 不直接支持 einsum，使用 `matmul` 和 `transpose` 组合实现。

#### 5.3 masked_fill 操作
使用 `mindspore.ops.masked_fill` 替代 PyTorch 的 `masked_fill_`。

#### 5.4 数据并行
MindSpore 的数据并行方式与 PyTorch 不同，需要使用不同的配置方法。

## 使用方法

### 1. 安装依赖
```bash
pip install -r requirements_mindspore.txt
```

### 2. 运行训练
```bash
python main_informer_mindspore.py --model informer --data ETTh1
```

### 3. 主要参数
所有参数与原始 PyTorch 版本保持一致，包括：
- `--model`: 模型类型 (informer, informerstack)
- `--data`: 数据集名称
- `--seq_len`: 输入序列长度
- `--pred_len`: 预测序列长度
- `--d_model`: 模型维度
- `--n_heads`: 注意力头数
- 等等...

## 注意事项

1. **设备设置**: MindSpore 通过 `context.set_context` 设置设备，而不是 `model.to(device)`
2. **数据类型**: 确保数据类型与 MindSpore 兼容，主要使用 `ms.float32`
3. **梯度计算**: 使用 `ops.value_and_grad` 进行梯度计算
4. **模型模式**: 使用 `model.set_train(True/False)` 设置训练/评估模式

## 性能说明

MindSpore 版本应该与原始 PyTorch 版本具有相同的模型性能，但可能在以下方面有所不同：
- 训练速度可能因框架优化不同而有差异
- 内存使用模式可能不同
- 某些操作的数值精度可能略有差异

## 兼容性

- MindSpore 2.6+
- Python 3.7+
- 支持 CPU 和 GPU 设备

## 测试

建议在迁移后进行以下测试：
1. 小批量数据的前向传播测试
2. 梯度计算测试
3. 完整训练流程测试
4. 模型保存和加载测试

## 已知问题

1. 某些复杂的张量操作可能需要进一步优化
2. 数据并行功能需要根据具体需求进行调整
3. 某些边界情况可能需要额外处理

这个迁移版本保持了原始模型的核心功能和性能，同时充分利用了 MindSpore 的特性。
