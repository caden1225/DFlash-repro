# DFlash Reproduce - 块扩散推测解码完整复现框架

<p align="center">
  <b>通过 YAML 配置文件指定目标模型，自动训练 DFlash 草稿模型</b>
</p>

---

## 目录

- [概述](#概述)
- [安装](#安装)
- [快速开始](#快速开始)
- [配置文件详解](#配置文件详解)
- [训练流程](#训练流程)
  - [数据准备](#1-数据准备)
  - [启动 vLLM 服务](#2-启动-vllm-服务)
  - [训练草稿模型](#3-训练草稿模型)
- [推理](#推理)
  - [单条推理](#单条推理)
  - [交互式对话](#交互式对话)
  - [批量推理](#批量推理)
- [评估](#评估)
  - [接受长度评估](#接受长度评估)
  - [加速比评估](#加速比评估)
  - [下游任务评估](#下游任务评估)
  - [完整基准测试](#完整基准测试)
- [模块架构](#模块架构)
- [配置参考](#配置参考)
- [故障排除](#故障排除)
- [引用](#引用)

---

## 概述

本项目是一个完整的 **DFlash (Block Diffusion for Flash Speculative Decoding)** 复现框架，允许用户通过简单的 YAML 配置文件指定任意目标大语言模型，自动完成：

1. **数据准备** - 下载/加载数据集，用目标模型重新生成回复以对齐分布
2. **隐藏状态提取** - 从目标模型指定层提取上下文特征（支持在线/离线/混合模式）
3. **草稿模型训练** - 训练轻量级块扩散草稿模型（默认5层Transformer）
4. **推测解码推理** - 使用训练好的草稿模型加速目标模型推理
5. **全面评估** - 接受长度、加速比、下游任务（GSM8K/MATH/HumanEval/MBPP）

### DFlash 核心创新

| 特性 | DFlash | EAGLE-3 | 传统自回归 |
|------|--------|---------|-----------|
| 草稿架构 | **块扩散（非因果注意力）** | 自回归（因果注意力） | - |
| 块生成 | **单次前向传播** | 多次串行前向 | - |
| 条件机制 | **KV注入目标特征** | 输入嵌入拼接 | - |
| 草稿层数 | 5层 | 1层 | - |
| 典型加速比 | **4-6x** | 1.5-2.5x | 1x |

---

## 安装

### 环境要求

- **Python**: 3.10+
- **CUDA**: 12.0+ (推荐)
- **GPU**: 至少 2x H100 (推荐 4x H100)
- **OS**: Linux (Ubuntu 20.04+ 推荐)

### 步骤

```bash
# 1. 克隆项目
git clone <repo-url> dflash-reproduce
cd dflash-reproduce

# 2. 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 3. 安装依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate datasets huggingface_hub
pip install flash-attn --no-build-isolation
pip install vllm>=0.18
pip install pyyaml numpy tqdm sentencepiece protobuf

# 4. 验证安装
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
python train.py --config configs/qwen3_8b.yaml --dry-run
```

### 推荐虚拟环境分离（训练与vLLM）

为避免依赖冲突，建议创建两个独立的虚拟环境：

```bash
# vLLM 环境（用于启动隐藏状态提取服务）
python -m venv venv_vllm
source venv_vllm/bin/activate
pip install "vllm>=0.18"

# 训练环境
python -m venv venv_train
source venv_train/bin/activate
pip install torch transformers accelerate datasets flash-attn pyyaml
```

---

## 快速开始

### 1. 选择配置文件

项目提供多个预置配置文件：

| 配置文件 | 目标模型 | 资源需求 | 备注 |
|---------|---------|---------|------|
| `configs/qwen3_8b.yaml` | Qwen3-8B | 4x H100 | 默认推荐配置 |
| `configs/qwen3_4b.yaml` | Qwen3-4B | 2x H100 | 适合资源有限 |
| `configs/llama3_8b.yaml` | Llama-3.1-8B | 4x H100 | block_size=10 |

### 2. 一键训练

```bash
# 使用预置配置
python train.py --config configs/qwen3_8b.yaml

# 分布式训练（4 GPU）
torchrun --standalone --nproc_per_node=4 train.py --config configs/qwen3_8b.yaml

# 覆盖特定参数
python train.py --config configs/qwen3_8b.yaml --training.epochs 10 --training.learning_rate 3e-4

# 从检查点恢复
python train.py --config configs/qwen3_8b.yaml --resume ./checkpoints/dflash_qwen3_8b/epoch_3
```

### 3. 推理测试

```bash
# 单条推理
python inference_cli.py --config configs/qwen3_8b.yaml --prompt "Explain quantum computing"

# 交互式对话
python inference_cli.py --config configs/qwen3_8b.yaml --interactive

# 批量推理
python inference_cli.py --config configs/qwen3_8b.yaml --input prompts.jsonl --output results.jsonl
```

### 4. 评估

```bash
# 完整基准测试
python evaluate.py --config configs/qwen3_8b.yaml --benchmark

# 仅加速比评估
python evaluate.py --config configs/qwen3_8b.yaml --speedup

# 特定任务评估
python evaluate.py --config configs/qwen3_8b.yaml --tasks gsm8k math500
```

---

## 配置文件详解

配置文件采用 YAML 格式，分为以下主要部分：

```yaml
model:          # 模型架构配置
data:           # 数据集配置
training:       # 训练超参数
hidden_states:  # 隐藏状态提取配置
inference:      # 推理参数
evaluation:     # 评估配置
output:         # 输出目录配置
distributed:    # 分布式训练配置
```

### 创建自定义配置

要为一个新模型训练 DFlash 草稿模型，只需创建一个新的 YAML 文件：

```yaml
# configs/my_model.yaml
model:
  target_model: "your-org/your-model-7b"   # 目标模型 (HuggingFace)
  target_model_type: "qwen3"                # 模型架构类型
  draft_num_layers: 5                       # 草稿模型层数
  draft_vocab_size: 8192                    # 草稿词表大小
  block_size: 16                            # 扩散块大小
  target_layer_ids: null                    # 自动计算
  max_seq_len: 3072                         # 最大序列长度
  dtype: "bfloat16"                         # 训练精度

data:
  dataset_name: "your-dataset"              # 训练数据集
  dataset_type: "huggingface"
  regenerate_responses: true                # 用目标模型重新生成回复
  regen_temperature: 0.6

training:
  epochs: 6
  batch_size: 4
  learning_rate: 0.0006
  gradient_clipping: 1.0
  warmup_ratio: 0.04
  num_anchors: 512
  loss_decay_gamma: 7.0                     # block_size=16 → gamma=7

hidden_states:
  extraction_mode: "online"                 # 在线提取隐藏状态
  vllm_endpoint: "http://localhost:8000/v1"

evaluation:
  datasets: ["gsm8k", "math500", "humaneval"]

output:
  checkpoint_dir: "./checkpoints/dflash_your_model"
```

然后运行：

```bash
python train.py --config configs/my_model.yaml
```

### 关键参数说明

#### loss_decay_gamma 与 block_size 对应关系

| block_size | loss_decay_gamma | 适用模型 |
|-----------|-----------------|---------|
| 8         | 4.0             | 小模型/快速实验 |
| 10        | 5.0             | Llama 系列 |
| 16        | 7.0             | Qwen3 系列 (默认) |

#### target_layer_ids 自动计算

如果设置为 `null`，系统会根据论文算法自动计算：

```python
# 5层草稿模型映射到28层目标模型
target_layer_ids = [1, 6, 11, 17, 22]  # 均匀采样，跳过首尾层
```

也可以手动指定：

```yaml
model:
  target_layer_ids: [2, 8, 14, 20, 26]  # 自定义提取层
```

---

## 训练流程

### 1. 数据准备

训练数据应为 prompt-response 对。推荐的数据集：

- **ShareGPT** (`lmsys/sharegpt_jsonl`) - 多轮对话数据
- **UltraChat** (`HuggingFaceH4/ultrachat_200k`) - 高质量指令数据
- **CodeAlpaca** - 代码生成数据
- **自定义数据** - JSONL 格式，每条记录包含 `prompt` 和 `response` 字段

数据将被预处理：
1. 应用 chat template 编码
2. 用目标模型重新生成回复（对齐分布）
3. Tokenize 到最大序列长度
4. 构造 anchor 点和 mask block
5. 生成稀疏注意力掩码

### 2. 启动 vLLM 服务

对于**在线训练模式**，需要先启动 vLLM 服务以实时提取目标模型的隐藏状态：

```bash
# Terminal 1: 启动 vLLM 服务（使用 2 个 GPU）
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --tensor-parallel-size 1 \
  --data-parallel-size 2 \
  --port 8000 \
  --gpu-memory-utilization 0.9 \
  --enable-hidden-states \
  --hidden-states-layer-ids 1 6 11 17 22
```

> 注意：`--hidden-states-layer-ids` 必须与配置文件中的 `target_layer_ids` 一致。

### 3. 训练草稿模型

```bash
# Terminal 2: 启动训练（使用剩余 2 个 GPU）
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 \
  train.py --config configs/qwen3_8b.yaml
```

训练过程将自动：
- 从 vLLM 服务获取目标模型的隐藏状态
- 构造 anchor 位置和 mask block
- 通过稀疏注意力掩码进行块扩散训练
- 使用位置加权损失（早期 token 权重更高）
- 每轮保存检查点
- 记录训练日志

### 训练时间参考

| 目标模型 | GPU | 数据量 | 训练时间 |
|---------|-----|--------|---------|
| Qwen3-4B | 2x H100 | 50K 样本 | ~20 分钟 |
| Qwen3-8B | 4x H100 | 80K 样本 | ~25 分钟 |
| Llama-3.1-8B | 4x H100 | 80K 样本 | ~30 分钟 |

### 训练监控

```bash
# 查看训练日志
tail -f checkpoints/dflash_qwen3_8b/logs/dflash_rank0.log

# TensorBoard (可选)
tensorboard --logdir checkpoints/dflash_qwen3_8b/logs
```

---

## 推理

### 单条推理

```bash
python inference_cli.py \
  --config configs/qwen3_8b.yaml \
  --prompt "Write a Python function to implement quicksort"
```

输出示例：
```
============================================================
Response:
============================================================
Here is a Python implementation of quicksort:

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
============================================================

Generation Stats:
  Tokens generated : 89
  Tokens accepted  : 72
  Acceptance (tau) : 6.54
  Draft calls      : 14
  Target calls     : 14
  Time             : 0.82s
  Throughput       : 108.5 tok/s
```

### 交互式对话

```bash
python inference_cli.py --config configs/qwen3_8b.yaml --interactive
```

```
============================================================
  DFlash Interactive Inference
  Type 'quit' or 'exit' to leave
============================================================

User> What is the capital of France?

Assistant> The capital of France is Paris.

  [12 tokens in 0.15s = 80.0 tok/s, tau=6.20]

User> quit
Goodbye!
```

### 批量推理

准备 `prompts.jsonl`：

```jsonl
{"prompt": "What is 2+2?"}
{"prompt": "Explain gravity in simple terms"}
{"prompt": "Write a haiku about coding"}
```

运行：

```bash
python inference_cli.py \
  --config configs/qwen3_8b.yaml \
  --input prompts.jsonl \
  --output results.jsonl
```

---

## 评估

### 接受长度评估

测量草稿模型每次生成的平均接受 token 数（tau）：

```bash
python evaluate.py --config configs/qwen3_8b.yaml --acceptance-length --num-samples 128
```

输出：
```
Acceptance Length (tau):
  Mean  : 6.54
  Median: 6.72
  Range : 3.21 - 8.95
  Std   : 0.87
```

### 加速比评估

对比推测解码与自回归基线的速度：

```bash
python evaluate.py --config configs/qwen3_8b.yaml --speedup --num-samples 128
```

输出：
```
Results:
  Speculative   : 108.5 tok/s (3456 tokens in 31.8s)
  Autoregressive: 22.1 tok/s (3456 tokens in 156.4s)
  Speedup       : 4.91x
  Avg tau       : 6.54
```

### 下游任务评估

```bash
# 评估特定任务
python evaluate.py --config configs/qwen3_8b.yaml --tasks gsm8k math500

# 评估所有支持的任务
python evaluate.py --config configs/qwen3_8b.yaml --tasks all
```

支持的数据集：

| 数据集 | 任务类型 | 评估指标 |
|--------|---------|---------|
| `gsm8k` | 数学推理 | 准确率 |
| `math500` | 数学竞赛 | 准确率 |
| `humaneval` | 代码生成 | pass@1 |
| `mbpp` | Python编程 | 准确率 |
| `mt_bench` | 对话质量 | 评分 |

### 完整基准测试

```bash
python evaluate.py --config configs/qwen3_8b.yaml --benchmark
```

这将运行：
1. 不同温度下的接受长度评估 (temp=0, temp=1)
2. 加速比对比
3. 所有下游任务评估
4. 生成完整报告 JSON

---

## 模块架构

```
dflash-reproduce/
|
|-- configs/                        # 预置配置文件
|   |-- qwen3_8b.yaml
|   |-- qwen3_4b.yaml
|   |-- llama3_8b.yaml
|
|-- dflash_reproduce/               # 核心模块
|   |-- __init__.py                 # 包导出
|   |-- config.py                   # 配置系统 (YAML加载、验证、自动计算)
|   |-- model.py                    # DFlash模型架构
|   |   |-- DFlashDraftModel        # 5层块扩散草稿模型
|   |   |-- DFlashAttention         # KV注入注意力层
|   |   |-- TargetFeatureProjection # 目标特征投影
|   |   |-- build_draft_model()     # 模型构建工厂
|   |-- trainer.py                  # 训练循环
|   |   |-- DFlashTrainer           # 完整训练器
|   |   |-- compute_loss()          # 加权交叉熵损失
|   |   |-- setup_fsdp()            # FSDP分布式
|   |-- data.py                     # 数据准备
|   |   |-- load_dataset()          # 数据集加载
|   |   |-- regenerate_responses()  # 回复重新生成
|   |   |-- DFlashTrainingDataset   # 训练数据集
|   |   |-- build_sparse_attention_mask()  # 稀疏注意力掩码
|   |-- hidden_states.py            # 隐藏状态提取
|   |   |-- OnlineHiddenStatesExtractor  # vLLM在线提取
|   |   |-- OfflineHiddenStatesExtractor # 离线缓存提取
|   |   |-- HybridHiddenStatesExtractor  # 混合模式
|   |-- inference.py                # 推测解码推理
|   |   |-- DFlashInferenceEngine   # 推理引擎
|   |   |-- speculative_generate()  # 推测解码生成
|   |   |-- verify_block()          # 目标模型验证
|   |-- verify.py                   # 评估模块
|   |   |-- evaluate_speedup()      # 加速比评估
|   |   |-- evaluate_downstream()   # 下游任务评估
|   |   |-- run_benchmark()         # 完整基准测试
|   |-- utils.py                    # 工具函数
|       |-- setup_logging()         # 日志配置
|       |-- set_seed()              # 随机种子
|       |-- load_target_model()     # 加载目标模型
|       |-- Timer                   # 计时器
|
|-- train.py                        # 训练入口脚本
|-- inference_cli.py                # 推理入口脚本
|-- evaluate.py                     # 评估入口脚本
|-- README.md                       # 本文件
```

### 核心数据流

```
Prompt + Response
     |
     v
[Target Model Prefill]
     |
     +---> Extract Hidden States (指定层)
     |         |
     |         v
     |   [TargetFeatureProjection] --> KV Injection
     |         |
     v         v
[Bonus Token] + [Mask Tokens x (B-1)]
     |
     v
[DFlash Draft Model] (非因果注意力，单次前向)
     |
     v
[Draft Tokens Block] (B tokens)
     |
     v
[Target Model Verification] (并行验证)
     |
     +---> Find first mismatch
     |         |
     |         +---> Accept prefix tokens
     |         +---> Reject suffix tokens
     |         +---> Bonus token from target model
     v
[Repeat until EOS or max_length]
```

---

## 配置参考

### model 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_model` | str | `"Qwen/Qwen3-8B"` | 目标模型 (HuggingFace) |
| `target_model_type` | str | `"qwen3"` | 架构类型 |
| `draft_num_layers` | int | `5` | 草稿模型层数 |
| `draft_vocab_size` | int | `8192` | 草稿词表大小 |
| `block_size` | int | `16` | 扩散块大小 |
| `target_layer_ids` | list/null | `null` | 隐藏状态提取层 |
| `max_seq_len` | int | `3072` | 最大序列长度 |
| `dtype` | str | `"bfloat16"` | 训练精度 |

### training 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `epochs` | int | `6` | 训练轮数 |
| `batch_size` | int | `4` | 每GPU批次大小 |
| `learning_rate` | float | `6e-4` | 学习率 |
| `gradient_clipping` | float | `1.0` | 梯度裁剪 |
| `warmup_ratio` | float | `0.04` | Warmup比例 |
| `num_anchors` | int | `512` | 每序列锚点数 |
| `loss_decay_gamma` | float | `7.0` | 损失衰减参数 |
| `online_training` | bool | `true` | 在线训练模式 |

### hidden_states 配置

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `extraction_mode` | str | `"online"` | online/offline/hybrid |
| `vllm_endpoint` | str | `"http://localhost:8000/v1"` | vLLM服务端点 |

---

## 故障排除

### OOM (Out of Memory)

```yaml
# 减少批次大小
training:
  batch_size: 2

# 减少序列长度
model:
  max_seq_len: 2048

# 减少锚点数
training:
  num_anchors: 256
```

### vLLM 连接失败

```bash
# 检查 vLLM 服务是否运行
curl http://localhost:8000/v1/models

# 检查端口和endpoint配置
# 确保配置文件中的 vllm_endpoint 与启动的 vLLM 服务一致
```

### 隐藏状态提取失败

```bash
# 切换到离线模式
# 1. 先预提取隐藏状态
python -c "
from dflash_reproduce.hidden_states import OfflineHiddenStatesExtractor
extractor = OfflineHiddenStatesExtractor('./cache', 'Qwen/Qwen3-8B', [1,6,11,17,22])
# ... extract and cache ...
"

# 2. 修改配置为离线模式
# hidden_states:
#   extraction_mode: "offline"
```

### 梯度爆炸/NaN

```yaml
# 降低学习率
training:
  learning_rate: 3e-4

# 增加梯度裁剪
training:
  gradient_clipping: 0.5

# 使用float32训练
model:
  dtype: "float32"
```

---

## 引用

如果本项目对你的研究有帮助，请引用 DFlash 论文：

```bibtex
@article{chen2026dflash,
  title={DFlash: Block Diffusion for Flash Speculative Decoding},
  author={Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal={arXiv preprint arXiv:2602.06036},
  year={2026}
}
```

---

## 许可

本项目基于 MIT 许可证开源。
