# DFlash MacBook 复现指南 (48GB Unified Memory)

> 在 Apple Silicon MacBook 上完整复现 DFlash，从训练草稿模型到与官方模型对比 Benchmark。

---

## 目录

- [硬件要求](#硬件要求)
- [环境准备](#环境准备)
- [快速体验：官方模型 Benchmark](#快速体验)
- [完整训练流程](#完整训练流程)
- [与官方模型对比](#与官方模型对比)
- [参数调整建议](#参数调整建议)
- [故障排除](#故障排除)

---

## 硬件要求

| 项目 | 最低配置 | 推荐配置 |
|------|---------|---------|
| **芯片** | Apple M1 Pro/Max | Apple M3 Max / M4 Max |
| **内存** | 36GB | 48GB+ |
| **存储** | 50GB 可用空间 | 100GB+ SSD |
| **系统** | macOS 14 Sonoma | macOS 15 Sequoia |

> **48GB 内存分配估算**：
> - 目标模型 (Qwen3-4B, 4-bit): ~4GB
> - 草稿模型: ~0.2GB
> - 隐藏状态缓存: ~5-10GB
> - 训练过程: ~5-10GB
> - 系统预留: ~4GB
> - **可用余量**: ~20GB+

---

## 环境准备

### 1. 安装依赖

```bash
# 进入项目目录
cd dflash-reproduce

# 创建虚拟环境
python3 -m venv venv_mac
source venv_mac/bin/activate

# 安装核心依赖 (MLX 优先)
pip install --upgrade pip
pip install mlx mlx-lm

# 安装通用依赖
pip install torch transformers accelerate datasets
pip install huggingface_hub pyyaml numpy tqdm

# 验证安装
python3 -c "import mlx.core as mx; print(f'MLX: {mx.__version__}')"
python3 -c "import mlx_lm; print('mlx-lm: OK')"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, MPS: {torch.backends.mps.is_available()}')"
```

### 2. 登录 HuggingFace

```bash
# 安装 CLI
pip install huggingface_hub

# 登录 (需要 token，从 https://huggingface.co/settings/tokens 获取)
huggingface-cli login
```

---

## 快速体验

### 运行官方 DFlash 模型 Benchmark

最简单的方式，不需要训练，直接对比官方发布的模型：

```bash
# 方式1: 一键运行脚本
chmod +x scripts/run_mac_benchmark.sh
./scripts/run_mac_benchmark.sh

# 方式2: 手动运行快速测试 (5条数据，1分钟)
python3 -m dflash_reproduce.mac_benchmark --quick

# 方式3: 手动运行完整测试 (50条数据，10-20分钟)
python3 -m dflash_reproduce.mac_benchmark --official --num-samples 50
```

**预期输出** (Qwen3-4B + z-lab/Qwen3-4B-DFlash-b16):

```
======================================================================
  DFlash Mac Benchmark Report
======================================================================

  Generation Speed
  ------------------------------------------------------------------
  Metric                              DFlash    AR Baseline
  ------------------------------    ---------    -------------
  Total tokens                            850            850
  Total time (s)                         12.5           45.2
  Tokens/sec                             68.0           18.8
  Speedup                                3.62x
  Avg acceptance (tau)                   5.83

  GSM8K Evaluation
  ------------------------------------------------------------------
  Model : official
  Samples: 50
  Correct: 42/50
  Accuracy: 84.0%
```

---

## 完整训练流程

如果你想**从头训练**自己的 DFlash 草稿模型，然后与官方模型对比：

### 阶段 1: 提取隐藏状态 (离线缓存)

这是最耗时的步骤（约 30-60 分钟），需要加载完整的 Qwen3-4B 模型：

```bash
# 仅提取隐藏状态
python3 dflash_reproduce/mac_train.py \
    --config configs/mac_qwen3_4b.yaml \
    --extract-only
```

**内存管理技巧**：
- 这个过程会加载 Qwen3-4B (~4GB) + 处理数据
- 如果内存不足，减小 `max_samples`:
  ```yaml
  hidden_states:
    max_samples: 2000  # 从 5000 减少到 2000
  ```

### 阶段 2: 训练草稿模型

隐藏状态缓存后，可以卸载目标模型，只用 ~2GB 内存训练草稿模型：

```bash
# 仅训练 (使用已缓存的隐藏状态)
python3 dflash_reproduce/mac_train.py \
    --config configs/mac_qwen3_4b.yaml \
    --train-only
```

**训练时间参考**：

| 数据量 | M3 Max | M4 Max |
|--------|--------|--------|
| 1,000 条 | ~15 分钟 | ~10 分钟 |
| 5,000 条 | ~60 分钟 | ~40 分钟 |

### 阶段 3: 对比 Benchmark

训练完成后，与官方模型进行 head-to-head 对比：

```bash
# 对比两种模型
python3 -m dflash_reproduce.mac_benchmark \
    --all \
    --draft-path ./mac_checkpoints/dflash_qwen3_4b/final \
    --num-samples 50
```

**预期对比输出**：

```
======================================================================
  HEAD-TO-HEAD COMPARISON
======================================================================
  Model                         Tok/s        Tau    Accuracy
  -------------------------    ---------    -----    --------
  Official DFlash                 68.0      5.83      84.0%
  Custom DFlash (yours)           65.2      5.41      81.5%
```

> 自训练模型的性能通常接近官方模型（~90-95%），差异主要来自训练数据量和超参数调优。

---

## 与官方模型对比

### 对比维度

| 指标 | 说明 | 预期值 |
|------|------|--------|
| **Acceptance (τ)** | 平均每次接受多少 token | 5.0 - 7.0 |
| **Tokens/sec** | 生成速度 | 50 - 80 tok/s |
| **Speedup** | 相比自回归的加速比 | 3.0x - 4.5x |
| **GSM8K Acc** | 数学推理准确率 | 75% - 85% |

### 单次推理测试

```python
from dflash_reproduce.mac_mlx_core import load_models, stream_generate_dflash

# 加载官方模型
model, tokenizer, draft = load_models(
    target_id="Qwen/Qwen3-4B",
    draft_id="z-lab/Qwen3-4B-DFlash-b16",
)

# 生成
prompt = "What is the sum of all integers from 1 to 100?"
for resp in stream_generate_dflash(model, draft, tokenizer, prompt, max_tokens=256):
    print(resp.text, end="", flush=True)

print(f"\n\nStats: {resp.generation_tokens} tokens, "
      f"{resp.generation_tps:.1f} tok/s, "
      f"tau={resp.accepted:.2f}")
```

---

## 参数调整建议

### 如果内存不足 (36GB)

```yaml
# configs/mac_qwen3_4b.yaml
model:
  max_seq_len: 1024        # 从 2048 减少到 1024
  quantization: "4bit"

training:
  batch_size: 1
  gradient_accumulation: 4  # 减少有效 batch size
  num_anchors: 64           # 从 128 减少到 64

data:
  max_samples: 2000         # 减少训练数据

hidden_states:
  max_samples: 2000
  batch_size: 1             # 减小提取批次
```

### 如果要更快训练

```yaml
training:
  epochs: 2                 # 从 3 减少到 2
  gradient_accumulation: 4  # 更多梯度累积步

data:
  max_samples: 1000         # 小数据集快速实验
```

### 如果要更高质量

```yaml
training:
  epochs: 6                 # 更多训练轮数
  learning_rate: 0.0003     # 更低学习率

data:
  max_samples: 10000        # 更多数据

hidden_states:
  max_samples: 10000
```

---

## 故障排除

### `mlx` 或 `mlx-lm` 安装失败

```bash
# 确保 macOS 版本 >= 14
sw_vers -productVersion  # 应显示 14.x 或更高

# 如果 pip 安装失败，尝试编译安装
pip install mlx --no-binary mlx
```

### MPS 内存不足

```bash
# 设置内存限制
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7

# 在 Python 中清理缓存
import torch
torch.mps.empty_cache()
```

### 隐藏状态提取太慢

- 减少 `max_samples` 到 1000 或更少
- 关闭其他应用释放内存
- 使用更小的目标模型（如 Qwen3-1.7B）

### 训练 loss 不下降

- 检查隐藏状态是否正确缓存
- 降低学习率到 `3e-4`
- 增加 `gradient_accumulation` 到 16

### 模型加载失败

```bash
# 清除 HuggingFace 缓存重新下载
rm -rf ~/.cache/huggingface/hub/models--z-lab--Qwen3-4B-DFlash-b16

# 或者手动下载
huggingface-cli download z-lab/Qwen3-4B-DFlash-b16
```

---

## 文件说明

| 文件 | 用途 |
|------|------|
| `configs/mac_qwen3_4b.yaml` | Mac 专用配置文件 |
| `dflash_reproduce/mac_mlx_core.py` | MLX 推理核心 |
| `dflash_reproduce/mac_benchmark.py` | Benchmark 对比脚本 |
| `dflash_reproduce/mac_train.py` | Mac 训练脚本 |
| `scripts/run_mac_benchmark.sh` | 一键运行脚本 |

---

## 下一步

1. **调整超参数**: 根据 benchmark 结果微调训练参数
2. **更换目标模型**: 修改 `target_model` 为其他 Qwen3/Llama 模型
3. **增大 block_size**: 尝试 block_size=16 获得更高加速比
4. **使用更多数据**: 增大 max_samples 提升模型质量

---

## 引用

```bibtex
@article{chen2026dflash,
  title={DFlash: Block Diffusion for Flash Speculative Decoding},
  author={Chen, Jian and Liang, Yesheng and Liu, Zhijian},
  journal={arXiv preprint arXiv:2602.06036},
  year={2026}
}
```
