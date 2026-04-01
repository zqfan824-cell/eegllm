# EEGLLM: 基于LLM重编程的EEG情绪分类框架

## 项目简介

本项目基于 [Time-LLM](https://github.com/KimMeen/Time-LLM)（ICLR 2024）的重编程框架思路，将大型语言模型（LLM）应用于**脑电信号（EEG）的情绪分类任务**。

原始 Time-LLM 是一个时间序列预测框架。本项目将其核心思路迁移到 EEG 领域，**将预测任务改为分类任务**，并加入 VQ 量化、域对抗学习等增强模块，命名为 **EEGLLM**。

### 核心思路

```
EEG信号 → Patch分割 → [VQ量化] → 重编程到LLM词空间 → 冻结LLM处理 → 分类头 → 情绪类别
```

### 支持的数据集

| 数据集 | 通道数 | 采样率 | 情绪类别 | 任务 |
|--------|--------|--------|----------|------|
| **DEAP** | 32 | 128 Hz | 2类（正面/负面）或 4类 | valence/arousal 分类 |
| **SEED** | 62 | 200 Hz | 3类（正面/中性/负面） | 情绪三分类 |

---

## 项目结构

```
eegllm/
│
├── run_main.py                    # 主入口脚本（训练+测试）
├── run_deap_vq.sh                 # DEAP数据集一键训练脚本
├── requirements.txt               # Python依赖
├── Time-LLM.pdf                   # 原始论文
├── EEGLLM_VQ改进.md               # 改进说明文档
│
├── models/                        # 模型定义
│   ├── EEGLLM.py                  # 基础EEGLLM模型（支持分类+预测）
│   └── EEGLLM_VQ.py               # 增强版：加入VQ量化+重建损失+域对抗学习
│
├── exp/                           # 实验管理
│   └── exp_classification_vq.py   # 分类实验类（训练/验证/测试流程）
│
├── data_provider/                 # 数据加载
│   ├── data_factory.py            # 统一数据工厂接口
│   └── data_loader_eeg.py         # DEAP和SEED数据集加载器 + 通道选择
│
├── layers/                        # 网络层组件
│   ├── Embed.py                   # Patch嵌入、位置编码等
│   └── StandardNorm.py            # 可逆标准化层（RevIN）
│
├── utils/                         # 工具函数
│   ├── tools.py                   # EarlyStopping、学习率调整等通用工具
│   ├── loss_classification.py     # 分类损失函数（CrossEntropy、FocalLoss等）
│   ├── metrics_classification.py  # 分类评估指标（Accuracy、F1、混淆矩阵等）
│   ├── reconstruction_losses.py   # VQ量化器 + 重建损失 + 自适应权重
│   ├── tools_classification.py    # 训练可视化工具
│   └── train_utils_classification.py  # 学习率调度、梯度裁剪等训练辅助
│
├── dataset/prompt_bank/           # LLM提示词模板
│   ├── DEAP.txt                   # DEAP数据集提示词
│   └── SEED.txt                   # SEED数据集提示词
│
├── figures/                       # 框架图
├── checkpoints/                   # 模型检查点（训练时自动生成）
└── test_results/                  # 测试结果（测试时自动生成）
```

---

## 模型架构

### 1. 基础模型：EEGLLM（`models/EEGLLM.py`）

1. **输入标准化**：对EEG信号做RevIN标准化
2. **Patch嵌入**：将EEG时间序列切分为固定大小的patch，映射到d_model维空间
3. **Prompt生成**：基于EEG信号的统计信息（最值、中位数、趋势等）生成自然语言提示
4. **重编程层**：通过交叉注意力将EEG patch嵌入映射到LLM词空间
5. **冻结LLM**：使用冻结的LLM（支持LLaMA和GPT-2）处理重编程后的特征
6. **分类头**：对LLM输出做平均池化后通过MLP分类

### 2. 增强模型：EEGLLM_VQ（`models/EEGLLM_VQ.py`）

在基础模型上增加了三个模块（参考NeuroLM）：

- **Vector Quantization（VQ）**：在patch嵌入后加入向量量化，将连续EEG特征离散化，降低噪声
- **重建损失**：VQ量化后的特征需要能重建原始信号的频域和时域表示，防止信息丢失
- **域对抗学习**：通过梯度反转+域分类器，让重编程后的EEG特征与LLM词空间对齐

### 数据流（EEGLLM_VQ）

```
EEG (B, seq_len, N_channels)
    ↓ RevIN标准化
    ↓ Patch分割 → (B*N, patches_per_var, d_model)
    ↓ VQ编码 → 量化 → 映射回d_model
    ↓ 重编程（交叉注意力 + 域对抗）→ (B*N, patches, d_llm)
    ↓ 拼接prompt嵌入
    ↓ 冻结LLM处理
    ↓ 截取后patches_per_var个token → reshape → (B, N, d_ff, patch_nums)
    ↓ ClassificationHead → (B, num_class)
```

---

## 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

主要依赖：PyTorch, Transformers (HuggingFace), accelerate, scipy, scikit-learn

### 数据准备

**DEAP数据集**：
1. 从 [DEAP官网](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/) 下载预处理数据
2. 将 `data_preprocessed_python/` 目录（包含 s01.dat ~ s32.dat）放到指定路径

**SEED数据集**：
1. 从 [SEED官网](https://bcmi.sjtu.edu.cn/~seed/) 下载数据
2. 将 .mat 文件放到指定路径

### 训练（DEAP示例）

```bash
# 使用脚本（推荐）
bash run_deap_vq.sh

# 或手动运行
python run_main.py \
    --task_name classification \
    --is_training 1 \
    --model_id DEAP_EEGLLM_VQ \
    --model EEGLLM_VQ \
    --data DEAP \
    --root_path /path/to/data_preprocessed_python/ \
    --seq_len 256 \
    --n_class 2 \
    --classification_type valence \
    --llm_model GPT2 \
    --llm_dim 768 \
    --llm_layers 2 \
    --d_model 32 \
    --d_ff 128 \
    --batch_size 8 \
    --train_epochs 20 \
    --learning_rate 0.0001 \
    --enable_vq \
    --enable_adversarial
```

### 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data` | DEAP | 数据集：DEAP 或 SEED |
| `--n_class` | 2 | 分类数（DEAP: 2或4, SEED: 3） |
| `--classification_type` | valence | DEAP二分类类型：valence 或 arousal |
| `--llm_model` | LLAMA | LLM骨干：LLAMA 或 GPT2 |
| `--llm_layers` | 32 | 使用LLM的前几层（减小可节省显存） |
| `--seq_len` | 256 | EEG输入序列长度（采样点数） |
| `--enable_vq` | True | 启用VQ量化 |
| `--enable_reconstruction` | False | 启用重建损失 |
| `--enable_adversarial` | True | 启用域对抗学习 |
| `--channel_selection` | comprehensive_emotion | 通道选择策略 |

### 通道选择策略

针对DEAP数据集，提供了多种基于神经科学文献的通道选择策略：

| 策略名 | 通道数 | 适用场景 |
|--------|--------|----------|
| `comprehensive_emotion` | 14 | 通用情绪分析（默认） |
| `valence_specific` | 8 | valence二分类 |
| `arousal_specific` | 8 | arousal二分类 |
| `frontal_emotion` | 9 | 前额叶情绪区域 |
| `frontal_asymmetry` | 6 | 前额叶不对称性分析 |

设置 `--use_channel_selection False` 可使用全部32/62个通道。

---

## 参考文献

- Jin et al., "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models", ICLR 2024
- Koelstra et al., "DEAP: A Database for Emotion Analysis using Physiological Signals", IEEE TAC 2012
- Zheng & Lu, "Investigating Critical Frequency Bands and Channels for EEG-Based Emotion Recognition", IEEE TAC 2015
