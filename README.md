# ResNet 训练框架使用指南

本项目基于 [PyTorch Vision](https://github.com/pytorch/vision) 项目修改和扩展。原始代码版权归PyTorch贡献者所有，遵循BSD 3-Clause许可证。](https://github.com/pytorch/vision)

## 1. 训练环境搭建

### 1.1 环境要求

- **操作系统**: Linux
- **Python**: >= 3.8
- **CUDA**: 10.2 或更高版本（使用GPU时）

### 1.2 依赖安装

#### 1.2.1 安装 PyTorch

根据您的 CUDA 版本选择合适的 PyTorch 安装命令：

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU 版本（不推荐用于训练）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 1.2.2 安装其他依赖

```bash
pip install tensorboard matplotlib tqdm
```

#### 1.2.3 验证安装

```bash
# 运行环境测试脚本
cd lx_Interface
python test_imports.py
```

## 2. 模型训练启动流程

### 2.1 基础命令

```bash
# 进入训练目录
cd lx_Interface

# 基本训练命令（默认 CIFAR-10 数据集）
python resnet_trainer.py --model resnet18 --epochs 10

# 完整训练命令
python resnet_trainer.py \
    --model resnet18 \
    --dataset cifar10 \
    --data-path ./data \
    --output-dir ./output \
    --batch-size 128 \
    --epochs 100 \
    --lr 0.1 \
    --opt sgd
```

### 2.2 路径配置说明

#### 2.2.1 绝对路径输出

训练过程中，程序会输出以下绝对路径信息：

| 输出类型              | 格式                                                                | 说明                                                   |
| ----------------- | ----------------------------------------------------------------- | ---------------------------------------------------- |
| TensorBoard日志路径   | `TensorBoard logs will be saved to: {绝对路径}`                       | 训练日志保存位置                                             |
| 模型保存路径            | `Final model saved to: {绝对路径}`                                    | 最终模型文件保存位置                                           |
| 检查点保存路径           | `Model checkpoint saved to: {绝对路径}`                               | 中间检查点保存位置                                            |
| MANAGER_PATH_INFO | `MANAGER_PATH_INFO: TensorBoard={绝对路径}`                           | 供实验管理器解析的TensorBoard路径                               |
| MANAGER_PATH_INFO | `MANAGER_PATH_INFO: Model={绝对路径}`                                 | 供实验管理器解析的模型路径                                        |
| 实验管理器日志路径         | `MANAGER_LOG_DIR = "/home/dell/Documents/lx/vision/manager_log/"` | 批量试验管理实验记录的输出路径，默认值位于 train_experiment_manager.py:16 |

#### 2.2.2 默认路径设置

程序默认使用以下绝对路径：

| 参数             | 默认值                                      | 说明              |
| -------------- | ---------------------------------------- | --------------- |
| `--data-path`  | `/home/dell/Documents/lx/vision/data/`   | 数据集保存目录         |
| `--log-dir`    | `/home/dell/Documents/lx/vision/log/`    | TensorBoard日志目录 |
| `--output-dir` | `/home/dell/Documents/lx/vision/output/` | 模型输出目录          |

建议在启动训练时，使用 `--data-path`、`--log-dir` 和 `--output-dir` 参数指定适合您环境的路径。

### 2.3 快速开始

#### 2.3.1 使用单GPU训练

```bash
python resnet_trainer.py --model resnet18 --dataset fashionmnist --epochs 50 --batch-size 128 --lr 0.01
```

#### 2.3.2 使用双GPU分布式训练

```bash
torchrun --nproc_per_node=2 resnet_trainer.py --model resnet18 --dataset cifar10 --epochs 100 --batch-size 256 --amp --model-ema
```

### 2.4 模型测试

```bash
# 使用保存的检查点评估模型
python resnet_trainer.py --test-only --resume ./output/checkpoint.pth --model resnet18 --dataset cifar10
```

## 3. 参数调优指南

### 3.1 关键超参数说明

| 参数               | 默认值      | 说明     | 调优建议                                                      |
| ---------------- | -------- | ------ | --------------------------------------------------------- |
| `--model`        | resnet18 | 模型名称   | 根据数据集复杂度选择，CIFAR-10 推荐 resnet18/34，CIFAR-100 推荐 resnet50+ |
| `--dataset`      | cifar10  | 数据集    | cifar10, cifar100, fashionmnist                           |
| `--batch-size`   | 32       | 批大小    | GPU 显存允许情况下越大越好，推荐 128-512                                |
| `--epochs`       | 90       | 训练轮数   | CIFAR-10: 100-200，CIFAR-100: 200-300，FashionMNIST: 50-100 |
| `--lr`           | 0.1      | 学习率    | SGD: 0.01-0.1，AdamW: 0.0001-0.001                         |
| `--opt`          | sgd      | 优化器    | SGD 适合长期训练，AdamW 收敛更快                                     |
| `--lr-scheduler` | steplr   | 学习率调度器 | 建议使用 cosineannealinglr                                    |
| `--weight-decay` | 1e-4     | 权重衰减   | 防止过拟合，推荐范围 1e-5 到 1e-3                                    |

### 3.2 调优策略

1. **学习率调整**:
   
   - 对于 CIFAR 数据集，建议从 `lr=0.1` 开始
   - 对于 Fashion MNIST，建议从 `lr=0.01` 开始
   - 使用 `--lr-scheduler cosineannealinglr` 获得更好的收敛效果

2. **优化器选择**:
   
   - SGD 通常效果最好，但需要调整学习率和动量
   - AdamW 更容易收敛，适合快速实验
   - 建议先尝试 SGD，然后用 AdamW 验证结果

3. **正则化策略**:
   
   - 使用 `--model-ema` 提高模型泛化能力
   - 使用 `--amp` 启用混合精度训练，加速训练并减少显存占用
   - 考虑使用 `--save-only-final` 减少磁盘空间占用

### 3.3 最佳实践

- **先跑短实验**：先用少量 epoch（10-20）验证配置，再运行完整训练
- **分布式优先**：双GPU训练可以显著提高速度
- **使用自动混合精度**：添加 `--amp` 参数可以加速训练 30-50%
- **监控验证准确率**：如果与训练准确率差距过大，考虑增加正则化
- **学习率预热**：对于大模型，考虑使用学习率预热

## 4. 批量训练功能

### 4.1 功能概述

`train_experiment_manager.py` 用于无人值守批量执行多个训练任务，支持不同的参数组合和GPU配置。

### 4.2 配置文件编写

编辑 `train_experiment_manager.py` 的 `main()` 函数，添加您的实验配置：

```python
def main():
    manager = TrainingExperimentManager()

    # 基础参数
    base_params = {
        "data_path": "/home/dell/Documents/lx/vision/data/",
        "model": "resnet18",
        "dataset": "fashionmnist",
        "lr": 0.1,
        "epochs": 60,
        "batch_size": 256,
        "weight_decay": 0.0001
    }

    # 添加实验
    manager.add_experiment(
        name="双GPU_SGD_lr0.1_epochs60_batch256",
        params=base_params,
        use_distributed=True  # 双GPU分布式训练
    )

    manager.add_experiment(
        name="单GPU_AdamW_lr0.001_epochs30_batch128",
        params={**base_params, "opt": "adamw", "lr": 0.001, "epochs": 30, "batch_size": 128},
        use_distributed=False,
        gpu_id=0  # 使用GPU 0
    )

    # 运行所有实验
    manager.run_all_experiments()
    manager.generate_comparison_report()
```

### 4.3 批量任务提交

```bash
# 运行批量训练
cd lx_Interface
python train_experiment_manager.py
```

### 4.4 任务管理

- 训练过程中，实时显示每个epoch的进度、loss和accuracy
- 支持按 Ctrl+C 中断当前实验并继续下一个实验
- 快速按 Ctrl+C 两次停止所有实验

### 4.5 结果分析

训练完成后，自动生成以下文件：

- **实验日志**: `train_experiment.log` - 包含所有实验的详细记录
- **结果汇总**: `experiment_results_YYYYMMDD_HHMMSS.txt` - 记录每个实验的状态和耗时
- **对比报告**: `experiment_comparison_YYYYMMDD_HHMMSS.md` - Markdown格式的实验对比表格

## 5. TensorBoard 可视化

### 5.1 启动步骤

#### 5.1.1 使用脚本启动（推荐）

```bash
cd lx_Interface
python view_tensorboard.py
```

#### 5.1.2 手动启动

```bash
# 查看所有训练日志
tensorboard --logdir=log/

# 查看特定实验日志
tensorboard --logdir=log/resnet18_cifar10_sgd_lr01_bs128_singlegpu_100_20251022_143052/
```

### 5.2 端口配置

```bash
# 指定端口
tensorboard --logdir=log/ --port=6007

# 允许外网访问（注意安全性）
tensorboard --logdir=log/ --host=0.0.0.0 --port=6006
```

### 5.3 关键指标监控

在浏览器中访问 `http://localhost:6006` 查看训练指标：

#### 5.3.1 SCALARS 标签页

| 指标                  | 说明        | 监控重点                        |
| ------------------- | --------- | --------------------------- |
| `Loss/train`        | 训练损失      | 应随训练逐渐下降，若震荡剧烈考虑降低学习率       |
| `Accuracy/train`    | 训练准确率     | 应随训练逐渐上升，最终应达到较高水平          |
| `Accuracy/test`     | 测试准确率     | 模型泛化能力的关键指标，与训练准确率差距过大可能过拟合 |
| `Learning_rate`     | 学习率变化     | 检查学习率调度是否按预期工作              |
| `Accuracy/test_ema` | EMA 模型准确率 | 如果启用了 EMA，查看其性能表现           |

#### 5.3.2 TEXT 标签页

查看 `Training/completion_time` 可以看到：

- 训练完成时间
- 总训练时长
- 模型配置信息

#### 5.3.3 IMAGES 标签页

查看模型生成的混淆矩阵：

- **`Confusion_Matrix/test`**: 测试集上的混淆矩阵热力图
- **`Confusion_Matrix/test_ema`**: EMA模型测试集上的混淆矩阵热力图（如果启用了EMA）

**混淆矩阵解读**：
- 对角线元素表示正确分类的样本数量
- 非对角线元素表示分类错误的样本数量
- 颜色越深表示该单元格数值越大
- 可直观查看哪些类别容易混淆

### 5.3.4 快速启动 TensorBoard

使用项目提供的脚本快速启动 TensorBoard：

```bash
# 自动查找所有日志目录
python lx_Interface/view_tensorboard.py

# 指定特定日志目录
python lx_Interface/view_tensorboard.py --logdir path/to/specific/log

# 指定端口和主机
python lx_Interface/view_tensorboard.py --port 6007 --host 0.0.0.0
```

### 5.4 对比不同实验

然后在浏览器中：

1. 所有实验的曲线会同时显示
2. 可以通过左侧的复选框选择要显示的实验
3. 可以通过颜色区分不同的实验

## 6. 高级功能

### 6.1 学习率策略

本训练框架支持多种学习率策略，其中 `deep_warmup` 是一种特殊的学习率策略，提供了深度预热机制：

```bash
# 使用 deep_warmup 策略
python resnet_trainer.py --lr-strategy deep_warmup [其他参数...]
```

**deep_warmup 策略特点**:

- 初始使用较低的预热学习率
- 根据训练进度和损失自动调整学习率
- 适合需要稳定初始训练阶段的场景

### 6.2 模型保存策略

- **仅保存最终模型**: `--save-only-final` - 节省磁盘空间，适合长时间训练
- **按间隔保存**: `--save-interval 5` - 每隔5个epoch保存一次模型

### 6.3 性能优化

- **增大 `--batch-size`**: 提升 GPU 利用率
- **增加 `--workers`**: 提升数据加载速度
- **使用 `--amp`**: 启用混合精度训练，加速训练并减少显存占用
- **使用 `--model-ema`**: 提高模型泛化能力

## 7. 故障排查

### 7.1 CUDA 内存不足

```bash
# 减小批大小
python resnet_trainer.py --batch-size 64

# 启用混合精度
python resnet_trainer.py --amp
```

### 7.2 TensorBoard 不显示数据

- 检查日志目录是否正确
- 刷新浏览器页面
- 重启 TensorBoard

### 7.3 导入错误

```bash
# 运行测试脚本
python test_imports.py

# 如果失败，重新安装依赖
pip install torch torchvision tensorboard
```

### 7.4 训练损失不下降

- 检查学习率是否过大或过小
- 检查数据集是否正确加载
- 检查模型是否正确初始化

## 8. 文件结构

```
lx_Interface/
├── resnet_trainer.py            # 主训练脚本
├── train_experiment_manager.py   # 批量训练管理器
├── view_tensorboard.py          # TensorBoard 启动脚本
├── test_imports.py              # 环境测试脚本
├── example_train.sh             # Linux/Mac 示例
├── example_train.bat            # Windows 示例
├── README.md                    # 本文件
└── readme/                      # 原始文档目录
    ├── CHANGELOG.md             # 修改历史
    └── 其他原始文档...

log/                              # 存储TensorBoard 日志
└── resnet{层数}_{数据集}_{优化器}_lr{学习率}_bs{batch_size}_{single/double}gpu_{epochs}_{时间戳}/

output/                           # 存储模型输出
└── resnet{层数}_{数据集}_{优化器}_lr{学习率}_bs{batch_size}_{single/double}gpu_{epochs}_{时间戳}.pth
```