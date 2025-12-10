import os
import sys
import argparse
import torch
import torchvision
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch.distributed as dist

# 添加父目录到路径，以便导入from_vision中的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入train.py中的主要功能
from from_vision.references.classification import train
from from_vision.references.classification import presets
from from_vision.references.classification import utils
from torchvision.transforms.functional import InterpolationMode


def evaluate_with_loss(model, criterion, data_loader, device, print_freq=100, log_suffix=""):
    """
    评估模型并返回accuracy和loss
    
    Args:
        model: 要评估的模型
        criterion: 损失函数
        data_loader: 数据加载器
        device: 设备
        print_freq: 打印频率
        log_suffix: 日志后缀
    
    Returns:
        tuple: (accuracy, loss)
    """
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(data_loader, print_freq, header):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    # 同步所有进程的指标
    metric_logger.synchronize_between_processes()

    print(f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} Loss {metric_logger.loss.global_avg:.6f}")
    return metric_logger.acc1.global_avg, metric_logger.loss.global_avg


def load_data(args):
    """
    加载数据集并创建数据加载器，支持CIFAR10、CIFAR100和FashionMNIST
    """
    print(f"Loading {args.dataset} dataset", flush=True)
    
    # 设置数据路径
    if not args.data_path:
        # 默认数据路径
        args.data_path = os.path.join(os.path.expanduser("~"), ".torch", "datasets")
    
    # 确保data_path是绝对路径，并规范化路径
    args.data_path = os.path.abspath(os.path.expanduser(args.data_path))
    
    # 验证路径是否存在，如果不存在则尝试创建（只创建最后一级目录，不创建整个路径树）
    if not os.path.exists(args.data_path):
        try:
            os.makedirs(args.data_path, exist_ok=True)
            print(f"Created data directory: {args.data_path}", flush=True)
        except (OSError, PermissionError) as e:
            print(f"Warning: Cannot create data directory {args.data_path}: {e}", flush=True)
            print(f"Will attempt to use the directory anyway.", flush=True)
    
    # 对于FashionMNIST，图像大小为28x28，需要调整转换参数
    if args.dataset == "fashionmnist":
        # FashionMNIST是灰度图像，需要特殊处理
        # 方案：将灰度图转换为3通道（重复通道），以适配RGB预训练模型
        import torchvision.transforms as transforms
        
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # 将灰度图转为3通道
            transforms.Resize(32),  # 稍微放大以便后续裁剪
            transforms.RandomCrop(28),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),  # 随机旋转±10度
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 调整亮度和对比度
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 灰度图归一化
        ])
        
        test_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),  # 将灰度图转为3通道
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        # 对于CIFAR数据集使用原始参数
        train_crop_size = args.train_crop_size
        val_crop_size = args.val_crop_size
        val_resize_size = args.val_resize_size
        
        # 定义转换
        train_transform = presets.ClassificationPresetTrain(
            crop_size=train_crop_size,
            interpolation=InterpolationMode(args.interpolation),
            auto_augment_policy=args.auto_augment,
            random_erase_prob=args.random_erase,
            backend=args.backend,
            use_v2=args.use_v2,
        )
        
        test_transform = presets.ClassificationPresetEval(
            crop_size=val_crop_size,
            resize_size=val_resize_size,
            interpolation=InterpolationMode(args.interpolation),
            backend=args.backend,
            use_v2=args.use_v2,
        )
    
    # 在分布式训练中，只让主进程下载数据，其他进程等待
    is_main_process = not args.distributed or utils.is_main_process()
    
    # 加载数据集
    if args.dataset == "fashionmnist":
        # Fashion MNIST数据集
        # 主进程先下载
        if is_main_process:
            print("Main process downloading dataset...", flush=True)
            torchvision.datasets.FashionMNIST(root=args.data_path, train=True, download=True)
            torchvision.datasets.FashionMNIST(root=args.data_path, train=False, download=True)
        
        # 等待主进程下载完成
        if args.distributed:
            torch.distributed.barrier()
        
        print(f"Rank {args.rank if args.distributed else 0}: Loading FashionMNIST dataset", flush=True)
        train_dataset = torchvision.datasets.FashionMNIST(
            root=args.data_path,
            train=True,
            transform=train_transform,
            download=False  # 不再下载，只加载
        )
        test_dataset = torchvision.datasets.FashionMNIST(
            root=args.data_path,
            train=False,
            transform=test_transform,
            download=False
        )
    elif args.dataset == "cifar10":
        # CIFAR-10数据集
        if is_main_process:
            print("Main process downloading dataset...", flush=True)
            torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True)
            torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True)
        
        if args.distributed:
            torch.distributed.barrier()
        
        print(f"Rank {args.rank if args.distributed else 0}: Loading CIFAR10 dataset", flush=True)
        train_dataset = torchvision.datasets.CIFAR10(
            root=args.data_path,
            train=True,
            transform=train_transform,
            download=False
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=args.data_path,
            train=False,
            transform=test_transform,
            download=False
        )
    elif args.dataset == "cifar100":
        # CIFAR-100数据集
        if is_main_process:
            print("Main process downloading dataset...", flush=True)
            torchvision.datasets.CIFAR100(root=args.data_path, train=True, download=True)
            torchvision.datasets.CIFAR100(root=args.data_path, train=False, download=True)
        
        if args.distributed:
            torch.distributed.barrier()
        
        print(f"Rank {args.rank if args.distributed else 0}: Loading CIFAR100 dataset", flush=True)
        train_dataset = torchvision.datasets.CIFAR100(
            root=args.data_path,
            train=True,
            transform=train_transform,
            download=False
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=args.data_path,
            train=False,
            transform=test_transform,
            download=False
        )
    
    # 创建数据加载器
    print(f"Rank {args.rank if args.distributed else 0}: Creating data loaders...", flush=True)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    
    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    
    print(f"Rank {args.rank if args.distributed else 0}: Data loaders created successfully", flush=True)
    return train_dataset, test_dataset, data_loader, data_loader_test


def create_tensorboard_writer(args):
    """
    创建TensorBoard的SummaryWriter并返回日志目录路径
    """
    # 获取当前时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 获取优化器名称
    opt_name = args.opt.lower()
    
    # 提取resnet层数（从model名称中提取数字部分）
    resnet_depth = ''.join(filter(str.isdigit, args.model))
    
    # 判断GPU配置：single或double
    gpu_config = "double" if args.distributed else "single"
    
    # 格式化学习率（去除小数点，例如0.1 -> lr01, 0.01 -> lr001）
    # 先格式化为固定小数位，再去除小数点
    lr_formatted = f"{args.lr:.6f}".rstrip('0').rstrip('.')
    lr_str = f"lr{lr_formatted.replace('.', '')}"
    
    # 统一采用"resnet层数_数据集_优化器_lr_batchsize_gpu配置_训练迭代次数_完成时间"的命名格式
    log_dir_name = f"resnet{resnet_depth}_{args.dataset}_{opt_name}_{lr_str}_bs{args.batch_size}_{gpu_config}gpu_{args.epochs}_{timestamp}"
    
    # 完整日志路径 - 使用用户指定的日志目录
    log_path = os.path.join(args.log_dir, log_dir_name)
    
    # 创建目录
    utils.mkdir(log_path)
    
    # 创建SummaryWriter
    writer = SummaryWriter(log_dir=log_path)
    
    print(f"TensorBoard logs will be saved to: {log_path}")
    print(f"To view, run: tensorboard --logdir={log_path}")
    # 输出特定格式的路径信息，方便manager解析
    print(f"MANAGER_PATH_INFO: TensorBoard={log_path}", flush=True)
    
    return writer, log_path


def compute_confusion_matrix(model, data_loader, device, num_classes):
    """
    计算测试集上的混淆矩阵
    """
    was_training = model.training
    model.eval()
    dl = data_loader
    try:
        from torch.utils.data.distributed import DistributedSampler
        if isinstance(getattr(dl, "sampler", None), DistributedSampler) and utils.is_main_process():
            dl = torch.utils.data.DataLoader(
                dl.dataset,
                batch_size=dl.batch_size,
                sampler=torch.utils.data.SequentialSampler(dl.dataset),
                num_workers=dl.num_workers,
                pin_memory=True,
            )
    except Exception:
        pass
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
    with torch.inference_mode():
        for images, targets in dl:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if targets.ndim > 1:
                targets = targets.argmax(dim=1)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            valid = (targets >= 0) & (targets < num_classes)
            if valid.sum() == 0:
                continue
            targets = targets[valid].view(-1)
            preds = preds[valid].view(-1)
            idx = targets * num_classes + preds
            binc = torch.bincount(idx, minlength=num_classes * num_classes)
            conf += binc.view(num_classes, num_classes)
    if utils.is_dist_avail_and_initialized():
        dist.barrier()
        dist.all_reduce(conf)
    if was_training:
        model.train()
    return conf.cpu()


def compute_weighted_metrics_from_confusion(conf_matrix):
    """
    从混淆矩阵计算加权平均的precision、recall、f1
    """
    cm = conf_matrix.to(torch.float32)
    tp = torch.diag(cm)
    support = cm.sum(dim=1)
    predicted = cm.sum(dim=0)
    eps = 1e-12
    precision_per = torch.where(predicted > 0, tp / (predicted + eps), torch.zeros_like(tp))
    recall_per = torch.where(support > 0, tp / (support + eps), torch.zeros_like(tp))
    denom = precision_per + recall_per
    f1_per = torch.where(denom > 0, 2 * precision_per * recall_per / (denom + eps), torch.zeros_like(tp))
    total = support.sum()
    if total > 0:
        w_precision = (precision_per * support).sum() / total
        w_recall = (recall_per * support).sum() / total
        w_f1 = (f1_per * support).sum() / total
    else:
        w_precision = torch.tensor(0.0)
        w_recall = torch.tensor(0.0)
        w_f1 = torch.tensor(0.0)
    return w_precision.item(), w_recall.item(), w_f1.item()


def plot_confusion_matrix_figure(conf_matrix, class_labels, title="Confusion Matrix"):
    """
    使用matplotlib生成带标注的混淆矩阵热力图（viridis）
    """
    cm = conf_matrix.numpy()
    n = cm.shape[0]
    fig_size = max(6.0, n * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm, interpolation="nearest", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticklabels(class_labels)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(n):
        for j in range(n):
            val = int(cm[i, j])
            if val > 0:
                ax.text(j, i, str(val), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=8)
            else:
                ax.text(j, i, "0", ha="center", va="center", color="black", fontsize=8)
    fig.tight_layout()
    return fig

class CustomLRScheduler:
    """
    自定义学习率调度器，支持多种策略
    
    策略1: deep_warmup
    - 预热阶段：0.01 LR，约400迭代，直到训练误差 < 80%（训练准确率 > 20%）
    - 预热后：0.1→0.01→0.001，在32k/48k迭代衰减，64k迭代终止
    """
    
    def __init__(self, optimizer, strategy="deep_warmup", total_iterations=None):
        self.optimizer = optimizer
        self.strategy = strategy
        self.total_iterations = total_iterations
        self.current_iteration = 0
        self.warmup_completed = False
        self.warmup_lr = 0.01
        self.main_lr = 0.1
        self.lr_milestones = [32000, 48000]  # 32k和48k迭代时衰减
        self.lr_gamma = 0.1  # 每次衰减10倍
        self.warmup_max_iterations = 400
        self.warmup_error_threshold = 0.80  # 训练误差 < 80%，即准确率 > 20%
        
    def step(self, iteration=None, train_error=None):
        """
        更新学习率
        
        Args:
            iteration: 当前迭代次数（如果为None，则使用内部计数器）
            train_error: 当前训练误差（用于判断预热结束条件）
        """
        if iteration is not None:
            self.current_iteration = iteration
        else:
            self.current_iteration += 1
        
        if self.strategy == "deep_warmup":
            self._step_deep_warmup(train_error)
        # 预留空间：可以在这里添加其他策略
        # elif self.strategy == "other_strategy":
        #     self._step_other_strategy()
    
    def _step_deep_warmup(self, train_error=None):
        """实现deep_warmup策略"""
        if not self.warmup_completed:
            # 预热阶段：使用0.01 LR
            # 结束条件：达到最大迭代次数 OR 训练误差 < 80%
            should_end_warmup = False
            
            if self.current_iteration >= self.warmup_max_iterations:
                should_end_warmup = True
                print(f"[LR策略] 预热达到最大迭代次数 {self.warmup_max_iterations}，结束预热", flush=True)
            
            if train_error is not None and train_error < self.warmup_error_threshold:
                should_end_warmup = True
                print(f"[LR策略] 训练误差 {train_error:.4f} < {self.warmup_error_threshold}，结束预热", flush=True)
            
            if should_end_warmup:
                self.warmup_completed = True
                # 切换到主学习率0.1
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.main_lr
                print(f"[LR策略] 预热结束，切换到主学习率 {self.main_lr}", flush=True)
            else:
                # 保持预热学习率
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.warmup_lr
        else:
            # 预热后阶段：0.1→0.01→0.001
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 检查是否到达衰减里程碑（使用 <= 而不是 ==，因为可能跳过某些迭代）
            for milestone in self.lr_milestones:
                if self.current_iteration == milestone:
                    new_lr = current_lr * self.lr_gamma
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    print(f"[LR策略] 迭代 {self.current_iteration}：学习率从 {current_lr} 衰减到 {new_lr}", flush=True)
                    break
    
    def get_last_lr(self):
        """获取当前学习率（兼容PyTorch调度器接口）"""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self):
        """保存状态（用于checkpoint）"""
        return {
            'current_iteration': self.current_iteration,
            'warmup_completed': self.warmup_completed,
            'strategy': self.strategy,
        }
    
    def load_state_dict(self, state_dict):
        """加载状态（用于恢复训练）"""
        self.current_iteration = state_dict.get('current_iteration', 0)
        self.warmup_completed = state_dict.get('warmup_completed', False)
        self.strategy = state_dict.get('strategy', 'deep_warmup')


def train_one_epoch_with_logging(model, criterion, optimizer, data_loader, device, epoch, args, 
                                   model_ema=None, scaler=None, custom_lr_scheduler=None):
    """
    训练一个epoch并返回平均loss和准确率，用于TensorBoard记录
    这是对原始train_one_epoch的扩展版本
    """
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    global_iteration = epoch * len(data_loader)  # 计算全局迭代次数
    
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target)
        if not torch.isfinite(loss):
            print("MANAGER_EVENT: LOSS_NAN", flush=True)
            sys.exit(2)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            if args.clip_grad_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if args.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        if model_ema and i % args.model_ema_steps == 0:
            model_ema.update_parameters(model)
            if epoch < args.lr_warmup_epochs:
                model_ema.n_averaged.fill_(0)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        
        # 计算当前batch的训练误差（1 - 准确率）
        current_train_error = 1.0 - (acc1.item() / 100.0)
        
        # 如果使用自定义学习率调度器，在每次迭代后更新
        if custom_lr_scheduler is not None:
            current_iteration = global_iteration + i
            custom_lr_scheduler.step(iteration=current_iteration, train_error=current_train_error)
        
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))
    
    # 同步所有进程的指标
    metric_logger.synchronize_between_processes()
    
    # 返回平均loss和准确率
    return metric_logger.loss.global_avg, metric_logger.acc1.global_avg


def get_args_parser(add_help=True):
    """
    获取支持多种数据集的参数解析器
    """
    # 首先获取原始的参数解析器
    parser = train.get_args_parser(add_help=False)
    
    # 覆盖data_path的默认值，避免使用原始train.py中的默认路径
    for action in parser._actions:
        if action.dest == 'data_path':
            action.default = "/home/dell/Documents/lx/vision/data/"
            break
    
    # 添加分布式训练参数
    parser.add_argument(
        "--local-rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (automatically set by torch.distributed.launch)"
    )
    
    # 添加数据集特定的参数组
    dataset_group = parser.add_argument_group('Dataset specific arguments')
    dataset_group.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
        choices=["cifar10", "cifar100", "fashionmnist"],
        help="Dataset to use: cifar10, cifar100, or fashionmnist (default: cifar10)"
    )
    
    # 添加路径参数组
    path_group = parser.add_argument_group('Path settings')
    path_group.add_argument(
        "--log-dir",
        default="/home/dell/Documents/lx/vision/log/",
        type=str,
        help="Path to save TensorBoard logs (default: /home/dell/Documents/lx/vision/log/)"
    )
    # 设置默认的输出目录为指定路径
    # 首先检查是否已经有output_dir参数，如果有则修改其默认值
    for action in parser._actions:
        if action.dest == 'output_dir':
            action.default = "/home/dell/Documents/lx/vision/output/"
            break
    else:
        # 如果没有找到，则添加该参数
        path_group.add_argument(
            "--output-dir",
            default="/home/dell/Documents/lx/vision/output/",
            type=str,
            help="Path to save model checkpoints (default: /home/dell/Documents/lx/vision/output/)")
    
    # 添加模型保存策略参数组
    saving_group = parser.add_argument_group('Model saving strategy')
    saving_group.add_argument(
        "--save-interval",
        default=1,
        type=int,
        help="Save model every N epochs (default: 1, save every epoch)"
    )
    saving_group.add_argument(
        "--save-only-final",
        action='store_true',
        help="Save only the final model (overrides --save-interval)"
    )
    
    # 添加学习率策略参数组
    lr_strategy_group = parser.add_argument_group('Learning rate strategy')
    lr_strategy_group.add_argument(
        "--lr-strategy",
        default=None,
        type=str,
        choices=["none", "deep_warmup"],
        help="Custom learning rate strategy. Options: none (use standard scheduler), deep_warmup (warmup + multi-stage decay)"
    )
    
    if add_help:
        parser.add_argument('-h', '--help', action='help', help='show this help message and exit')
    
    return parser


def main():
    """
    主函数，用于训练或测试ResNet模型在各种数据集上的性能
    """
    # 获取参数
    parser = get_args_parser()
    args = parser.parse_args()
    
    # 初始化分布式模式
    # 确保输出目录存在
    if args.output_dir:
        utils.mkdir(args.output_dir)
    utils.init_distributed_mode(args)
    
    # 打印详细的分布式信息
    if args.distributed:
        print(f"=" * 50, flush=True, force=True)
        print(f"Distributed Training Info:", flush=True, force=True)
        print(f"  Rank: {args.rank}", flush=True, force=True)
        print(f"  World Size: {args.world_size}", flush=True, force=True)
        print(f"  GPU: {args.gpu}", flush=True, force=True)
        print(f"  Device: cuda:{args.gpu}", flush=True, force=True)
        print(f"=" * 50, flush=True, force=True)
    
    print(args)
    
    # 设置设备
    device = torch.device(args.device)
    
    # 设置确定性算法
    if args.use_deterministic_algorithms:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
    
    # 加载数据
    train_dataset, test_dataset, data_loader, data_loader_test = load_data(args)
    
    # 获取类别数量
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    
    # 创建模型 - 使用ResNet并设置正确的类别数量
    print(f"Creating model: {args.model}", flush=True)
    model = torchvision.models.get_model(args.model, weights=args.weights, num_classes=num_classes)
    model.to(device)
    print(f"Model moved to device: {device}", flush=True)
    
    # 同步批归一化（如果需要）
    if args.distributed and args.sync_bn:
        print("Converting to SyncBatchNorm...", flush=True)
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # 损失函数
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    
    # 设置权重衰减  
    custom_keys_weight_decay = []
    if args.bias_weight_decay is not None:
        custom_keys_weight_decay.append(("bias", args.bias_weight_decay))
    if args.transformer_embedding_decay is not None:
        for key in ["class_token", "position_embedding", "relative_position_bias_table"]:
            custom_keys_weight_decay.append((key, args.transformer_embedding_decay))
    parameters = utils.set_weight_decay(
        model,
        args.weight_decay,
        norm_weight_decay=args.norm_weight_decay,
        custom_keys_weight_decay=custom_keys_weight_decay if len(custom_keys_weight_decay) > 0 else None,
    )
    
    # 优化器
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.amp else None
    
    # 学习率调度器
    custom_lr_scheduler = None
    lr_scheduler = None
    
    # 检查是否使用自定义学习率策略
    if hasattr(args, 'lr_strategy') and args.lr_strategy and args.lr_strategy.lower() != "none":
        strategy = args.lr_strategy.lower()
        # 计算总迭代次数（用于某些策略）
        total_iterations = len(data_loader) * args.epochs
        
        if strategy == "deep_warmup":
            # 使用自定义的deep_warmup策略
            custom_lr_scheduler = CustomLRScheduler(
                optimizer, 
                strategy="deep_warmup",
                total_iterations=total_iterations
            )
            # 初始化学习率为预热学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = custom_lr_scheduler.warmup_lr
            print(f"[LR策略] 使用 deep_warmup 策略：预热LR={custom_lr_scheduler.warmup_lr}，主LR={custom_lr_scheduler.main_lr}", flush=True)
            # 不创建标准的lr_scheduler，因为学习率由custom_lr_scheduler管理
            lr_scheduler = None
        # 预留空间：可以在这里添加其他策略
        # elif strategy == "other_strategy":
        #     custom_lr_scheduler = CustomLRScheduler(optimizer, strategy="other_strategy", total_iterations=total_iterations)
        #     lr_scheduler = None
        else:
            raise RuntimeError(f"Unknown LR strategy: {strategy}")
    else:
        # 使用标准的学习率调度器
        args.lr_scheduler = args.lr_scheduler.lower()
        if args.lr_scheduler == "steplr":
            main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
        elif args.lr_scheduler == "cosineannealinglr":
            main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
            )
        elif args.lr_scheduler == "exponentiallr":
            main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
        else:
            raise RuntimeError(
                f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
                "are supported."
            )
        
        # 学习率预热
        if args.lr_warmup_epochs > 0:
            if args.lr_warmup_method == "linear":
                warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
                )
            elif args.lr_warmup_method == "constant":
                warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                    optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
                )
            else:
                raise RuntimeError(
                    f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
                )
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
            )
        else:
            lr_scheduler = main_lr_scheduler
    
    # 分布式数据并行
    model_without_ddp = model
    if args.distributed:
        print(f"Rank {args.rank}: Wrapping model with DistributedDataParallel on GPU {args.gpu}", flush=True, force=True)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
        print(f"Rank {args.rank}: DDP model created successfully", flush=True, force=True)
    
    # 模型EMA（指数移动平均）
    model_ema = None
    if args.model_ema:
        adjust = args.world_size * args.batch_size * args.model_ema_steps / args.epochs
        alpha = 1.0 - args.model_ema_decay
        alpha = min(1.0, alpha * adjust)
        model_ema = utils.ExponentialMovingAverage(model_without_ddp, device=device, decay=1.0 - alpha)
    
    # 恢复检查点
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if not args.test_only:
            optimizer.load_state_dict(checkpoint["optimizer"])
            if lr_scheduler is not None and "lr_scheduler" in checkpoint:
                lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        args.start_epoch = checkpoint["epoch"] + 1
        if model_ema:
            model_ema.load_state_dict(checkpoint["model_ema"])
        if scaler:
            scaler.load_state_dict(checkpoint["scaler"])
        if custom_lr_scheduler and "custom_lr_scheduler" in checkpoint:
            custom_lr_scheduler.load_state_dict(checkpoint["custom_lr_scheduler"])
            # 恢复后需要根据状态设置当前学习率
            if custom_lr_scheduler.warmup_completed:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = custom_lr_scheduler.main_lr
            else:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = custom_lr_scheduler.warmup_lr
    
    # 仅测试模式
    if args.test_only:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if model_ema:
            train.evaluate(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
        else:
            train.evaluate(model, criterion, data_loader_test, device=device)
        return
    
    # 创建TensorBoard writer（只在主进程中创建）
    writer = None
    log_path = None
    if utils.is_main_process():
        writer, log_path = create_tensorboard_writer(args)
        print(f"TensorBoard writer created on main process", flush=True)
    
    # 训练循环
    print("Start training", flush=True)
    start_time = time.time()
    
    # 早停参数
    patience = 15  # 连续15轮无改善则停止
    min_improvement_threshold = 1e-4  # loss降低至少1e-4才算改善
    min_loss_threshold = 0.01  # 达到预设最小loss阈值也可停止
    
    # 早停状态跟踪
    best_val_loss = float('inf')
    no_improvement_count = 0
    best_epoch = 0
    best_model_path = os.path.join(args.output_dir, "best_model.pth")
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader.sampler.set_epoch(epoch)
        
        # 使用带日志记录的训练函数
        train_loss, train_acc = train_one_epoch_with_logging(
            model, criterion, optimizer, data_loader, device, epoch, args, model_ema, scaler, custom_lr_scheduler
        )
        
        # 如果使用标准学习率调度器，在每个epoch后更新
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # 在验证集上评估
        val_acc, val_loss = evaluate_with_loss(model, criterion, data_loader_test, device=device)
        
        # 记录到TensorBoard（只在主进程）
        if writer is not None:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_acc, epoch)
            writer.add_scalar('Accuracy/test', val_acc, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # 早停逻辑
        if val_loss < best_val_loss - min_improvement_threshold or val_loss < min_loss_threshold:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improvement_count = 0
            
            # 保存最优模型
            if args.output_dir:
                checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "args": args,
                    "best_val_loss": best_val_loss,
                }
                if lr_scheduler is not None:
                    checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
                if model_ema:
                    checkpoint["model_ema"] = model_ema.state_dict()
                if scaler:
                    checkpoint["scaler"] = scaler.state_dict()
                if custom_lr_scheduler:
                    checkpoint["custom_lr_scheduler"] = custom_lr_scheduler.state_dict()
                utils.save_on_master(checkpoint, best_model_path)
                if utils.is_main_process():
                    print(f"Best model saved at epoch {epoch}, val_loss: {best_val_loss:.6f}", flush=True)
        else:
            no_improvement_count += 1
            print(f"No improvement for {no_improvement_count}/{patience} epochs, current val_loss: {val_loss:.6f}, best_val_loss: {best_val_loss:.6f}", flush=True)
        
        # 检查是否触发早停
        if no_improvement_count >= patience:
            print(f"Early stopping triggered at epoch {epoch}!", flush=True)
            print(f"Best epoch: {best_epoch}, Best val_loss: {best_val_loss:.6f}", flush=True)
            print(f"Current epoch: {epoch}, Current val_loss: {val_loss:.6f}", flush=True)
            print(f"No improvement for {no_improvement_count} consecutive epochs.", flush=True)
            break
        
        # 如果使用EMA模型，也记录其性能
        if model_ema:
            ema_acc, ema_loss = evaluate_with_loss(model_ema, criterion, data_loader_test, device=device, log_suffix="EMA")
            if writer is not None:
                writer.add_scalar('Accuracy/test_ema', ema_acc, epoch)
                writer.add_scalar('Loss/test_ema', ema_loss, epoch)
        
        # 保存检查点
        if args.output_dir:
            checkpoint = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            if lr_scheduler is not None:
                checkpoint["lr_scheduler"] = lr_scheduler.state_dict()
            if model_ema:
                checkpoint["model_ema"] = model_ema.state_dict()
            if scaler:
                checkpoint["scaler"] = scaler.state_dict()
            if custom_lr_scheduler:
                checkpoint["custom_lr_scheduler"] = custom_lr_scheduler.state_dict()
            
            # 提取resnet层数（从model名称中提取数字部分）
            resnet_depth = ''.join(filter(str.isdigit, args.model))
            
            # 获取当前时间戳（用于最终模型命名）
            current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 判断GPU配置：single或double
            gpu_config = "double" if args.distributed else "single"
            
            # 格式化学习率（去除小数点，例如0.1 -> lr01, 0.01 -> lr001）
            # 先格式化为固定小数位，再去除小数点
            lr_formatted = f"{args.lr:.6f}".rstrip('0').rstrip('.')
            lr_str = f"lr{lr_formatted.replace('.', '')}"
            
            # 根据保存策略决定是否保存当前模型
            is_last_epoch = (epoch == args.epochs - 1)
            should_save_current = False
            
            # 只保存最终模型
            if args.save_only_final:
                if is_last_epoch:
                    should_save_current = True
                    # 最终模型采用统一命名格式："resnet层数_数据集_优化器_lr_batchsize_gpu配置_训练迭代次数_完成时间.pth"
                    final_model_name = f"resnet{resnet_depth}_{args.dataset}_{args.opt.lower()}_{lr_str}_bs{args.batch_size}_{gpu_config}gpu_{args.epochs}_{current_timestamp}.pth"
                    model_path = os.path.join(args.output_dir, final_model_name)
                    utils.save_on_master(checkpoint, model_path)
                    if utils.is_main_process():
                        print(f"Final model saved to: {model_path}", flush=True)
                        # 输出特定格式的路径信息，方便manager解析
                        print(f"MANAGER_PATH_INFO: Model={model_path}", flush=True)
            else:
                # 按间隔保存模型
                if (epoch % args.save_interval == 0) or is_last_epoch:
                    should_save_current = True
                    # 中间检查点也采用统一命名格式，但使用当前epoch作为迭代次数
                    model_name = f"resnet{resnet_depth}_{args.dataset}_{args.opt.lower()}_{lr_str}_bs{args.batch_size}_{gpu_config}gpu_{epoch+1}_{current_timestamp}.pth"
                    model_path = os.path.join(args.output_dir, model_name)
                    utils.save_on_master(checkpoint, model_path)
                    if utils.is_main_process():
                        print(f"Model checkpoint saved to: {model_path}", flush=True)
            
            # 总是保存最新的检查点用于恢复训练
            utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))
    
    total_time = time.time() - start_time
    total_time_str = str(time.strftime('%H:%M:%S', time.gmtime(int(total_time))))
    print(f"Training time {total_time_str}", flush=True)
    
    # 在TensorBoard中记录训练完成时间（只在主进程）
    if writer is not None:
        try:
            eval_model = model_ema if model_ema is not None else model
            conf_matrix = compute_confusion_matrix(eval_model, data_loader_test, device, num_classes)
            p, r, f1 = compute_weighted_metrics_from_confusion(conf_matrix)
            acc = (torch.diag(conf_matrix).sum().item() / conf_matrix.sum().item()) if conf_matrix.sum().item() > 0 else 0.0
            writer.add_scalar('Test/precision', p, args.epochs)
            writer.add_scalar('Test/recall', r, args.epochs)
            writer.add_scalar('Test/f1', f1, args.epochs)
            print(f"MANAGER_METRIC: precision={p:.6f}, recall={r:.6f}, f1={f1:.6f}, acc={acc:.6f}", flush=True)
            writer.add_text('Test/summary', f"Acc@1: {acc:.4f}, Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}", args.epochs)
            labels = getattr(test_dataset, 'classes', None)
            if not labels or len(labels) != conf_matrix.shape[0]:
                labels = [str(i) for i in range(conf_matrix.shape[0])]
            fig = plot_confusion_matrix_figure(conf_matrix, labels, title='Confusion Matrix (Test)')
            writer.add_figure('Test/confusion_matrix', fig, args.epochs)
        except Exception as e:
            print(f"Warning: failed to log test metrics to TensorBoard: {e}", flush=True)
        writer.add_text('Training/completion_time', 
                        f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"Total training time: {total_time_str}\n"
                        f"Model: {args.model}\n"
                        f"Optimizer: {args.opt}\n"
                        f"Dataset: {args.dataset}")
        
        # 关闭writer
        writer.close()
        print(f"TensorBoard logs saved to: {log_path}", flush=True)


if __name__ == "__main__":
    main()