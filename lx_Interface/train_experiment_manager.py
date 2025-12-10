#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练实验管理器
用于依次执行多个训练任务，支持不同的参数组合和GPU配置
"""
import os
import sys
import time
import subprocess
import logging
from datetime import datetime
import re

# 创建日志目录
MANAGER_LOG_DIR = "/home/dell/Documents/lx/vision/manager_log/"
os.makedirs(MANAGER_LOG_DIR, exist_ok=True)

# 生成带时间戳的日志文件名
log_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(MANAGER_LOG_DIR, f"train_experiment_{log_timestamp}.log")

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"训练实验管理器日志文件: {log_file}")

class TrainingExperimentManager:
    """训练实验管理器类"""
    
    def __init__(self, base_script="resnet_trainer.py"):
        """
        初始化实验管理器
        
        Args:
            base_script: 基础训练脚本名称
        """
        self.base_script = base_script
        self.experiments = []
        self.results = []
    
    def add_experiment(self, name, params, use_distributed=True, gpu_id=None):
        """
        添加一个训练实验
        
        Args:
            name: 实验名称
            params: 参数字典
            use_distributed: 是否使用分布式训练（双GPU）
            gpu_id: 指定使用的GPU ID，如果不使用分布式训练
        """
        experiment = {
            "name": name,
            "params": params,
            "use_distributed": use_distributed,
            "gpu_id": gpu_id
        }
        self.experiments.append(experiment)
        logger.info(f"添加实验: {name}, 分布式: {use_distributed}, GPU: {gpu_id}")
    
    def generate_command(self, experiment):
        """
        根据实验配置生成命令
        
        Args:
            experiment: 实验配置字典
            
        Returns:
            命令列表
        """
        params = experiment["params"]
        use_distributed = experiment["use_distributed"]
        gpu_id = experiment["gpu_id"]
        
        # 基础命令
        if use_distributed:
            # 分布式训练
            cmd = ["torchrun", "--nproc_per_node=2", self.base_script]
        else:
            # 单GPU训练
            cmd = ["python", self.base_script]
            # 注意：CUDA_VISIBLE_DEVICES通过环境变量传递，不放在命令列表中
        
        # 添加参数
        for key, value in params.items():
            # 忽略值为None的参数
            if value is None:
                continue

            # 处理参数名（将下划线转换为破折号）
            param_name = f"--{key.replace('_', '-')}"

            # 对于布尔类型，True 仅添加参数名，False 则跳过（使用默认值）
            if isinstance(value, bool):
                if value:
                    cmd.append(param_name)
                continue

            cmd.append(param_name)

            # 对于路径参数，确保使用绝对路径并正确转义
            if isinstance(value, (list, tuple)):
                # 保持列表参数的原始格式
                for item in value:
                    if 'path' in key.lower() or 'dir' in key.lower():
                        item_str = os.path.abspath(os.path.expanduser(str(item)))
                    else:
                        item_str = str(item)
                    cmd.append(item_str)
                continue

            if 'path' in key.lower() or 'dir' in key.lower():
                # 转换为绝对路径并规范化
                value_str = os.path.abspath(os.path.expanduser(str(value)))
            else:
                value_str = str(value)
            cmd.append(value_str)
        
        return cmd
    
    def calculate_output_paths(self, experiment):
        """
        根据实验配置计算预期的输出路径
        
        Args:
            experiment: 实验配置字典
            
        Returns:
            包含TensorBoard路径和模型路径的字典
        """
        from datetime import datetime
        
        params = experiment["params"]
        use_distributed = experiment["use_distributed"]
        
        # 获取参数值
        model = params.get("model", "resnet18")
        dataset = params.get("dataset", "fashionmnist")
        opt = params.get("opt", "sgd").lower()
        lr = params.get("lr", 0.1)
        batch_size = params.get("batch_size", 256)
        epochs = params.get("epochs", 60)
        log_dir = params.get("log_dir", "/home/dell/Documents/lx/vision/log/")
        output_dir = params.get("output_dir", "/home/dell/Documents/lx/vision/output/")
        
        # 提取resnet层数
        resnet_depth = ''.join(filter(str.isdigit, model))
        
        # 判断GPU配置
        gpu_config = "double" if use_distributed else "single"
        
        # 格式化学习率
        lr_formatted = f"{lr:.6f}".rstrip('0').rstrip('.')
        lr_str = f"lr{lr_formatted.replace('.', '')}"
        
        # 生成时间戳（使用当前时间，因为实际时间戳会在训练时生成）
        # 这里只生成路径模板，实际路径会在训练时确定
        timestamp_template = "YYYYMMDD_HHMMSS"
        
        # TensorBoard日志路径模板
        log_dir_name_template = f"resnet{resnet_depth}_{dataset}_{opt}_{lr_str}_bs{batch_size}_{gpu_config}gpu_{epochs}_{timestamp_template}"
        tensorboard_path_template = os.path.join(log_dir, log_dir_name_template)
        
        # 模型文件路径模板
        model_name_template = f"resnet{resnet_depth}_{dataset}_{opt}_{lr_str}_bs{batch_size}_{gpu_config}gpu_{epochs}_{timestamp_template}.pth"
        model_path_template = os.path.join(output_dir, model_name_template)
        
        return {
            "tensorboard_path_template": tensorboard_path_template,
            "model_path_template": model_path_template,
            "tensorboard_dir": log_dir,
            "model_dir": output_dir
        }
    
    def run_experiment(self, experiment):
        """
        运行单个实验
        
        Args:
            experiment: 实验配置字典
            
        Returns:
            实验结果字典
        """
        name = experiment["name"]
        logger.info(f"[实验开始] {name}")
        logger.info(f"[实验配置] 分布式: {experiment['use_distributed']}, GPU ID: {experiment['gpu_id']}")
        logger.info(f"[实验参数] {experiment['params']}")
        
        # 计算输出路径
        output_paths = self.calculate_output_paths(experiment)
        logger.info(f"[输出路径] TensorBoard日志目录: {output_paths['tensorboard_dir']}")
        logger.info(f"[输出路径] TensorBoard日志模板: {output_paths['tensorboard_path_template']}")
        logger.info(f"[输出路径] 模型文件目录: {output_paths['model_dir']}")
        logger.info(f"[输出路径] 模型文件模板: {output_paths['model_path_template']}")
        
        # 生成命令
        cmd = self.generate_command(experiment)
        # 生成命令字符串用于日志记录（对路径进行引号转义）
        cmd_parts = []
        for i, part in enumerate(cmd):
            if i > 0 and (part.startswith('/') or 'path' in cmd[i-1].lower() or 'dir' in cmd[i-1].lower()):
                # 对路径参数添加引号，防止空格或特殊字符导致解析错误
                cmd_parts.append(f'"{part}"')
            else:
                cmd_parts.append(part)
        cmd_str = " ".join(cmd_parts)
        logger.info(f"[执行命令] {cmd_str}")
        logger.info(f"\n{'='*80}\n开始训练，实时输出如下：\n{'='*80}\n")
        
        # 记录开始时间
        start_time = time.time()
        start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"[开始时间] {start_datetime}")
        
        # 用于捕获路径信息的变量
        actual_tensorboard_path = None
        actual_model_path = None
        
        try:
            # 使用Popen来同时捕获输出和实时显示
            # 使用列表形式传递命令，避免shell解析路径时出现问题
            env = os.environ.copy()
            actual_cmd = []
            
            # 从experiment字典中提取变量
            use_distributed = experiment["use_distributed"]
            gpu_id = experiment["gpu_id"]
            
            # 处理CUDA_VISIBLE_DEVICES环境变量
            if not use_distributed and gpu_id is not None:
                env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
                # 移除cmd中的CUDA_VISIBLE_DEVICES部分（如果存在）
                actual_cmd = [c for c in cmd if not c.startswith('CUDA_VISIBLE_DEVICES=')]
            else:
                actual_cmd = cmd
            
            # 使用列表形式传递命令，避免shell解析问题
            process = subprocess.Popen(
                actual_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            
            # 实时读取并显示输出，同时解析路径信息与指标
            output_lines = []
            acc1_val = None
            acc5_val = None
            precision_val = None
            recall_val = None
            f1_val = None
            loss_nan_detected = False
            for line in process.stdout:
                print(line, end='', flush=True)  # 实时显示
                output_lines.append(line)
                
                # 解析路径信息
                if "MANAGER_PATH_INFO: TensorBoard=" in line:
                    actual_tensorboard_path = line.split("MANAGER_PATH_INFO: TensorBoard=")[1].strip()
                elif "MANAGER_PATH_INFO: Model=" in line:
                    actual_model_path = line.split("MANAGER_PATH_INFO: Model=")[1].strip()
                # 解析准确率
                m1 = re.search(r"Acc@1\s+([0-9.]+)", line)
                if m1:
                    try:
                        acc1_val = float(m1.group(1))
                    except Exception:
                        pass
                m5 = re.search(r"Acc@5\s+([0-9.]+)", line)
                if m5:
                    try:
                        acc5_val = float(m5.group(1))
                    except Exception:
                        pass
                mm = re.search(r"MANAGER_METRIC:\s*precision=([0-9.]+),\s*recall=([0-9.]+),\s*f1=([0-9.]+)", line)
                if mm:
                    try:
                        precision_val = float(mm.group(1))
                        recall_val = float(mm.group(2))
                        f1_val = float(mm.group(3))
                    except Exception:
                        pass
                # 检测Loss为NaN事件
                if "MANAGER_EVENT: LOSS_NAN" in line:
                    loss_nan_detected = True
            
            # 等待进程完成
            return_code = process.wait()
            
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, cmd_str)
            
            # 记录结束时间
            end_time = time.time()
            duration = end_time - start_time
            end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 更新输出路径信息
            if actual_tensorboard_path:
                output_paths['actual_tensorboard_path'] = actual_tensorboard_path
            if actual_model_path:
                output_paths['actual_model_path'] = actual_model_path
            
            logger.info(f"\n{'='*80}\n[实验成功] {name}")
            logger.info(f"[结束时间] {end_datetime}")
            logger.info(f"[耗时] {duration:.2f}秒 ({duration/60:.2f}分钟)")
            logger.info(f"[输出路径] TensorBoard日志目录: {output_paths['tensorboard_dir']}")
            if actual_tensorboard_path:
                logger.info(f"[输出路径] TensorBoard实际路径: {actual_tensorboard_path}")
            else:
                logger.info(f"[输出路径] TensorBoard日志模板: {output_paths['tensorboard_path_template']}")
            logger.info(f"[输出路径] 模型文件目录: {output_paths['model_dir']}")
            if actual_model_path:
                logger.info(f"[输出路径] 模型文件实际路径: {actual_model_path}")
            else:
                logger.info(f"[输出路径] 模型文件模板: {output_paths['model_path_template']}")
            logger.info(f"{'='*80}\n")
            
            # 保存结果
            experiment_result = {
                "name": name,
                "status": "success",
                "duration": duration,
                "command": cmd_str,
                "start_time": start_datetime,
                "end_time": end_datetime,
                "stdout": "输出已实时显示",
                "stderr": "",
                "output_paths": output_paths,
                "acc1": acc1_val,
                "acc5": acc5_val,
                "precision": precision_val,
                "recall": recall_val,
                "f1": f1_val
            }
            if loss_nan_detected:
                experiment_result["status"] = "failed"
                experiment_result["stderr"] = "Loss 为 NaN，实验失败"
            
        except subprocess.CalledProcessError as e:
            end_time = time.time()
            duration = end_time - start_time
            end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            logger.error(f"\n{'='*80}\n[实验失败] {name}")
            logger.error(f"[失败时间] {end_datetime}")
            logger.error(f"[耗时] {duration:.2f}秒 ({duration/60:.2f}分钟)")
            logger.error(f"[错误码] {e.returncode}")
            logger.error(f"[错误类型] CalledProcessError")
            logger.error(f"{'='*80}\n")
            
            experiment_result = {
                "name": name,
                "status": "failed",
                "duration": duration,
                "command": cmd_str,
                "start_time": start_datetime,
                "end_time": end_datetime,
                "stdout": "输出已实时显示",
                "stderr": f"进程返回码: {e.returncode}",
                "error_code": e.returncode,
                "output_paths": output_paths,
                "acc1": acc1_val,
                "acc5": acc5_val,
                "precision": precision_val,
                "recall": recall_val,
                "f1": f1_val
            }
        except KeyboardInterrupt:
            end_time = time.time()
            duration = end_time - start_time
            end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            logger.warning(f"\n{'='*80}\n[实验中断] {name}")
            logger.warning(f"[中断时间] {end_datetime}")
            logger.warning(f"[耗时] {duration:.2f}秒 ({duration/60:.2f}分钟)")
            logger.warning(f"[中断原因] 用户中断 (Ctrl+C)")
            logger.warning(f"{'='*80}\n")
            
            experiment_result = {
                "name": name,
                "status": "interrupted",
                "duration": duration,
                "command": cmd_str,
                "start_time": start_datetime,
                "end_time": end_datetime,
                "stdout": "输出已实时显示",
                "stderr": "用户中断 (Ctrl+C)",
                "output_paths": output_paths,
                "acc1": acc1_val,
                "acc5": acc5_val,
                "precision": precision_val,
                "recall": recall_val,
                "f1": f1_val
            }
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            logger.error(f"\n{'='*80}\n[实验异常] {name}")
            logger.error(f"[异常时间] {end_datetime}")
            logger.error(f"[耗时] {duration:.2f}秒 ({duration/60:.2f}分钟)")
            logger.error(f"[异常类型] {type(e).__name__}")
            logger.error(f"[异常信息] {str(e)}")
            logger.error(f"{'='*80}\n")
            
            experiment_result = {
                "name": name,
                "status": "failed",
                "duration": duration,
                "command": cmd_str,
                "start_time": start_datetime,
                "end_time": end_datetime,
                "stdout": "输出已实时显示",
                "stderr": f"异常: {type(e).__name__}: {str(e)}",
                "output_paths": output_paths,
                "acc1": acc1_val,
                "acc5": acc5_val,
                "precision": precision_val,
                "recall": recall_val,
                "f1": f1_val
            }
        
        return experiment_result
    
    def run_all_experiments(self):
        """
        运行所有添加的实验
        """
        total_start_time = time.time()
        total_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        logger.info(f"\n{'='*80}")
        logger.info(f"[批次开始] 开始运行所有实验")
        logger.info(f"[批次时间] {total_start_datetime}")
        logger.info(f"[实验总数] {len(self.experiments)} 个")
        logger.info(f"[提示] 按 Ctrl+C 可以中断当前实验并继续下一个")
        logger.info(f"{'='*80}\n")
        
        skipped_count = 0
        
        try:
            for i, experiment in enumerate(self.experiments, 1):
                logger.info(f"\n{'='*80}")
                logger.info(f"[进度] 实验 {i}/{len(self.experiments)}")
                logger.info(f"{'='*80}\n")
                
                # 检查是否应该跳过实验（可以根据需要添加跳过逻辑）
                # 例如：检查输出目录是否已存在模型文件
                should_skip = False
                skip_reason = ""
                
                # 这里可以添加跳过逻辑，例如检查模型是否已存在
                # if self._check_model_exists(experiment):
                #     should_skip = True
                #     skip_reason = "模型文件已存在"
                
                if should_skip:
                    skipped_count += 1
                    logger.warning(f"[实验跳过] {experiment['name']}")
                    logger.warning(f"[跳过原因] {skip_reason}")
                    
                    skip_result = {
                        "name": experiment["name"],
                        "status": "skipped",
                        "duration": 0,
                        "command": "",
                        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "stdout": "",
                        "stderr": skip_reason
                    }
                    self.results.append(skip_result)
                    continue
                
                result = self.run_experiment(experiment)
                self.results.append(result)
                
                # 如果实验被中断，自动继续下一个实验（因为是无人值守模式）
                if result['status'] == 'interrupted':
                    logger.warning("[自动继续] 实验被中断，5秒后继续下一个实验...")
                    time.sleep(5)
                
                # 如果实验失败，记录但继续下一个
                if result['status'] == 'failed':
                    logger.warning("[继续执行] 实验失败，继续执行下一个实验...")
                
                # 不在中途保存或询问，统一在批次结束时保存
        
        except KeyboardInterrupt:
            total_end_time = time.time()
            total_duration = total_end_time - total_start_time
            logger.warning(f"\n\n{'='*80}")
            logger.warning(f"[批次中断] 所有实验被用户中断")
            logger.warning(f"[已执行] {len(self.results)}/{len(self.experiments)} 个实验")
            logger.warning(f"[已耗时] {total_duration:.2f}秒 ({total_duration/60:.2f}分钟)")
            logger.warning(f"{'='*80}\n")
        
        finally:
            total_end_time = time.time()
            total_duration = total_end_time - total_start_time
            total_end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            success_count = sum(1 for r in self.results if r['status'] == 'success')
            failed_count = sum(1 for r in self.results if r['status'] == 'failed')
            interrupted_count = sum(1 for r in self.results if r['status'] == 'interrupted')
            skipped_count = sum(1 for r in self.results if r['status'] == 'skipped')
            
            logger.info(f"\n{'='*80}")
            logger.info(f"[批次结束] 实验批次执行完成")
            logger.info(f"[结束时间] {total_end_datetime}")
            logger.info(f"[总耗时] {total_duration:.2f}秒 ({total_duration/60:.2f}分钟)")
            logger.info(f"[实验统计]")
            logger.info(f"  - 总数: {len(self.experiments)}")
            logger.info(f"  - 完成: {len(self.results)}")
            logger.info(f"  - 成功: {success_count}")
            logger.info(f"  - 失败: {failed_count}")
            logger.info(f"  - 中断: {interrupted_count}")
            logger.info(f"  - 跳过: {skipped_count}")
            logger.info(f"{'='*80}\n")
            
            # 最终保存结果
            self.save_results()
    
    def save_results(self):
        """
        保存实验结果到文件
        """
        # 询问用户是否确认保存
        print(f"\n{'='*80}")
        print("是否保存实验结果？")
        print(f"{'='*80}")
        confirm = input("确认保存实验结果到文件？(y/n): ")
        
        if confirm.lower() != 'y':
            logger.info("用户选择不保存实验结果")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = os.path.join(MANAGER_LOG_DIR, f"experiment_results_{timestamp}.txt")
        
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write(f"实验结果汇总 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # 统计信息
            success_count = sum(1 for r in self.results if r['status'] == 'success')
            failed_count = sum(1 for r in self.results if r['status'] == 'failed')
            interrupted_count = sum(1 for r in self.results if r['status'] == 'interrupted')
            skipped_count = sum(1 for r in self.results if r['status'] == 'skipped')
            
            f.write("统计信息:\n")
            f.write(f"  总数: {len(self.experiments)}\n")
            f.write(f"  完成: {len(self.results)}\n")
            f.write(f"  成功: {success_count}\n")
            f.write(f"  失败: {failed_count}\n")
            f.write(f"  中断: {interrupted_count}\n")
            f.write(f"  跳过: {skipped_count}\n\n")
            f.write("="*80 + "\n\n")
            
            for i, result in enumerate(self.results, 1):
                f.write(f"实验 {i}: {result['name']}\n")
                f.write(f"状态: {result['status']}\n")
                if 'start_time' in result:
                    f.write(f"开始时间: {result['start_time']}\n")
                if 'end_time' in result:
                    f.write(f"结束时间: {result['end_time']}\n")
                f.write(f"耗时: {result['duration']:.2f}秒\n")
                if result.get('command'):
                    f.write(f"命令: {result['command']}\n")
                
                # 输出路径信息
                if 'output_paths' in result:
                    paths = result['output_paths']
                    f.write(f"输出路径:\n")
                    f.write(f"  TensorBoard日志目录: {paths['tensorboard_dir']}\n")
                    if 'actual_tensorboard_path' in paths:
                        f.write(f"  TensorBoard实际路径: {paths['actual_tensorboard_path']}\n")
                    else:
                        f.write(f"  TensorBoard日志模板: {paths['tensorboard_path_template']}\n")
                    f.write(f"  模型文件目录: {paths['model_dir']}\n")
                    if 'actual_model_path' in paths:
                        f.write(f"  模型文件实际路径: {paths['actual_model_path']}\n")
                    else:
                        f.write(f"  模型文件模板: {paths['model_path_template']}\n")
                    if 'actual_tensorboard_path' not in paths or 'actual_model_path' not in paths:
                        f.write(f"  注意: 实际文件路径中的时间戳会在训练时确定，请查看目录中的最新文件\n")
                
                if result['status'] in ['failed', 'interrupted', 'skipped'] and result.get('stderr'):
                    status_desc = {'failed': '错误', 'interrupted': '中断', 'skipped': '跳过'}.get(result['status'], '信息')
                    f.write(f"{status_desc}信息: {result['stderr']}\n")
                
                f.write("-"*80 + "\n\n")
        
        logger.info(f"实验结果已保存到: {result_file}")
    
    def generate_comparison_report(self):
        """
        生成实验对比报告
        """
        if not self.results:
            logger.warning("没有实验结果可以生成报告")
            return
        
        # 询问用户是否确认生成报告
        print(f"\n{'='*80}")
        print("是否生成实验对比报告？")
        print(f"{'='*80}")
        confirm = input("确认生成实验对比报告？(y/n): ")
        
        if confirm.lower() != 'y':
            logger.info("用户选择不生成实验对比报告")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(MANAGER_LOG_DIR, f"experiment_comparison_{timestamp}.md")
        
        success_count = sum(1 for r in self.results if r['status'] == 'success')
        failed_count = sum(1 for r in self.results if r['status'] == 'failed')
        interrupted_count = sum(1 for r in self.results if r['status'] == 'interrupted')
        skipped_count = sum(1 for r in self.results if r['status'] == 'skipped')
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"# 训练实验对比报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 实验概述\n")
            f.write(f"- 总实验数: {len(self.experiments)}\n")
            f.write(f"- 完成实验数: {len(self.results)}\n")
            f.write(f"- 成功实验数: {success_count}\n")
            f.write(f"- 失败实验数: {failed_count}\n")
            f.write(f"- 中断实验数: {interrupted_count}\n")
            f.write(f"- 跳过实验数: {skipped_count}\n\n")
            
            f.write("## 实验详情\n")
            f.write("| 实验名称 | 状态 | Acc@1 | Acc@5 | Precision | Recall | F1 | 耗时(秒) | 备注 |\n")
            f.write("|---------|------|-------|-------|----------|--------|----|---------|------|\n")
            
            for result in self.results:
                if result['status'] == 'success':
                    status = "✅ 成功"
                    note = ""
                elif result['status'] == 'failed':
                    status = "❌ 失败"
                    note = "查看日志了解错误详情"
                elif result['status'] == 'interrupted':
                    status = "⚠️ 中断"
                    note = "训练被用户中断"
                elif result['status'] == 'skipped':
                    status = "⏭️ 跳过"
                    note = result.get('stderr', '实验被跳过')
                else:
                    status = "❓ 未知"
                    note = ""
                acc1_str = f"{result.get('acc1'):.3f}" if isinstance(result.get('acc1'), (int, float)) else "-"
                acc5_str = f"{result.get('acc5'):.3f}" if isinstance(result.get('acc5'), (int, float)) else "-"
                precision_str = f"{result.get('precision'):.3f}" if isinstance(result.get('precision'), (int, float)) else "-"
                recall_str = f"{result.get('recall'):.3f}" if isinstance(result.get('recall'), (int, float)) else "-"
                f1_str = f"{result.get('f1'):.3f}" if isinstance(result.get('f1'), (int, float)) else "-"
                f.write(f"| {result['name']} | {status} | {acc1_str} | {acc5_str} | {precision_str} | {recall_str} | {f1_str} | {result['duration']:.2f} | {note} |\n")
        
        logger.info(f"对比报告已生成: {report_file}")

def main():
    """
    主函数，设置并运行实验
    """
    # 创建实验管理器
    manager = TrainingExperimentManager()
    
    # 基础参数
    base_params = {
        "data_path": "/home/dell/Documents/lx/vision/data/",
        "dataset": "fashionmnist",
        "epochs": 100,
        "weight_decay": 0.0001,
        "amp": True,  # 启用自动混合精度训练
        "model_ema": False,  # 启用指数移动平均
        "save_only_final": True,  # 只保存最终模型
    }
    
    # 定义参数组合
    # model_depths = [50, 101, 152]  # 网络层数
    # optimizers = ["sgd", "adamW", "rmsprop"]  # 优化器类型
    # batch_sizes = [128, 256]  # 批处理大小
    
    # # 遍历所有参数组合
    # for depth in model_depths:
    #     for opt_type in optimizers:
    #         for batch_size in batch_sizes:
    #             # 复制基础参数
    #             exp_params = base_params.copy()
    #             exp_params["model"] = f"resnet{depth}"
    #             exp_params["opt"] = opt_type
    #             exp_params["batch_size"] = batch_size
                
    #             # 根据优化器类型设置合适的学习率和动量
    #             if opt_type == "sgd":
    #                 exp_params["lr"] = 0.1
    #                 exp_params["momentum"] = 0.9
    #             elif opt_type == "adamW":
    #                 exp_params["lr"] = 0.001
    #             elif opt_type == "rmsprop":
    #                 exp_params["lr"] = 0.001
    #                 exp_params["momentum"] = 0.9
                
    #             # 生成实验名称
    #             exp_name = f"ResNet{depth}_fashionmnist_{opt_type}_bs{batch_size}"
                
    #             # 添加实验
    #             manager.add_experiment(
    #                 name=exp_name,
    #                 params=exp_params,
    #                 use_distributed=False
    #             )
    #             
    # 添加ResNet152实验 - sgd优化器 +  大学习率实验
    resnet152_sgd_fixed_params = base_params.copy()
    resnet152_sgd_fixed_params["model"] = "resnet152"
    resnet152_sgd_fixed_params["opt"] = "sgd"
    resnet152_sgd_fixed_params["lr"] = 0.01
    resnet152_sgd_fixed_params["batch_size"] = 128
    resnet152_sgd_fixed_params["lr_strategy"] = "none"  # 使用标准调度器
    resnet152_sgd_fixed_params["lr_scheduler"] = "CosineAnnealingLR"  
    manager.add_experiment(
        name="ResNet152_fashionmnist_sgd_lr001_bs128_cos",
        params=resnet152_sgd_fixed_params,
        use_distributed=False
    )
    
        # 添加ResNet101实验 - sgd优化器 +  固定学习率  大学习率实验
    resnet101_sgd_fixed_params = base_params.copy()
    resnet101_sgd_fixed_params["model"] = "resnet101"
    resnet101_sgd_fixed_params["opt"] = "sgd"
    resnet101_sgd_fixed_params["lr"] = 0.01
    resnet101_sgd_fixed_params["batch_size"] = 128
    resnet101_sgd_fixed_params["lr_strategy"] = "none"  # 使用标准调度器
    resnet101_sgd_fixed_params["lr_scheduler"] = "CosineAnnealingLR"  
    manager.add_experiment(
        name="ResNet101_fashionmnist_sgd_lr001_bs128_cos",
        params=resnet101_sgd_fixed_params,
        use_distributed=False
    )
    
    
    # 询问用户是否确认开始批量运行所有实验
    print(f"\n{'='*80}")
    print(f"即将开始运行 {len(manager.experiments)} 个实验")
    print(f"{'='*80}")
    
    # 显示即将运行的实验列表
    for i, exp in enumerate(manager.experiments, 1):
        print(f"{i}. {exp['name']}")
        print(f"   - 分布式: {exp['use_distributed']}, GPU ID: {exp['gpu_id']}")
    
    confirm = input("\n确认开始运行所有实验？(y/n): ")
    
    if confirm.lower() == 'y':
        # 运行所有实验
        manager.run_all_experiments()
        
        # 生成对比报告
        manager.generate_comparison_report()
    else:
        print("实验运行已取消")
        sys.exit(0)

if __name__ == "__main__":
    main()