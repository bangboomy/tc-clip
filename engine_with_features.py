"""
Reference: https://github.com/muzairkhattak/ViFi-CLIP/blob/main/main.py
Modified for using pre-extracted features.
"""

import wandb
from torch.amp import autocast, GradScaler
import torch
import torch.distributed as dist
import numpy as np

from utils.tools import accuracy_top1_top5
from utils.logger import MetricLogger, SmoothedValue


@torch.no_grad()
def validate_with_features(val_loader, model, logger, config, features_path):
    model.eval()
    num_classes = len(val_loader.dataset.classes)
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Val with pre-extracted features:'

    # 添加类别级别的统计
    class_correct = torch.zeros(num_classes).cuda()
    class_total = torch.zeros(num_classes).cuda()

    # 加载预提取的特征
    logger.info(f"Loading pre-extracted features from {features_path}")
    features_data = torch.load(features_path, map_location='cpu')

    # 获取特征和标签
    pre_extracted_features = features_data['features']  # [N, T, D]
    pre_extracted_labels = features_data['labels']     # [N]

    # 检查特征数量是否匹配
    if len(pre_extracted_features) != len(val_loader.dataset):
        logger.warning(f"Number of pre-extracted features ({len(pre_extracted_features)}) "
                      f"does not match dataset size ({len(val_loader.dataset)})")

    # 将特征和标签移到GPU
    pre_extracted_features = pre_extracted_features.cuda()
    pre_extracted_labels = pre_extracted_labels.cuda()

    # 确保特征类型与模型权重类型一致
    model_dtype = next(model.parameters()).dtype
    if hasattr(config, 'opt_level') and config.opt_level != 'O0':
        pre_extracted_features = pre_extracted_features.half()
    else:
        pre_extracted_features = pre_extracted_features.float()

    # 确保特征类型与模型权重类型一致
    pre_extracted_features = pre_extracted_features.to(model_dtype)

    # 创建索引映射（如果有文件名）
    if 'filenames' in features_data and features_data['filenames'] is not None:
        filenames = features_data['filenames']
        # 创建文件名到索引的映射
        filename_to_idx = {filename: i for i, filename in enumerate(filenames)}

        # 创建数据集文件名到索引的映射
        dataset_filenames = [sample.get('file_id', f"sample_{i}") for i, sample in enumerate(val_loader.dataset)]
        dataset_to_feature_idx = [filename_to_idx.get(fname, -1) for fname in dataset_filenames]
    else:
        # 如果没有文件名，假设顺序匹配
        dataset_to_feature_idx = list(range(min(len(pre_extracted_features), len(val_loader.dataset))))

    logger.info(f"{config.num_clip * config.num_crop} views inference with pre-extracted features")

    # 获取模型权重数据类型，确保在循环内部可访问
    model_dtype = next(model.parameters()).dtype

    # 处理每个批次
    batch_idx = 0
    for batch_data in metric_logger.log_every(val_loader, config.print_freq, logger, header):
        label_id = batch_data["label"]
        label_id = label_id.reshape(-1)  # [b]
        b = label_id.size(0)

        # 获取对应的预提取特征
        batch_features = []
        batch_labels = []

        for i in range(b):
            idx = dataset_to_feature_idx[batch_idx * val_loader.batch_size + i]
            if idx >= 0 and idx < len(pre_extracted_features):
                batch_features.append(pre_extracted_features[idx])
                batch_labels.append(pre_extracted_labels[idx])
            else:
                # 如果找不到对应的特征，创建一个零特征
                feat_dim = pre_extracted_features.size(-1)
                num_frames = pre_extracted_features.size(1) if len(pre_extracted_features.size()) > 2 else 1
                batch_features.append(torch.zeros(num_frames, feat_dim, device='cuda'))
                batch_labels.append(label_id[i])

        batch_features = torch.stack(batch_features)  # [b, t, d]
        batch_labels = torch.stack(batch_labels)     # [b]

        # 处理多视图（如果有）
        tot_similarity = torch.zeros((b, num_classes)).cuda()

        # 模拟多视图处理
        n_views = config.num_clip * config.num_crop

        for view_idx in range(n_views):
            # 对于预提取特征，我们假设已经包含了时间信息
            # 这里只是简单地将特征平均来模拟多视图
            view_features = batch_features

            # 计算时间维度上的均值
            if len(view_features.size()) > 2:  # [b, t, d]
                view_features_mean = view_features.mean(dim=1)  # [b, d]
            else:  # [b, d]
                view_features_mean = view_features

            # 归一化特征
            view_features_mean = view_features_mean / view_features_mean.norm(dim=-1, keepdim=True)

            # 使用ViFi-CLIP的方式处理特征
            tokenized_prompts = model.module.tokenized_prompts  # (num_classes, token_len)
            prompts = model.module.prompt_learner()

            # 使用ViFi-CLIP的文本编码器处理文本特征
            text_features = model.module.text_encoder(prompts=prompts,
                                                    tokenized_prompts=tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logit_scale = logit_scale.to(model_dtype)
            view_features_mean = view_features_mean.to(model_dtype)
            text_features = text_features.to(model_dtype)
            # 计算logits
            logits = logit_scale * view_features_mean @ text_features.t()  # [b, n_cls]

            similarity = logits.softmax(dim=-1)
            tot_similarity += similarity

        # 平均所有视图的相似度
        tot_similarity = tot_similarity / n_views

        # Classification score
        acc1, acc5, indices_1, _ = accuracy_top1_top5(tot_similarity, batch_labels)
        metric_logger.meters['acc1'].update(float(acc1) / b * 100, n=b)
        metric_logger.meters['acc5'].update(float(acc5) / b * 100, n=b)

        # 更新类别级别的统计
        correct = (indices_1.squeeze() == batch_labels)
        for i in range(b):
            class_correct[batch_labels[i]] += correct[i].item()
            class_total[batch_labels[i]] += 1

        batch_idx += 1

    metric_logger.synchronize_between_processes()
    logger.info(f' * Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f}')

    # 计算并记录每个类别的准确率
    class_acc = class_correct / (class_total + 1e-7)
    class_acc = class_acc.cpu().numpy()
    class_names = val_loader.dataset.classes

    # 记录每个类别的准确率
    for i, (name, acc) in enumerate(zip(class_names, class_acc)):
        logger.info(f'{name}: {acc*100:.2f}%')

    # 返回包含类别准确率的字典
    stats = metric_logger.get_stats()
    stats['class_accuracy'] = zip(class_names, class_acc.tolist())
    return stats
