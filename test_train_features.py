
"""
测试训练集特征的分类准确率
基于 main_with_features.py 和 validate_training_features.py 创建
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from pathlib import Path
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.amp import autocast, GradScaler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from datasets.build import build_train_dataloader, build_val_dataloader
from datasets.blending import CutmixMixupBlending
from trainers.build_trainer import returnCLIP
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import epoch_saving, load_checkpoint, is_main, init_dist, get_dist_info, set_random_seed
from utils.logger import create_logger
from utils.print_utils import colorstr, print_configs

from engine_with_features import validate_with_features, train_with_features


@hydra.main(version_base=None, config_path="configs", config_name="fully_supervised")
def main(config: DictConfig) -> None:
    OmegaConf.set_struct(config, False)  # Needed to add fields at runtime below

    # Add features_path to config if not present
    if not hasattr(config, 'features_path'):
        config.features_path = None

    if config.eval is None and config.protocol in ['zero_shot', 'few_shot', 'base2novel', 'fully_supervised']:
        assert config.protocol in config.selected_option.data, "Selected data should be same with the protocol"
    if config.protocol == "few_shot":
        assert config.shot in [2, 4, 8, 16], "Number of shot 'config.shot' should be defined"
    if config.protocol == "base2novel":
        assert config.base in [1, 2, 3], "Base seed 'config.base' should be defined"

    # Force num_workers=4 in hmdb51
    if 'hmdb51' in config.selected_option.data:
        config.num_workers = 4

    # Init DDP
    if os.getenv('RANK') is None:
        raise Exception("This code only supports DDP mode. Try with DDP")
    init_dist()

    # Define working dir
    Path(config.output).mkdir(parents=True, exist_ok=True)

    # logger
    logger = create_logger(output_dir=config.output, dist_rank=dist.get_rank(), name=f"{config.trainer_name}")
    logger.info(f"working dir: {config.output}")

    config.rank, config.world_size = get_dist_info()
    config.num_gpus = config.world_size
    if config.num_gpus == 1:
        logger.info(colorstr('Single GPU'))
        config.distributed = False
    else:
        logger.info(colorstr('DDP')+f' with {config.num_gpus} GPUs')
        config.distributed = True

    # Random seed
    if config.seed is not None:
        set_random_seed(config.seed + config.rank, use_cudnn=config.use_cudnn)

    # Set accumulation steps
    config.accumulation_steps = config.total_batch_size // (config.num_gpus*config.batch_size)
    logger.info(f"Total batch size ({config.total_batch_size}) "
                f"= num_gpus ({config.num_gpus}) * batch_size ({config.batch_size}) "
                f"* accumulation_steps ({config.accumulation_steps})")

    # wandb logger
    if config.eval is not None or config.get('debug', False):
        config.use_wandb = False
    elif is_main() and config.use_wandb:
        os.environ["WANDB_API_KEY"] = config.wandb_api_key
        expr_name = os.path.split(config.output)[-1]
        tags = [f"{config.shot}shot" if config.protocol == "few_shot" else None,
                f"s{config.base}" if config.protocol == "base2novel" else None]
        tags = [t for t in tags if t is not None]
        tags.extend(config.get('wandb_tags', []))
        cfg_dict = OmegaConf.to_container(config, resolve=True)
        wandb.init(name=expr_name, project=config.wandb_project, dir=config.wandb_logging_dir,
                   config=cfg_dict, tags=tags)

    # print configs
    print_configs(logger, config)

    if config.eval == "train_features_test":
        main_training_features_test(logger, config)
    else:
        raise NotImplementedError("Only train_features_test mode is supported with this script")

    if is_main():
        wandb.finish()


def main_training_features_test(logger, config):
    # Check if features path is provided
    if not hasattr(config, 'features_path') or config.features_path is None:
        raise ValueError("features_path must be provided in the config when using pre-extracted features")

    # Check if training features path is provided
    if not hasattr(config, 'train_features_path') or config.train_features_path is None:
        raise ValueError("train_features_path must be provided in the config when testing training features")

    if config.protocol == 'fully_supervised' and config.multi_view_inference:
        config.num_clip = 4
        config.num_crop = 3
    elif config.protocol == 'zero_shot' and config.multi_view_inference:
        config.num_clip = 2

    if config.num_clip != 1 or config.num_crop != 1:
        logger.info(f"======== Testing with multi-view inference: "
                    f"{config.num_frames}x{config.num_clip}x{config.num_crop} ========")

    model, clip_model = None, None
    result_dict = {}

    # Build training dataloader
    train_data, train_loader, class_names = build_train_dataloader(logger, config)

    # Build model
    model, clip_model = returnCLIP(config, logger, class_names, return_clip_model=True)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.rank], broadcast_buffers=False,
                                                      find_unused_parameters=False)

    if config.resume:
        epoch_loaded, max_accuray_loaded = load_checkpoint(config, model, None, None, logger, model_only=True)
        logger.info(f"Loaded checkpoint at epoch {epoch_loaded} with max accuracy {max_accuray_loaded:.1f}")

    # Test training features
    logger.info(f"======== Start evaluation on training set with pre-extracted features =======")
    train_stats = train_with_features(train_loader, model, logger, config, config.train_features_path)

    # Test validation features if provided
    if hasattr(config, 'features_path') and config.features_path is not None:
        logger.info(f"======== Start evaluation on validation set with pre-extracted features =======")
        val_data, val_loader, _ = build_val_dataloader(logger, config)
        val_stats = validate_with_features(val_loader, model, logger, config, config.features_path)

        # Log comparison
        logger.info(f"======== Feature Classification Comparison ========")
        logger.info(f"Training set - Acc@1: {train_stats['acc1']:.1f}, Acc@5: {train_stats['acc5']:.1f}")
        logger.info(f"Validation set - Acc@1: {val_stats['acc1']:.1f}, Acc@5: {val_stats['acc5']:.1f}")

        # Save results
        result_dict = {
            'train': {'acc1': train_stats['acc1'], 'acc5': train_stats['acc5']},
            'val': {'acc1': val_stats['acc1'], 'acc5': val_stats['acc5']}
        }
    else:
        # Only training set results
        result_dict = {
            'train': {'acc1': train_stats['acc1'], 'acc5': train_stats['acc5']}
        }

    # Log to wandb
    if is_main() and config.use_wandb:
        wandb.log({
            'train/acc1': train_stats['acc1'],
            'train/acc5': train_stats['acc5']
        })

        if 'val' in result_dict:
            wandb.log({
                'val/acc1': val_stats['acc1'],
                'val/acc5': val_stats['acc5']
            })

            wandb.log({
                'comparison/train_val_acc1_diff': train_stats['acc1'] - val_stats['acc1'],
                'comparison/train_val_acc5_diff': train_stats['acc5'] - val_stats['acc5']
            })

    # Log class-wise accuracy
    if 'class_accuracy' in train_stats:
        logger.info("======== Training Set Class-wise Accuracy ========")
        for class_name, acc in train_stats['class_accuracy']:
            logger.info(f"{class_name}: {acc*100:.2f}%")

            if is_main() and config.use_wandb:
                wandb.log({f'train/class_acc/{class_name}': acc*100})

    if 'val' in result_dict and 'class_accuracy' in val_stats:
        logger.info("======== Validation Set Class-wise Accuracy ========")
        for class_name, acc in val_stats['class_accuracy']:
            logger.info(f"{class_name}: {acc*100:.2f}%")

            if is_main() and config.use_wandb:
                wandb.log({f'val/class_acc/{class_name}': acc*100})

    return


if __name__ == '__main__':
    main()
