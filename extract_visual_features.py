"""
TC-CLIP Feature Extraction Script
Copyright (c) 2024-present NAVER Cloud Corp.
CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import torch
import torch.nn as nn
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import json
import logging
from tqdm import tqdm

from datasets.build import build_val_dataloader
from trainers.build_trainer import returnCLIP
from utils.tools import init_dist, get_dist_info, set_random_seed
from utils.logger import create_logger
from utils.print_utils import colorstr, print_configs


def extract_features(config: DictConfig, checkpoint_path: str, output_dir: str, logger: logging.Logger, extract_train: bool = False):
    """
    Extract visual features from videos using TC-CLIP model.

    Args:
        config: Configuration object
        checkpoint_path: Path to the model checkpoint
        output_dir: Directory to save extracted features
        logger: Logger instance
        extract_train: Whether to extract features from training data instead of validation data
    """
    # Initialize distributed training
    if config.distributed:
        init_dist()
        config.rank, config.world_size = get_dist_info()
    else:
        config.rank, config.world_size = 0, 1

    # Set random seed
    if config.seed is not None:
        set_random_seed(config.seed+config.rank, use_cudnn=config.use_cudnn)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Build model
    logger.info("Building model...")
    class_names = ["empty",]  # Placeholder for class names, not needed for feature extraction
    model = returnCLIP(config, logger, class_names)

    # Load checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.cuda()
    model.eval()

    # Build dataloader
    logger.info("Building dataloader...")
    if extract_train:
        from datasets.build import build_train_dataloader
        data, data_loader, _ = build_train_dataloader(logger, config)
        data_type = "train"
    else:
        data, data_loader, _ = build_val_dataloader(logger, config, config.data.val)
        data_type = "val"

    # Extract features
    logger.info("Extracting features...")
    all_features = []
    all_labels = []
    all_filenames = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, disable=config.rank != 0)):
            images = batch['imgs'].cuda(non_blocking=True)
            labels = batch['label']

            # Get features from vision encoder (ViFi-CLIP style)
            # Ensure the input type matches the model weight type
            model_dtype = next(model.image_encoder.parameters()).dtype
            image_features, _ = model.image_encoder(
                images.type(model_dtype),
                return_attention=False
            )

            # image_features shape: [B, T, D]
            features = image_features  # [B, T, D]

            # Save features
            all_features.append(features.cpu())
            all_labels.append(labels)

            # Save filenames if available
            if 'file_id' in batch:
                all_filenames.extend(batch['file_id'])

    # Concatenate all features
    all_features = torch.cat(all_features, dim=0)  # [N, T, D]
    all_labels = torch.cat(all_labels, dim=0)  # [N]

    # Save features
    output_file = output_path / f'{data_type}_features_rank{config.rank}.pth'
    torch.save({
        'features': all_features,
        'labels': all_labels,
        'filenames': all_filenames if all_filenames else None,
        'config': OmegaConf.to_container(config, resolve=True)
    }, output_file)

    logger.info(f"Features saved to {output_file}")

    # If using distributed training, gather all features
    if config.distributed and config.rank == 0:
        logger.info("Gathering features from all processes...")
        gather_features(output_path, config.world_size, logger)

    return output_file


def gather_features(output_path: Path, world_size: int, logger: logging.Logger):
    """
    Gather features from all distributed processes.

    Args:
        output_path: Directory where individual rank features are saved
        world_size: Number of processes
        logger: Logger instance
    """
    all_features = []
    all_labels = []
    all_filenames = []

    for rank in range(world_size):
        rank_file = output_path / f'features_rank{rank}.pth'
        if not rank_file.exists():
            logger.warning(f"Feature file for rank {rank} not found: {rank_file}")
            continue

        rank_data = torch.load(rank_file)
        all_features.append(rank_data['features'])
        all_labels.append(rank_data['labels'])

        if rank_data['filenames'] is not None:
            all_filenames.extend(rank_data['filenames'])

    # Concatenate all features
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Save gathered features
    gathered_file = output_path / 'features_gathered.pth'
    torch.save({
        'features': all_features,
        'labels': all_labels,
        'filenames': all_filenames if all_filenames else None,
        'num_samples': len(all_features)
    }, gathered_file)

    logger.info(f"Gathered features saved to {gathered_file}")

    # Save metadata
    metadata = {
        'num_samples': len(all_features),
        'feature_shape': list(all_features.shape),
        'num_classes': len(torch.unique(all_labels))
    }

    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Feature metadata: {metadata}")


@hydra.main(config_path="configs", config_name="fully_supervised")
def main(config: DictConfig) -> None:
    """
    Main function for feature extraction.

    Args:
        config: Configuration object
    """
    OmegaConf.set_struct(config, False)
    
    # Determine if extracting training data features
    extract_train = getattr(config, 'extract_train', False)
    
    # Set up output directory if not provided
    if not hasattr(config, 'output'):
        dataset_name = config.data.train.dataset_name if extract_train else config.data.val.dataset_name
        config.output = f"./features/{dataset_name}"

    # Create logger
    logger = create_logger(output_dir=config.output, dist_rank=0)
    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # Extract checkpoint path if not provided
    if not hasattr(config, 'resume') or config.resume is None:
        raise ValueError("checkpoint_path must be provided in the config")

    # Extract features
    extract_features(config, config.resume, config.output, logger, extract_train=extract_train)


if __name__ == "__main__":
    main()
