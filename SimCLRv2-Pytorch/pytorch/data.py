import functools
import logging
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

import data_util
from .config_args import args


def build_input_fn(builder, global_batch_size, is_training):
    def _input_fn(input_context):
        batch_size = global_batch_size // torch.distributed.get_world_size()
        logging.info(f'Global batch size: {global_batch_size}')
        logging.info(f'Per-replica batch size: {batch_size}')

        preprocess_fn_pretrain = get_preprocess_fn(is_training, is_pretrain=True)
        preprocess_fn_finetune = get_preprocess_fn(is_training, is_pretrain=False)

        num_classes = len(builder.classes)

        def map_fn(image, label):
            if is_training and args.train_mode == 'pretrain':
                xs = [preprocess_fn_pretrain(image) for _ in range(2)]
                image = torch.cat(xs, dim=1)
            else:
                image = preprocess_fn_finetune(image)
            label = F.one_hot(torch.tensor(label), num_classes=num_classes).float()
            return image, label

        logging.info('num_input_pipelines: %d', input_context.num_input_pipelines)

        if is_training:
            dataset = datasets.ImageFolder(builder.root, transform=None)
            sampler = DistributedSampler(dataset, shuffle=True)
        else:
            dataset = datasets.ImageFolder(builder.root, transform=None)
            sampler = DistributedSampler(dataset, shuffle=False)

        class MappedDataset(torch.utils.data.Dataset):
            def __init__(self, dataset):
                self.dataset = dataset

            def __len__(self):
                return len(self.dataset)

            def __getitem__(self, idx):
                image, label = self.dataset[idx]
                return map_fn(image, label)

        dataset = MappedDataset(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=is_training,
            num_workers=4,
            prefetch_factor=2,
            pin_memory=True
        )

        return dataloader

    return _input_fn


def build_distributed_dataset(builder, batch_size, is_training, strategy):
    return build_input_fn(builder, batch_size, is_training)(strategy)


def get_preprocess_fn(is_training, is_pretrain):
    test_crop = False if args.image_size <= 32 else True
    color_jitter_strength = args.color_jitter_strength if is_pretrain else 0.0

    return functools.partial(
        data_util.preprocess_image,
        height=args.image_size,
        width=args.image_size,
        is_training=is_training,
        color_jitter_strength=color_jitter_strength,
        test_crop=test_crop
    )


class DatasetBuilder:
    def __init__(self, root):
        self.root = root
        self.classes = sorted(os.listdir(root))  # If using classification-style structure

# Usage
#builder = DatasetBuilder(root='/path/to/classification/dataset')  # or use your regression version
#strategy = torch.distributed.get_rank()  # Dummy placeholder; replace with your actual strategy context

#train_loader = build_distributed_dataset(builder, batch_size=64, is_training=True, strategy=strategy)
#eval_loader = build_distributed_dataset(builder, batch_size=64, is_training=False, strategy=strategy)
