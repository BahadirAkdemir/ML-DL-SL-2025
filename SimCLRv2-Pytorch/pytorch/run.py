import glob
import os
import math
import json
import logging
import shutil

from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy

import pytorch.objective as obj_lib
import pytorch.metrics as metrics
from pytorch.config_args import args  # args loaded externally or merged manually
import pytorch.model as model_lib

from pytorch.metrics import MeanMetric, Top1Accuracy, Top5Accuracy

from FaceDataset import AgePredictionDataset


def build_saved_model(model, include_projection_head=True):
    """
    Returns a model ready for saving.
    In PyTorch, this typically just means having a consistent forward pass.
    """

    class SimCLRWrapper(torch.nn.Module):
        def __init__(self, model, include_projection_head):
            super().__init__()
            self.model = model
            self.include_projection_head = include_projection_head

        def forward(self, x):
            # Assume the model returns (features, projection)
            features, projection = self.model(x, return_projection=True)
            return projection if self.include_projection_head else features

    return SimCLRWrapper(model, include_projection_head)


def save(model, global_step, args):
    """
    Saves the model for fine-tuning and inference.
    It also keeps only the latest `args.keep_hub_module_max` checkpoints.
    """
    export_root = os.path.join(args.model_dir, 'saved_model')
    export_path = os.path.join(export_root, str(global_step))

    # Wrap model if projection head needs to be conditionally included
    wrapped_model = build_saved_model(model, include_projection_head=True)

    os.makedirs(export_path, exist_ok=True)
    model_path = os.path.join(export_path, 'model.pt')
    torch.save(wrapped_model.state_dict(), model_path)
    logging.info(f'Model saved to {model_path}')

    # Remove older exports if needed
    if getattr(args, 'keep_hub_module_max', 0) > 0:
        # Get all export subdirectories with numeric names
        all_exported = [d for d in os.listdir(export_root) if d.isdigit()]
        all_exported = sorted(all_exported, key=lambda x: int(x))
        to_delete = all_exported[:-args.keep_hub_module_max]
        for step in to_delete:
            dir_to_delete = os.path.join(export_root, step)
            shutil.rmtree(dir_to_delete)
            logging.info(f'Deleted old export: {dir_to_delete}')


def try_restore_from_checkpoint(model, optimizer, args):
    checkpoint_dir = args.model_dir
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    checkpoint_files.sort(key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)), reverse=True)

    latest_ckpt = os.path.join(checkpoint_dir, checkpoint_files[0]) if checkpoint_files else None

    if latest_ckpt:
        logging.info(f'Restoring from latest checkpoint: {latest_ckpt}')
        ckpt = torch.load(latest_ckpt, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        return ckpt.get('global_step', 0)

    elif args.checkpoint:
        logging.info(f'Restoring from checkpoint: {args.checkpoint}')
        ckpt = torch.load(args.checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        return ckpt.get('global_step', 0)

    logging.info('No checkpoint found.')
    return 0


def json_serializable(val):
    try:
        json.dumps(val)
        return True
    except (TypeError, OverflowError):
        return False


def list_checkpoints(model_dir):
    ckpt_paths = sorted(
        glob.glob(os.path.join(model_dir, 'checkpoint_*.pt')),
        key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0])
    )
    return ckpt_paths


def perform_evaluation(model, dataloader, eval_steps, ckpt_path, device, model_dir, num_classes=70):
    """Perform evaluation in PyTorch."""

    if args.train_mode == 'pretrain' and not args.lineareval_while_pretraining:
        logging.info('Skipping eval during pretraining without linear eval.')
        return

    if not os.path.exists(ckpt_path):
        logging.warning(f'Checkpoint not found at {ckpt_path}')
        return

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    global_step = checkpoint.get('global_step', 0)
    model.eval()

    logging.info(f'Performing evaluation at step {global_step}')
    summary_writer = SummaryWriter(model_dir)

    # Define metrics
    regularization_loss = 0.0
    top1_acc = MulticlassAccuracy(top_k=1, num_classes=num_classes).to(device)
    top5_acc = MulticlassAccuracy(top_k=5, num_classes=num_classes).to(device)
    model.to(device)

    step = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            if step >= eval_steps:
                break

            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            if isinstance(outputs, tuple):
                _, supervised_outputs = outputs
            else:
                supervised_outputs = outputs

            if supervised_outputs is None:
                continue

            top1_acc.update(supervised_outputs, labels.int())
            top5_acc.update(supervised_outputs, labels.int())

            reg_loss = model_lib.add_weight_decay(model, adjust_per_optimizer=True)
            regularization_loss += reg_loss.item()
            step += 1

    top1 = top1_acc.compute().item()
    top5 = top5_acc.compute().item()
    reg_loss_avg = regularization_loss / eval_steps

    print("Top 1:", top1)
    print("Top 5:", top5)

    # Log to TensorBoard
    summary_writer.add_scalar('eval/label_top_1_accuracy', top1, global_step)
    summary_writer.add_scalar('eval/label_top_5_accuracy', top5, global_step)
    summary_writer.add_scalar('eval/regularization_loss', reg_loss_avg, global_step)
    summary_writer.flush()

    result = {
        'eval/label_top_1_accuracy': top1,
        'eval/label_top_5_accuracy': top5,
        'eval/regularization_loss': reg_loss_avg,
        'global_step': global_step
    }

    logging.info(result)

    # Save result to JSON
    result_json_path = os.path.join(model_dir, f'result_{global_step}.json')
    with open(result_json_path, 'w') as f:
        json.dump(result, f, indent=2)

    return result


def train():
    # Basic params
    image_size = (args.channel_size, args.image_size, args.image_size)
    num_classes = args.num_classes
    num_train_examples = 1000  # TODO: Can be functionized
    num_eval_examples = 3000  # TODO: Can be functionized

    # Derived values
    steps_per_epoch = math.ceil(num_train_examples / args.train_batch_size)
    train_steps = args.train_epochs * steps_per_epoch
    eval_steps = args.eval_steps or math.ceil(num_eval_examples / args.eval_batch_size)
    checkpoint_steps = args.checkpoint_steps or (args.checkpoint_epochs * steps_per_epoch)

    print(f'# train examples: {num_train_examples}')
    print(f'# steps per epoch: {steps_per_epoch}')
    print(f'# total train steps: {train_steps}')
    print(f'# eval examples: {num_eval_examples}')
    print(f'# eval steps: {eval_steps}')

    # Device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model_lib.Model(num_classes).to(device)
    model.train()  # Explicitly set training mode

    # Datasets
    transform = transforms.Compose([
        transforms.Resize((args.training_image_size, args.training_image_size)),
        transforms.ToTensor(),
    ])

    # Datasets
    train_dataset = AgePredictionDataset(
        csv_path=args.train_csv_file,
        root_dir=args.train_root_folder,
        transform=transform
    )

    train_dataset = AgePredictionDataset(csv_path=args.train_csv_file, root_dir=args.train_root_folder,
                                         transform=transform)
    eval_dataset = AgePredictionDataset(csv_path=args.test_csv_file, root_dir=args.test_root_folder,
                                        transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size)

    # Evaluation-only mode
    if args.mode == 'eval':
        for ckpt in list_checkpoints(args.model_dir):
            result = perform_evaluation(model, eval_loader, eval_steps, ckpt, device, args.model_dir)
            if result['global_step'] >= train_steps:
                logging.info('Eval complete. Exiting...')
                return
        return

    # Training setup
    summary_writer = SummaryWriter(args.model_dir)
    learning_rate_fn = model_lib.WarmUpAndCosineDecay(args.learning_rate, num_train_examples)
    # Initialize with a larger learning rate and proper scaling
    optimizer = model_lib.build_optimizer(model.parameters(), args.learning_rate)

    # Print optimizer configuration
    print("\nOptimizer Configuration:")
    print(f"Optimizer type: {type(optimizer).__name__}")
    print(f"Number of parameter groups: {len(optimizer.param_groups)}")
    for i, param_group in enumerate(optimizer.param_groups):
        print(f"\nParameter group {i}:")
        print(f"Learning rate: {param_group['lr']}")
        print(f"Weight decay: {param_group['weight_decay']}")
        if 'momentum' in param_group:
            print(f"Momentum: {param_group['momentum']}")
        print(f"Number of parameters: {len(param_group['params'])}")

    # Metrics
    weight_decay_metric = MeanMetric('train/weight_decay')
    total_loss_metric = MeanMetric('train/total_loss')
    all_metrics = [weight_decay_metric, total_loss_metric]

    contrast_loss_metric = contrast_acc_metric = contrast_entropy_metric = None
    supervised_loss_metric = supervised_acc_metric = None

    if args.train_mode == 'pretrain':
        contrast_loss_metric = MeanMetric('train/contrast_loss')
        contrast_acc_metric = Top1Accuracy('train/contrast_acc')
        contrast_entropy_metric = MeanMetric('train/contrast_entropy')
        all_metrics += [contrast_loss_metric, contrast_acc_metric, contrast_entropy_metric]

    if args.train_mode == 'finetune' or args.lineareval_while_pretraining:
        supervised_loss_metric = MeanMetric('train/supervised_loss')
        supervised_acc_metric = Top1Accuracy('train/supervised_acc')
        all_metrics += [supervised_loss_metric, supervised_acc_metric]

    def save_checkpoint(model, optimizer, step, model_dir):
        checkpoint_path = os.path.join(model_dir, f'checkpoint_{step}.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': step
        }, checkpoint_path)
        logging.info(f'Saved checkpoint at {checkpoint_path}')

    global_step = 0
    epoch = 1
    while global_step < train_steps:
        logging.info(f"Starting Epoch {epoch}/{args.train_epochs}")

        step_bar = tqdm(train_loader, desc=f"Epoch {epoch}", position=0, leave=True, dynamic_ncols=True)

        for imgs, labels in step_bar:
            if global_step >= train_steps:
                break

            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            projection_head_outputs, supervised_head_outputs = model(imgs)

            loss = 0

            # Contrastive loss
            if projection_head_outputs is not None:
                con_loss, logits_con, labels_con = obj_lib.add_contrastive_loss(
                    projection_head_outputs,
                    hidden_norm=args.hidden_norm,
                    temperature=args.temperature
                )
                if global_step % 100 == 0:  # Print debug info every 20 steps
                    print(f"\nLoss Debug Info (step {global_step}):")
                    print(f"Contrastive loss: {con_loss.item():.6f}")
                    print(f"Projection outputs shape: {projection_head_outputs.shape}")
                    print(f"Logits shape: {logits_con.shape}")
                    print(f"Labels shape: {labels_con.shape}")
                    print(f"Projection outputs norm: {torch.norm(projection_head_outputs, dim=1).mean():.6f}")
                    print(f"Logits norm: {torch.norm(logits_con, dim=1).mean():.6f}")
                    print(f"Temperature: {args.temperature}")
                loss += con_loss
                if contrast_loss_metric:
                    metrics.update_pretrain_metrics_train(
                        contrast_loss_metric,
                        contrast_acc_metric,
                        contrast_entropy_metric,
                        con_loss, logits_con, labels_con
                    )

            # Supervised loss
            if supervised_head_outputs is not None:
                l = labels
                if args.train_mode == 'pretrain' and args.lineareval_while_pretraining:
                    num_transforms = imgs.shape[1] // 3
                    l = l.repeat_interleave(num_transforms, dim=0)
                sup_loss = obj_lib.add_supervised_loss(labels=l, logits=supervised_head_outputs)
                loss += sup_loss
                if supervised_loss_metric:
                    metrics.update_finetune_metrics_train(
                        supervised_loss_metric,
                        supervised_acc_metric,
                        sup_loss, l, supervised_head_outputs
                    )

            weight_decay = model_lib.add_weight_decay(model)
            loss += weight_decay
            weight_decay_metric.update_state(weight_decay)
            total_loss_metric.update_state(loss.item())

            loss.backward()

            # Increase gradient clipping threshold
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            # Detailed gradient debugging
            grad_info = {
                'total_norm': 0.0,
                'num_zero_grads': 0,
                'num_params': 0,
                'max_grad': float('-inf'),
                'min_grad': float('inf')
            }

            for name, p in model.named_parameters():
                if p.grad is not None:
                    grad_info['num_params'] += 1
                    param_norm = p.grad.data.norm(2)
                    grad_info['total_norm'] += param_norm.item() ** 2

                    # Check for zero gradients
                    if torch.allclose(p.grad, torch.zeros_like(p.grad)):
                        grad_info['num_zero_grads'] += 1

                    # Track max/min gradients
                    grad_info['max_grad'] = max(grad_info['max_grad'], p.grad.abs().max().item())
                    grad_info['min_grad'] = min(grad_info['min_grad'], p.grad.abs().min().item())

            grad_info['total_norm'] = grad_info['total_norm'] ** 0.5

            # Log detailed gradient information
            summary_writer.add_scalar('train/grad_norm', grad_info['total_norm'], global_step)
            summary_writer.add_scalar('train/zero_grads_ratio',
                                      grad_info['num_zero_grads'] / max(1, grad_info['num_params']), global_step)
            summary_writer.add_scalar('train/max_grad', grad_info['max_grad'], global_step)
            summary_writer.add_scalar('train/min_grad', grad_info['min_grad'], global_step)

            if global_step % 20 == 0:  # Print detailed info every 20 steps
                print(f"\nGradient Debug Info (step {global_step}):")
                print(f"Total gradient norm: {grad_info['total_norm']:.6f}")
                print(f"Zero gradients ratio: {grad_info['num_zero_grads']}/{grad_info['num_params']}")
                print(f"Max gradient: {grad_info['max_grad']:.6f}")
                print(f"Min gradient: {grad_info['min_grad']:.6f}")

            optimizer.step()

            # Update learning rate
            current_lr = learning_rate_fn(global_step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            global_step += 1
            step_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'step': f'{global_step}/{train_steps}',
                'lr': f'{current_lr:.6f}'
            })

            if global_step % checkpoint_steps == 0:
                logging.info(f'Checkpointing at step {global_step}')
                metrics.log_and_write_metrics_to_summary(all_metrics, global_step, summary_writer)
                summary_writer.add_scalar('learning_rate', learning_rate_fn(global_step), global_step)
                save_checkpoint(model, optimizer, global_step, args.model_dir)
                for metric in all_metrics:
                    metric.reset()

        epoch += 1

    logging.info('Training complete.')

    def latest_checkpoint(model_dir):
        checkpoints = list_checkpoints(model_dir)
        return checkpoints[-1] if checkpoints else None

    if args.mode == 'train_then_eval':
        perform_evaluation(model, eval_loader, eval_steps, latest_checkpoint(args.model_dir), device, args.model_dir)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm


def simple_train():

    # Basic params
    num_classes = args.num_classes

    batch_size = args.train_batch_size
    num_epochs = args.train_epochs
    learning_rate = args.learning_rate

    # Device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((args.training_image_size, args.training_image_size)),
        transforms.ToTensor(),
    ])

    # Datasets
    train_dataset = AgePredictionDataset(
        csv_path=args.train_csv_file,
        root_dir=args.train_root_folder,
        transform=transform
    )

    # builder = DatasetBuilder(root='/path/to/classification/dataset')  # or use your regression version
    # strategy = torch.distributed.get_rank()  # Dummy placeholder; replace with your actual strategy context

    # train_loader = build_distributed_dataset(builder, batch_size=64, is_training=True, strategy=strategy)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = model_lib.Model(num_classes=num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            images, labels = images.to(device), labels.to(device)
            loss = 0.0

            projection_head_outputs, supervised_head_outputs = model(images)

            if projection_head_outputs is not None:
                con_loss, logits_con, labels_con = obj_lib.add_contrastive_loss(
                    projection_head_outputs,
                    hidden_norm=args.hidden_norm,
                    temperature=args.temperature
                )
                loss += con_loss

            if supervised_head_outputs is not None:
                l = labels
                if args.train_mode == 'pretrain' and args.lineareval_while_pretraining:
                    num_transforms = images.shape[1] // 3
                    l = l.repeat_interleave(num_transforms, dim=0)
                sup_loss = obj_lib.add_supervised_loss(labels=l, logits=supervised_head_outputs)
                loss += sup_loss

                # Compute accuracy
                _, predicted = torch.max(supervised_head_outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total * 100 if total > 0 else 0.0
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    print("Training finished.")


if __name__ == '__main__':
    train()
    #simple_train()
