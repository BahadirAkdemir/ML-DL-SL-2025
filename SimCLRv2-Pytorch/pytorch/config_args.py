# config.py
import argparse

parser = argparse.ArgumentParser()

# Learning settings
parser.add_argument('--learning_rate', type=float, default=0.3)
parser.add_argument('--learning_rate_scaling', type=str, choices=['linear', 'sqrt'], default='linear')
parser.add_argument('--warmup_epochs', type=float, default=2)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--batch_norm_decay', type=float, default=0.9)

# Training setup
parser.add_argument('--train_batch_size', type=int, default=64)
parser.add_argument('--train_split', type=str, default='train')
parser.add_argument('--train_epochs', type=int, default=1)
parser.add_argument('--train_steps', type=int, default=0)
parser.add_argument('--eval_steps', type=int, default=0)
parser.add_argument('--eval_batch_size', type=int, default=256)
parser.add_argument('--checkpoint_epochs', type=int, default=1)
parser.add_argument('--checkpoint_steps', type=int, default=0)
parser.add_argument('--eval_split', type=str, default='validation')

# Dataset
parser.add_argument('--dataset', type=str, default='imagenet2012')
parser.add_argument('--cache_dataset', action='store_true')

# Mode
parser.add_argument('--mode', type=str, choices=['train', 'eval', 'train_then_eval'], default='train_then_eval')
parser.add_argument('--train_mode', type=str, choices=['pretrain', 'finetune'], default='pretrain')
parser.add_argument('--lineareval_while_pretraining', action='store_true', default=True)

# Checkpointing
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--zero_init_logits_layer', action='store_true')
parser.add_argument('--fine_tune_after_block', type=int, default=-1)

# TPU / Distributed
parser.add_argument('--master', type=str, default=None)
parser.add_argument('--model_dir', type=str, default="")
parser.add_argument('--data_dir', type=str, default=None)
parser.add_argument('--use_tpu', action='store_true')
parser.add_argument('--tpu_name', type=str, default=None)
parser.add_argument('--tpu_zone', type=str, default=None)
parser.add_argument('--gcp_project', type=str, default=None)

# Optimizer
parser.add_argument('--optimizer', type=str, choices=['momentum', 'adam', 'lars'], default='adam')
parser.add_argument('--momentum', type=float, default=0.9)

# Eval
parser.add_argument('--eval_name', type=str, default=None)
parser.add_argument('--keep_checkpoint_max', type=int, default=5)
parser.add_argument('--keep_hub_module_max', type=int, default=1)

# Contrastive learning / Projections
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--hidden_norm', action='store_true')
parser.add_argument('--proj_head_mode', type=str, choices=['none', 'linear', 'nonlinear'], default='nonlinear')
parser.add_argument('--proj_out_dim', type=int, default=128)
parser.add_argument('--num_proj_layers', type=int, default=3)
parser.add_argument('--ft_proj_selector', type=int, default=0)

# Architecture
parser.add_argument('--global_bn', action='store_true')
parser.add_argument('--width_multiplier', type=int, default=1)
parser.add_argument('--resnet_depth', type=int, default=50)
parser.add_argument('--sk_ratio', type=float, default=0.0)
parser.add_argument('--se_ratio', type=float, default=0.0)
parser.add_argument('--image_size', type=int, default=224)
#parser.add_argument('--size', nargs='+', type=int, default=[8,3,224,224]) #for vit()

# Augmentations
parser.add_argument('--color_jitter_strength', type=float, default=1.0)
parser.add_argument('--use_blur', action='store_true')

# Distillation specific arguments
parser.add_argument('--teacher_checkpoint', type=str, default="",
                  help='Path to the teacher model checkpoint')
parser.add_argument('--temperature', type=float, default=1.0,
                  help='Temperature for knowledge distillation')
parser.add_argument('--model_dir', type=str, default='',
                  help='Directory to save student model checkpoints')
parser.add_argument('--save_freq', type=int, default=10,
                  help='Frequency of saving checkpoints (in epochs)')


parser.add_argument('--train_csv_file', type=str, default="",
                  help='If you use custom (mine) dataset. add train.csv path')
parser.add_argument('--test_csv_file', type=str, default="",
                  help='If you use custom (mine) dataset. add test.csv path')

parser.add_argument('--train_root_folder', type=str, default="",
                  help='If you use custom (mine) dataset, add image locations. Else add the root train folder')
parser.add_argument('--test_root_folder', type=str, default="",
                  help='If you use custom (mine) dataset, add image locations. Else add the root test folder')
parser.add_argument('--channel_size', type=int, default=3)
parser.add_argument('--num_classes', type=int, default=70)
parser.add_argument('--training_image_size', type=int, default=224)


args = parser.parse_args()
