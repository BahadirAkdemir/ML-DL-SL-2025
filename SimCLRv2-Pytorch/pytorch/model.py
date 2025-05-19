import math

from . import data_util
from .resnet import *
import pytorch.lars_optimizer as l
import torch
import torch.nn as nn
import torch.nn.functional as F
from config_args import args


def build_optimizer(params, learning_rate):
    if args.optimizer == 'momentum':
        return torch.optim.SGD(params, lr=learning_rate, momentum=args.momentum, nesterov=True)
    elif args.optimizer == 'adam':
        return torch.optim.Adam(params, lr=learning_rate)
    elif args.optimizer == 'lars':
        return l.LARSOptimizer(params=params,
                             learning_rate=learning_rate,
                             momentum=args.momentum,
                             weight_decay=args.weight_decay,
                             exclude_from_weight_decay=[
                                 'batch_normalization', 'bias', 'head_supervised'
                             ])
    else:
        raise ValueError(f'Unknown optimizer {args.optimizer}')


def add_weight_decay(model, adjust_per_optimizer=True):
    """
    Compute weight decay manually for specific parameters.

    Args:
        model (torch.nn.Module): The model to compute weight decay for.
        adjust_per_optimizer (bool): If True, respects LARS-specific behavior.

    Returns:
        torch.Tensor: The L2 regularization loss.
    """
    if adjust_per_optimizer and 'lars' in args.optimizer.lower():
        # Apply weight decay only to "head_supervised" and not to biases
        l2_losses = [
            torch.norm(param, p=2) ** 2 / 2  # equivalent to tf.nn.l2_loss
            for name, param in model.named_parameters()
            if 'head_supervised' in name and 'bias' not in name and param.requires_grad
        ]
        if l2_losses:
            return args.weight_decay * sum(l2_losses)
        else:
            return torch.tensor(0.0, device=next(model.parameters()).device)

    # Default: Apply weight decay to all except batch norm
    l2_losses = [
        torch.norm(param, p=2) ** 2 / 2
        for name, param in model.named_parameters()
        if 'batchnorm' not in name.lower() and 'bn' not in name.lower() and param.requires_grad
    ]
    if l2_losses:
        return args.weight_decay * sum(l2_losses)
    else:
        return torch.tensor(0.0, device=next(model.parameters()).device)



def get_train_steps(num_examples):
    return args.train_steps or (num_examples * args.train_epochs // args.train_batch_size + 1)


class WarmUpAndCosineDecay:
    def __init__(self, base_lr, num_examples):
        self.base_lr = base_lr
        self.num_examples = num_examples
        self.total_steps = get_train_steps(num_examples)
        self.warmup_steps = int(round(args.warmup_epochs * num_examples / args.train_batch_size))

    def __call__(self, step):
        if args.learning_rate_scaling == 'linear':
            scaled_lr = self.base_lr * args.train_batch_size / 256.
        elif args.learning_rate_scaling == 'sqrt':
            scaled_lr = self.base_lr * math.sqrt(args.train_batch_size)
        else:
            raise ValueError(f'Unknown learning rate scaling {args.learning_rate_scaling}')

        learning_rate = (step / float(self.warmup_steps) * scaled_lr if self.warmup_steps else scaled_lr)

        if step < self.warmup_steps:
            return learning_rate
        else:
            decay_steps = self.total_steps - self.warmup_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * (step - self.warmup_steps) / decay_steps))
            return scaled_lr * cosine_decay


class LinearLayer(nn.Module):

    def __init__(self,
                 num_classes,
                 use_bias=True,
                 use_bn=False,
                 name='linear_layer',
                 **kwargs):
        # Note: use_bias is ignored for the linear layer when use_bn=True.
        # However, it is still used for batch norm.
        super(LinearLayer, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.use_bias = use_bias
        self.use_bn = use_bn
        self._name = name
        if self.use_bn:
            # num_features will be set dynamically in build()
            self.bn_relu = None
        self.linear = None

    def build(self, input_shape):
        if callable(self.num_classes):
            num_classes = self.num_classes(input_shape)
        else:
            num_classes = self.num_classes

        input_dim = input_shape[1]
        self.linear = nn.Linear(
            input_dim,
            num_classes,
            bias=(self.use_bias and not self.use_bn)
        )

        nn.init.normal_(self.linear.weight, std=0.01)

        if self.use_bn:
            self.bn_relu = BatchNormRelu(num_features=num_classes, relu=False, center=self.use_bias)

    def forward(self, inputs, training=False):
        assert inputs.dim() == 2, f"Expected 2D input, got shape {inputs.shape}"

        if self.linear is None:
            self.build(inputs.shape)

        self.to(inputs.device) # TODO: Find a way to fix (device_problem)
        inputs = self.linear(inputs)

        if self.use_bn:
            inputs = self.bn_relu(inputs)

        return inputs

class ProjectionHead(nn.Module):

    def __init__(self):
        super(ProjectionHead, self).__init__()
        out_dim = args.proj_out_dim
        self.linear_layers = nn.ModuleList()

        if args.proj_head_mode == 'none':
            pass  # directly use the output hiddens as hiddens

        elif args.proj_head_mode == 'linear':
            self.linear_layers.append(
                LinearLayer(
                    num_classes=out_dim,
                    use_bias=False,
                    use_bn=True,
                    name='l_0'
                )
            )

        elif args.proj_head_mode == 'nonlinear':
            for j in range(args.num_proj_layers):
                if j != args.num_proj_layers - 1:
                    # for the middle layers, use bias and relu for the output.
                    self.linear_layers.append(
                        LinearLayer(
                            num_classes=lambda shape: shape[1],
                            use_bias=True,
                            use_bn=True,
                            name=f'nl_{j}'
                        )
                    )
                else:
                    # for the final layer, neither bias nor relu is used.
                    self.linear_layers.append(
                        LinearLayer(
                            num_classes=args.proj_out_dim,
                            use_bias=False,
                            use_bn=True,
                            name=f'nl_{j}'
                        )
                    )
        else:
            raise ValueError(f'Unknown head projection mode {args.proj_head_mode}')

    def forward(self, inputs, training=False):
        if args.proj_head_mode == 'none':
            return inputs

        hiddens_list = [inputs]  # Equivalent to tf.identity(inputs, 'proj_head_input')

        if args.proj_head_mode == 'linear':
            assert len(self.linear_layers) == 1, f"Expected 1 linear layer, got {len(self.linear_layers)}"
            out = self.linear_layers[0](hiddens_list[-1], training=training)
            hiddens_list.append(out)

        elif args.proj_head_mode == 'nonlinear':
            for j in range(args.num_proj_layers):
                layer = self.linear_layers[j]
                # Lazy build if needed
                if hasattr(layer, 'build') and layer.linear is None:
                    layer.build(hiddens_list[-1].shape)
                out = layer(hiddens_list[-1], training=training)
                if j != args.num_proj_layers - 1:
                    out = F.relu(out)
                hiddens_list.append(out)

        else:
            raise ValueError(f'Unknown head projection mode {args.proj_head_mode}')

        proj_head_output = hiddens_list[-1]  # Equivalent to tf.identity(..., 'proj_head_output')
        ft_input = hiddens_list[args.ft_proj_selector]
        return proj_head_output, ft_input

class SupervisedHead(nn.Module):
    def __init__(self, num_classes, name='head_supervised'):
        """
        Supervised classification head.

        Args:
            num_classes (int): Number of output classes.
            name (str): Layer name (for compatibility; not used in PyTorch directly).
        """
        super(SupervisedHead, self).__init__()
        self._name = name  # Name is kept for consistency/logging
        self.linear_layer = LinearLayer(num_classes=num_classes)

    def forward(self, x):
        x = self.linear_layer(x)
        return x  # Equivalent to tf.identity with name 'logits_sup'


class Model(nn.Module):
    """ResNet model with projection or supervised head."""

    def __init__(self, num_classes):
        super(Model, self).__init__()
        self.train_mode = args.train_mode
        self.lineareval_while_pretraining = args.lineareval_while_pretraining
        self.use_blur = args.use_blur

        self.resnet_model = resnet(resnet_depth=args.resnet_depth, width_multiplier=args.width_multiplier)
        #self.vit_model = vit()
        self._projection_head = ProjectionHead()

        if self.train_mode == 'finetune' or self.lineareval_while_pretraining:
            self.supervised_head = SupervisedHead(num_classes)

    def forward(self, inputs):
        features = inputs  # shape: (B, H, W, C) or (B, C, H, W) if already in PyTorch format

        if self.training and self.train_mode == 'pretrain':
            if args.fine_tune_after_block > -1:
                raise ValueError(
                    'Does not support layer freezing during pretraining, '
                    'should set fine_tune_after_block <= -1 for safety.')

        if inputs.shape[1] is None:
            raise ValueError(
                f'The input channels dimension must be statically known (got shape {inputs.shape})')

        # Infer number of transforms: channels // 3
        # Assumes input shape is (B, C, H, W) in PyTorch format
        c = inputs.shape[1]
        if c % 3 != 0:
            raise ValueError(f"Expected input channels to be multiple of 3, got {c}")
        num_transforms = c // 3

        # Split input into multiple augmented views
        features_list = torch.chunk(features, num_transforms, dim=1)  # split along channel dimension

        if self.use_blur and self.training and self.train_mode == 'pretrain':
            features_list = data_util.batch_random_blur(
                features_list,
                height=args.image_size,
                width=args.image_size
            )

        features = torch.cat(features_list, dim=0)  # shape: (num_transforms * B, C, H, W)

        # Base network forward pass
        hiddens = self.resnet_model(features)
        #hiddens = self.vit_model(features)

        # Projection head
        projection_head_outputs, supervised_head_inputs = self._projection_head(hiddens)

        if self.train_mode == 'finetune':
            supervised_outputs = self.supervised_head(supervised_head_inputs)
            return None, supervised_outputs

        elif self.train_mode == 'pretrain' and self.lineareval_while_pretraining:
            # Stop gradients from flowing into backbone
            supervised_inputs = supervised_head_inputs.detach()
            supervised_outputs = self.supervised_head(supervised_inputs)
            return projection_head_outputs, supervised_outputs

        else:
            return projection_head_outputs, None
