import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from config_args import args  # simulate FLAGS using global args

BATCH_NORM_EPSILON = 1e-5

class BatchNormRelu(nn.Module):
    def __init__(self,
                 relu=True,
                 init_zero=False,
                 center=True,
                 scale=True,
                 **kwargs):
        super(BatchNormRelu, self).__init__()
        self.relu = relu
        self.center = center
        self.scale = scale
        self.gamma_initializer = 'zeros' if init_zero else 'ones'

        self.bn = None
        self._built = False

    def build(self, inputs):
        num_features = inputs.shape[1]

        affine = self.center or self.scale

        if inputs.dim() == 4:
            self.bn = nn.BatchNorm2d(num_features, eps=BATCH_NORM_EPSILON,
                                     momentum=args.batch_norm_decay,
                                     affine=affine)
        elif inputs.dim() in [2, 3]:
            self.bn = nn.BatchNorm1d(num_features, eps=BATCH_NORM_EPSILON,
                                     momentum=args.batch_norm_decay,
                                     affine=affine)
        else:
            raise ValueError(f"Unsupported input dimension {inputs.dim()} for BatchNormRelu.")

        if affine:
            if self.gamma_initializer == 'zeros':
                nn.init.zeros_(self.bn.weight)
            else:
                nn.init.ones_(self.bn.weight)

        self._built = True

    def forward(self, inputs, training=False):
        if not self._built:
            self.build(inputs)

        self.bn.train(training)
        x = inputs

        self.to(x.device)  # TODO: Find a way to fix (device_problem)
        x = self.bn(x)

        if self.relu:
            x = F.relu(x, inplace=True)

        return x

class DropBlock(nn.Module):  # pylint: disable=missing-docstring

    def __init__(self,
                 keep_prob,
                 dropblock_size,
                 **kwargs):
        super(DropBlock, self).__init__(**kwargs)
        self.keep_prob = keep_prob
        self.dropblock_size = dropblock_size

    def forward(self, net):
        keep_prob = self.keep_prob
        dropblock_size = self.dropblock_size

        if keep_prob is None:
            return net

        logging.info(
            f'Applying DropBlock: dropblock_size {dropblock_size}, net.shape {net.shape}')

        _, _, width, height = net.shape

        if width != height:
            raise ValueError('Input tensor with width != height is not supported.')

        dropblock_size = min(dropblock_size, width)
        gamma = ((1.0 - keep_prob) * width ** 2) / (
                dropblock_size ** 2 * (width - dropblock_size + 1) ** 2)

        # Generate mask
        mask = torch.ones((net.shape[0], 1, width, height), device=net.device)
        rand_noise = torch.rand(net.shape, device=net.device)

        block_center = torch.zeros((1, 1, width, height), dtype=torch.bool, device=net.device)
        mid = dropblock_size // 2
        valid_start = mid
        valid_end = width - (dropblock_size - 1) // 2

        block_center[:, :, valid_start:valid_end, valid_start:valid_end] = True

        block_pattern = (
                                (~block_center).float()
                                + (1 - gamma)
                                + rand_noise
                        ) >= 1.0
        block_pattern = block_pattern.float()

        if dropblock_size == width:
            block_pattern = torch.amin(block_pattern, dim=(2, 3), keepdim=True)
        else:
            k = dropblock_size
            block_pattern = -F.max_pool2d(
                -block_pattern,
                kernel_size=(k, k),
                stride=1,
                padding=k // 2
            )

        percent_ones = block_pattern.sum() / block_pattern.numel()
        net = net / percent_ones * block_pattern

        return net

class FixedPadding(nn.Module):  # pylint: disable=missing-docstring

    def __init__(self, kernel_size, **kwargs):
        super(FixedPadding, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def forward(self, inputs):
        kernel_size = self.kernel_size
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))

        return padded_inputs

class Conv2dFixedPadding(nn.Module):  # pylint: disable=missing-docstring

    def __init__(self,
                 filters,
                 kernel_size,
                 strides,
                 **kwargs):
        super(Conv2dFixedPadding, self).__init__(**kwargs)
        self.fixed_padding = FixedPadding(kernel_size) if strides > 1 else None

        padding = 0 if strides > 1 else kernel_size // 2  # SAME padding if stride == 1

        self.conv2d = nn.LazyConv2d(
            out_channels=filters,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,
            bias=False
        )

        # Delay in_channels setup until input is available
        self.inited = False
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

    def forward(self, inputs):
        if not self.inited:
            in_channels = inputs.shape[1]
            self.conv2d.in_channels = in_channels
            self.conv2d = nn.Conv2d(
                in_channels=in_channels,
                out_channels=self.filters,
                kernel_size=self.kernel_size,
                stride=self.strides,
                padding=(0 if self.strides > 1 else self.kernel_size // 2),
                bias=False
            )
            nn.init.kaiming_normal_(self.conv2d.weight, mode='fan_out', nonlinearity='relu')
            self.inited = True

        if self.fixed_padding is not None:
            inputs = self.fixed_padding(inputs)
        self.to(inputs.device)  # TODO: Find a way to fix (device_problem)
        outputs = self.conv2d(inputs)

        return outputs

class IdentityLayer(nn.Module):
    def forward(self, inputs):
        return inputs.clone()

class SK_Conv2D(nn.Module):  # pylint: disable=invalid-name
    """Selective kernel convolutional layer (https://arxiv.org/abs/1903.06586)."""

    def __init__(self,
                 filters,
                 strides,
                 sk_ratio,
                 min_dim=32,
                 **kwargs):
        super(SK_Conv2D, self).__init__(**kwargs)
        self.filters = filters
        self.sk_ratio = sk_ratio
        self.min_dim = min_dim

        self.conv2d_fixed_padding = Conv2dFixedPadding(
            filters=2 * filters,
            kernel_size=3,
            strides=strides,
            )

        self.batch_norm_relu = BatchNormRelu()

        mid_dim = max(int(filters * sk_ratio), min_dim)

        self.conv2d_0 = nn.Conv2d(
            in_channels=2 * filters,
            out_channels=mid_dim,
            kernel_size=1,
            stride=1,
            bias=False)

        self.batch_norm_relu_1 = BatchNormRelu()

        self.conv2d_1 = nn.Conv2d(
            in_channels=mid_dim,
            out_channels=2 * filters,
            kernel_size=1,
            stride=1,
            bias=False)

        self.inited = False

    def forward(self, inputs, training=True):
        channel_axis = 1
        pooling_axes = [2, 3]

        # Two stream convs
        out = self.conv2d_fixed_padding(inputs)
        out = self.batch_norm_relu(out)

        # Split into 2 along channel axis: shape -> [2, N, C//2, H, W]
        split_out = torch.stack(torch.chunk(out, 2, dim=channel_axis), dim=0)

        # Global descriptor
        global_feat = torch.sum(split_out, dim=0)  # shape: [N, C, H, W]
        global_feat = torch.mean(global_feat, dim=pooling_axes, keepdim=True)

        # Mixing weights
        mix = self.conv2d_0(global_feat)
        mix = self.batch_norm_relu_1(mix)
        mix = self.conv2d_1(mix)

        mix_split = torch.stack(torch.chunk(mix, 2, dim=channel_axis), dim=0)
        mix_softmax = F.softmax(mix_split, dim=0)

        out = torch.sum(split_out * mix_softmax, dim=0)

        return out

class SE_Layer(nn.Module):  # pylint: disable=invalid-name
    """Squeeze and Excitation layer (https://arxiv.org/abs/1709.01507)."""

    def __init__(self, filters, se_ratio, **kwargs):
        super(SE_Layer, self).__init__(**kwargs)
        reduced_channels = max(1, int(filters * se_ratio))

        self.se_reduce = nn.Conv2d(
            in_channels=filters,
            out_channels=reduced_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

        self.se_expand = nn.Conv2d(
            in_channels=reduced_channels,
            out_channels=filters,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )

    def forward(self, inputs):

        # Global average pooling: [N, C, H, W] → [N, C, 1, 1]
        se_tensor = F.adaptive_avg_pool2d(inputs, output_size=1)
        se_tensor = self.se_reduce(se_tensor)
        se_tensor = F.relu(se_tensor)
        se_tensor = self.se_expand(se_tensor)
        se_tensor = torch.sigmoid(se_tensor)

        out = inputs * se_tensor  # broadcast multiply

        return out

class ResidualBlock(nn.Module):  # pylint: disable=missing-docstring

    def __init__(self,
                 filters,
                 strides,
                 use_projection=False,
                 dropblock_keep_prob=None,
                 dropblock_size=None,
                 **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        del dropblock_keep_prob
        del dropblock_size

        self.shortcut_layers = nn.ModuleList()
        self.conv2d_bn_layers = nn.ModuleList()

        if use_projection:
            if args.sk_ratio > 0:  # Use ResNet-D shortcut
                if strides > 1:
                    self.shortcut_layers.append(FixedPadding(2))
                self.shortcut_layers.append(
                    nn.AvgPool2d(
                        kernel_size=2,
                        stride=strides,
                        padding=0 if strides > 1 else 1,  # same padding effect
                    )
                )
                self.shortcut_layers.append(
                    Conv2dFixedPadding(
                        filters=filters,
                        kernel_size=1,
                        strides=1,
                        
                    )
                )
            else:
                self.shortcut_layers.append(
                    Conv2dFixedPadding(
                        filters=filters,
                        kernel_size=1,
                        strides=strides,
                        
                    )
                )
            self.shortcut_layers.append(
                BatchNormRelu(relu=False, )
            )

        self.conv2d_bn_layers.append(
            Conv2dFixedPadding(
                filters=filters,
                kernel_size=3,
                strides=strides,
                
            )
        )
        self.conv2d_bn_layers.append(
            BatchNormRelu()
        )
        self.conv2d_bn_layers.append(
            Conv2dFixedPadding(
                filters=filters,
                kernel_size=3,
                strides=1,
                
            )
        )
        self.conv2d_bn_layers.append(
            BatchNormRelu(relu=False, init_zero=True, )
        )

        self.use_se = args.se_ratio > 0
        if self.use_se:
            self.se_layer = SE_Layer(filters, args.se_ratio, )

    def forward(self, inputs, training=True):
        shortcut = inputs

        for layer in self.shortcut_layers:
            shortcut = layer(shortcut)

        out = inputs
        for layer in self.conv2d_bn_layers:
            out = layer(out)

        if self.use_se:
            out = self.se_layer(out)

        out += shortcut
        out = F.relu(out)
        return out

class BottleneckBlock(nn.Module):  # """BottleneckBlock."""

    def __init__(self,
                 filters,
                 strides,
                 use_projection=False,
                 dropblock_keep_prob=None,
                 dropblock_size=None,
                 **kwargs):
        super(BottleneckBlock, self).__init__(**kwargs)


        self.projection_layers = nn.ModuleList()
        if use_projection:
            filters_out = 4 * filters
            if args.sk_ratio > 0:
                if strides > 1:
                    self.projection_layers.append(FixedPadding(2))
                self.projection_layers.append(
                    nn.AvgPool2d(
                        kernel_size=2,
                        stride=strides,
                        padding=0 if strides > 1 else 1
                    )
                )
                self.projection_layers.append(
                    Conv2dFixedPadding(
                        filters=filters_out,
                        kernel_size=1,
                        strides=1,
                        
                    )
                )
            else:
                self.projection_layers.append(
                    Conv2dFixedPadding(
                        filters=filters_out,
                        kernel_size=1,
                        strides=strides,
                        
                    )
                )
            self.projection_layers.append(
                BatchNormRelu(relu=False, )
            )

        self.shortcut_dropblock = DropBlock(
            keep_prob=dropblock_keep_prob,
            dropblock_size=dropblock_size,
            
        )

        self.conv_relu_dropblock_layers = nn.ModuleList()

        self.conv_relu_dropblock_layers.append(
            Conv2dFixedPadding(filters=filters, kernel_size=1, strides=1, )
        )
        self.conv_relu_dropblock_layers.append(
            BatchNormRelu()
        )
        self.conv_relu_dropblock_layers.append(
            DropBlock(
                keep_prob=dropblock_keep_prob,
                dropblock_size=dropblock_size,
                
            )
        )

        if args.sk_ratio > 0:
            self.conv_relu_dropblock_layers.append(
                SK_Conv2D(filters, strides, args.sk_ratio, )
            )
        else:
            self.conv_relu_dropblock_layers.append(
                Conv2dFixedPadding(filters=filters, kernel_size=3, strides=strides, )
            )
            self.conv_relu_dropblock_layers.append(
                BatchNormRelu()
            )

        self.conv_relu_dropblock_layers.append(
            DropBlock(
                keep_prob=dropblock_keep_prob,
                dropblock_size=dropblock_size,
                
            )
        )

        self.conv_relu_dropblock_layers.append(
            Conv2dFixedPadding(filters=4 * filters, kernel_size=1, strides=1, )
        )
        self.conv_relu_dropblock_layers.append(
            BatchNormRelu(relu=False, init_zero=True, )
        )
        self.conv_relu_dropblock_layers.append(
            DropBlock(
                keep_prob=dropblock_keep_prob,
                dropblock_size=dropblock_size,
                
            )
        )

        self.use_se = args.se_ratio > 0
        if self.use_se:
            self.se_layer = SE_Layer(filters, args.se_ratio, )

    def forward(self, inputs, training=True):
        shortcut = inputs

        for layer in self.projection_layers:
            shortcut = layer(shortcut)
        shortcut = self.shortcut_dropblock(shortcut)

        out = inputs
        for layer in self.conv_relu_dropblock_layers:
            out = layer(out)

        if self.use_se:
            out = self.se_layer(out)

        out += shortcut
        out = F.relu(out)
        return out

class BlockGroup(nn.Module):  # pylint: disable=missing-docstring

    def __init__(self,
                 filters,
                 block_fn,
                 blocks,
                 strides,
                 dropblock_keep_prob=None,
                 dropblock_size=None,
                 **kwargs):
        super(BlockGroup, self).__init__()
        self._name = kwargs.get('name')

        layers = []
        layers.append(
            block_fn(
                filters,
                strides,
                use_projection=True,
                dropblock_keep_prob=dropblock_keep_prob,
                dropblock_size=dropblock_size
            )
        )

        for _ in range(1, blocks):
            layers.append(
                block_fn(
                    filters,
                    1,
                    use_projection=False,
                    dropblock_keep_prob=dropblock_keep_prob,
                    dropblock_size=dropblock_size
                )
            )

        self.layers = nn.ModuleList(layers)

    def forward(self, inputs):
        out = inputs
        for layer in self.layers:
            out = layer(out)

        # Mimic tf.identity(out, name) (name used for tracking only in PyTorch)
        return out.clone()

class Resnet(nn.Module):  # pylint: disable=missing-docstring

    def __init__(self,
                 block_fn,
                 layers,
                 width_multiplier,
                 dropblock_keep_probs=None,
                 dropblock_size=None,
                 **kwargs):
        super(Resnet, self).__init__(**kwargs)

        if dropblock_keep_probs is None:
            dropblock_keep_probs = [None] * 4
        if not isinstance(dropblock_keep_probs, list) or len(dropblock_keep_probs) != 4:
            raise ValueError('dropblock_keep_probs is not valid:', dropblock_keep_probs)

        trainable = (
                args.train_mode != 'finetune' or args.fine_tune_after_block == -1
        )

        self.initial_conv_relu_max_pool = nn.ModuleList()

        if args.sk_ratio > 0:
            self.initial_conv_relu_max_pool.append(
                Conv2dFixedPadding(
                    filters=64 * width_multiplier // 2,
                    kernel_size=3,
                    strides=2,
                )
            )
            self.initial_conv_relu_max_pool.append(
                BatchNormRelu()
            )
            self.initial_conv_relu_max_pool.append(
                Conv2dFixedPadding(
                    filters=64 * width_multiplier // 2,
                    kernel_size=3,
                    strides=1,
                )
            )
            self.initial_conv_relu_max_pool.append(
                BatchNormRelu()
            )
            self.initial_conv_relu_max_pool.append(
                Conv2dFixedPadding(
                    filters=64 * width_multiplier,
                    kernel_size=3,
                    strides=1,
                )
            )
        else:
            self.initial_conv_relu_max_pool.append(
                Conv2dFixedPadding(
                    filters=64 * width_multiplier,
                    kernel_size=7,
                    strides=2,
                )
            )
        self.initial_conv_relu_max_pool.append(
            IdentityLayer()
        )
        self.initial_conv_relu_max_pool.append(
            BatchNormRelu()
        )
        self.initial_conv_relu_max_pool.append(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.initial_conv_relu_max_pool.append(
            IdentityLayer()
        )

        self.block_groups = nn.ModuleList()

        def add_block_group(index, filters, blocks, strides):
            nonlocal trainable
            if args.train_mode == 'finetune' and args.fine_tune_after_block == index:
                trainable = True
            block_group = BlockGroup(
                filters=filters,
                block_fn=block_fn,
                blocks=blocks,
                strides=strides,
                dropblock_keep_prob=dropblock_keep_probs[index],
                dropblock_size=dropblock_size,
                name=f'block_group{index + 1}'
            )
            self.block_groups.append(block_group)

        add_block_group(0, 64 * width_multiplier, layers[0], strides=1)
        add_block_group(1, 128 * width_multiplier, layers[1], strides=2)
        add_block_group(2, 256 * width_multiplier, layers[2], strides=2)
        add_block_group(3, 512 * width_multiplier, layers[3], strides=2)

    def forward(self, inputs):
        out = inputs
        for layer in self.initial_conv_relu_max_pool:
            out = layer(out)

        for i, block_group in enumerate(self.block_groups):
            if args.train_mode == 'finetune' and args.fine_tune_after_block == i:
                out = out.detach()  # stop gradient at finetune
            out = block_group(out)

        if args.train_mode == 'finetune' and args.fine_tune_after_block == 4:
          out = out.detach()

        out = torch.mean(out, dim=(2, 3))  # Global average pool NCHW

        return out.clone()  # tf.identity(out, 'final_avg_pool')

def resnet(resnet_depth,
           width_multiplier,
           dropblock_keep_probs=None,
           dropblock_size=None):
    """Returns the ResNet model for a given size and number of output classes."""
    model_params = {
        18: {
            'block': ResidualBlock,
            'layers': [2, 2, 2, 2]
        },
        34: {
            'block': ResidualBlock,
            'layers': [3, 4, 6, 3]
        },
        50: {
            'block': BottleneckBlock,
            'layers': [3, 4, 6, 3]
        },
        101: {
            'block': BottleneckBlock,
            'layers': [3, 4, 23, 3]
        },
        152: {
            'block': BottleneckBlock,
            'layers': [3, 8, 36, 3]
        },
        200: {
            'block': BottleneckBlock,
            'layers': [3, 24, 36, 3]
        }
    }

    if resnet_depth not in model_params:
        raise ValueError('Not a valid resnet_depth:', resnet_depth)

    params = model_params[resnet_depth]
    return Resnet(
        params['block'],
        params['layers'],
        width_multiplier,
        dropblock_keep_probs=dropblock_keep_probs,
        dropblock_size=dropblock_size)