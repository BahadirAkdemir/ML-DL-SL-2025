import re
import torch
from torch.optim.optimizer import Optimizer, required

EETA_DEFAULT = 0.001

class LARSOptimizer(Optimizer):
    """Layer-wise Adaptive Rate Scaling for large batch training.

    Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
    I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
    """

    def __init__(self,
                 params,
                 learning_rate=required,
                 momentum=0.9,
                 use_nesterov=False,
                 weight_decay=0.0,
                 exclude_from_weight_decay=None,
                 exclude_from_layer_adaptation=None,
                 classic_momentum=True,
                 eeta=EETA_DEFAULT):
        if learning_rate is not required and learning_rate < 0.0:
            raise ValueError("Invalid learning rate: {}".format(learning_rate))
        defaults = dict(
            learning_rate=learning_rate,
            momentum=momentum,
            use_nesterov=use_nesterov,
            weight_decay=weight_decay,
            classic_momentum=classic_momentum,
            eeta=eeta,
        )
        super(LARSOptimizer, self).__init__(params=params, defaults=defaults)

        self.exclude_from_weight_decay = exclude_from_weight_decay or []
        self.exclude_from_layer_adaptation = exclude_from_layer_adaptation or self.exclude_from_weight_decay

    def _use_weight_decay(self, param_name):
        if not self.defaults['weight_decay']:
            return False
        for r in self.exclude_from_weight_decay:
            if re.search(r, param_name):
                return False
        return True

    def _do_layer_adaptation(self, param_name):
        for r in self.exclude_from_layer_adaptation:
            if re.search(r, param_name):
                return False
        return True

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['learning_rate']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            use_nesterov = group['use_nesterov']
            classic_momentum = group['classic_momentum']
            eeta = group['eeta']

            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad
                state = self.state[param]

                # State initialization
                if 'Momentum' not in state:
                    state['Momentum'] = torch.zeros_like(param)

                v = state['Momentum']
                param_name = param.__repr__()  # No actual name, so best effort

                if self._use_weight_decay(param_name):
                    grad = grad.add(param, alpha=weight_decay)

                if classic_momentum:
                    trust_ratio = 1.0
                    if self._do_layer_adaptation(param_name):
                        w_norm = torch.norm(param)
                        g_norm = torch.norm(grad)
                        trust_ratio = eeta * w_norm / (g_norm + 1e-10) if w_norm > 0 and g_norm > 0 else 1.0

                    scaled_lr = lr * trust_ratio
                    next_v = momentum * v + scaled_lr * grad

                    if use_nesterov:
                        update = momentum * next_v + scaled_lr * grad
                    else:
                        update = next_v

                    param.add_(-update)
                    state['Momentum'] = next_v

                else:
                    next_v = momentum * v + grad
                    if use_nesterov:
                        update = momentum * next_v + grad
                    else:
                        update = next_v

                    trust_ratio = 1.0
                    if self._do_layer_adaptation(param_name):
                        w_norm = torch.norm(param)
                        v_norm = torch.norm(update)
                        trust_ratio = eeta * w_norm / (v_norm + 1e-10) if w_norm > 0 and v_norm > 0 else 1.0

                    scaled_lr = lr * trust_ratio
                    param.add_(-scaled_lr * update)
                    state['Momentum'] = next_v

        return loss