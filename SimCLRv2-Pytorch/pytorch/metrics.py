import torch
import torch.nn.functional as F
import numpy as np
import logging


class MeanMetric:
    def __init__(self, name='mean'):
        self.name = name
        self.total = 0.0
        self.count = 0

    def update_state(self, value):
        self.total += value
        self.count += 1

    def result(self):
        return self.total / self.count if self.count != 0 else 0.0

    def reset(self):
        self.total = 0.0
        self.count = 0


class Top1Accuracy:
    def __init__(self, name='top1_acc'):
        self.name = name
        self.correct = 0
        self.total = 0

    def update_state(self, y_true, y_pred):
        if y_true.dim() != 1:
            y_true = torch.argmax(y_true, dim=1)
        if y_pred.dim() != 1:
            y_pred = torch.argmax(y_pred, dim=1)

        self.correct += (y_true == y_pred).sum().item()
        self.total += y_true.size(0)

    def result(self):
        return self.correct / self.total if self.total != 0 else 0.0

    def reset(self):
        self.correct = 0
        self.total = 0

class Top5Accuracy:
    def __init__(self, name='top5_acc'):
        self.name = name
        self.correct = 0
        self.total = 0

    def update_state(self, y_true, y_pred):
        """
        y_true: (batch_size, num_classes) or (batch_size,)
        y_pred: (batch_size, num_classes)
        """
        if y_true.dim() != 1:
            y_true = torch.argmax(y_true, dim=1)

        # Top 5 predictions
        top5 = torch.topk(y_pred, k=5, dim=1).indices  # (batch_size, 5)
        match = top5.eq(y_true.unsqueeze(1))  # (batch_size, 5)
        self.correct += match.any(dim=1).sum().item()
        self.total += y_true.size(0)

    def result(self):
        return self.correct / self.total if self.total != 0 else 0.0

    def reset(self):
        self.correct = 0
        self.total = 0


def update_pretrain_metrics_train(contrast_loss, contrast_acc, contrast_entropy,
                                  loss, logits_con, labels_con):
    """Updated pretraining metrics."""
    contrast_loss.update_state(loss.item())

    y_true = torch.argmax(labels_con, dim=1)
    y_pred = torch.argmax(logits_con, dim=1)
    contrast_acc.update_state(y_true, y_pred)

    prob_con = F.softmax(logits_con, dim=1)
    entropy_con = -torch.sum(prob_con * torch.log(prob_con + 1e-8), dim=1).mean().item()
    contrast_entropy.update_state(entropy_con)


def update_pretrain_metrics_eval(contrast_loss_metric,
                                 contrastive_top_1_accuracy_metric,
                                 contrastive_top_5_accuracy_metric,
                                 contrast_loss, logits_con, labels_con):
    contrast_loss_metric.update_state(contrast_loss.item())

    y_true = torch.argmax(labels_con, dim=1)
    y_pred = torch.argmax(logits_con, dim=1)
    contrastive_top_1_accuracy_metric.update_state(y_true, y_pred)

    contrastive_top_5_accuracy_metric.update_state(labels_con, logits_con)


def update_finetune_metrics_train(supervised_loss_metric, supervised_acc_metric,
                                  loss, labels, logits):
    supervised_loss_metric.update_state(loss.item())

    y_true = labels                     # Already class indices
    y_pred = torch.argmax(logits, dim=1)
    supervised_acc_metric.update_state(y_true, y_pred)



def update_finetune_metrics_eval(label_top_1_accuracy_metrics,
                                 label_top_5_accuracy_metrics, outputs, labels):
    y_true = torch.argmax(labels, dim=1)
    y_pred = torch.argmax(outputs, dim=1)
    label_top_1_accuracy_metrics.update_state(y_true, y_pred)

    label_top_5_accuracy_metrics.update_state(labels, outputs)


def _float_metric_value(metric):
    return float(metric.result())


def log_and_write_metrics_to_summary(all_metrics, global_step, summary_writer):
    for metric in all_metrics:
        metric_value = _float_metric_value(metric)
        logging.info('Step: [%d] %s = %f', global_step, metric.name, metric_value)
        summary_writer.add_scalar(metric.name, metric_value, global_step)

