import numpy as np
import torch
import tensorflow as tf

# === Dummy metric classes for PyTorch ===
class MeanMetric:
    def __init__(self, name):
        self.name = name
        self.total = 0.0
        self.count = 0

    def update_state(self, value):
        self.total += value
        self.count += 1

    def result(self):
        return self.total / self.count if self.count != 0 else 0.0

# Top-1 accuracy assuming labels are either one-hot or class indices
class Top1Accuracy:
    def __init__(self, name):
        self.name = name
        self.correct = 0
        self.total = 0

    def update_state(self, y_true, y_pred):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        self.correct += np.sum(y_true == y_pred)
        self.total += len(y_true)

    def result(self):
        return self.correct / self.total if self.total != 0 else 0.0

# Dummy top-5 accuracy
class Top5Accuracy:
    def __init__(self, name):
        self.name = name
        self.correct = 0
        self.total = 0

    def update_state(self, y_true, y_pred):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        y_true_idx = np.argmax(y_true, axis=1)
        top5 = np.argsort(-y_pred, axis=1)[:, :5]
        self.correct += sum([yt in top for yt, top in zip(y_true_idx, top5)])
        self.total += len(y_true)

    def result(self):
        return self.correct / self.total if self.total != 0 else 0.0

# === TensorFlow metric equivalents ===
def create_tf_metrics():
    return {
        "loss": tf.keras.metrics.Mean(name="loss"),
        "acc": tf.keras.metrics.Mean(name="acc"),
        "entropy": tf.keras.metrics.Mean(name="entropy"),
        "top1": tf.keras.metrics.Accuracy(name="top1"),
        "top5": tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")
    }

# === Dummy metric objects for PyTorch ===
def create_torch_metrics():
    return {
        "loss": MeanMetric("loss"),
        "acc": MeanMetric("acc"),
        "entropy": MeanMetric("entropy"),
        "top1": Top1Accuracy("top1"),
        "top5": Top5Accuracy("top5")
    }

# === Import the metric update functions ===
import tf2.metrics as tf_metrics  # should contain the TF version
import pytorch.metrics as torch_metrics  # should contain the PyTorch version

# === Set same seed ===
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)

# === Shared dummy data ===
logits_np = np.random.randn(10, 5).astype(np.float32)
labels_np = np.zeros_like(logits_np)
labels_np[np.arange(10), np.random.randint(0, 5, 10)] = 1

# === TensorFlow tensors ===
logits_tf = tf.constant(logits_np)
labels_tf = tf.constant(labels_np)
loss_tf = tf.reduce_mean(tf.reduce_sum(tf.square(logits_tf - labels_tf), axis=1))

# === PyTorch tensors ===
logits_torch = torch.tensor(logits_np)
labels_torch = torch.tensor(labels_np)
loss_torch = torch.mean(torch.sum((logits_torch - labels_torch) ** 2, dim=1))

# === Run TensorFlow metric updates ===
tf_metrics_objs = create_tf_metrics()
tf_metrics.update_pretrain_metrics_train(
    tf_metrics_objs["loss"],
    tf_metrics_objs["acc"],
    tf_metrics_objs["entropy"],
    loss_tf,
    logits_tf,
    labels_tf
)
tf_metrics.update_pretrain_metrics_eval(
    tf_metrics_objs["loss"],
    tf_metrics_objs["top1"],
    tf_metrics_objs["top5"],
    loss_tf,
    logits_tf,
    labels_tf
)

# === Run PyTorch metric updates ===
torch_metrics_objs = create_torch_metrics()
torch_metrics.update_pretrain_metrics_train(
    torch_metrics_objs["loss"],
    torch_metrics_objs["acc"],
    torch_metrics_objs["entropy"],
    loss_torch,
    logits_torch,
    labels_torch
)
torch_metrics.update_pretrain_metrics_eval(
    torch_metrics_objs["loss"],
    torch_metrics_objs["top1"],
    torch_metrics_objs["top5"],
    loss_torch,
    logits_torch,
    labels_torch
)

# === Compare metrics ===
print("\n=== Metric Comparison ===")
for key in tf_metrics_objs:
    tf_value = tf_metrics_objs[key].result().numpy() if hasattr(tf_metrics_objs[key], 'result') else tf_metrics_objs[key]
    torch_value = torch_metrics_objs[key].result()
    print(f"{key}: tf = {tf_value:.6f}, torch = {torch_value:.6f}, equal = {np.allclose(tf_value, torch_value, atol=1e-5)}")
