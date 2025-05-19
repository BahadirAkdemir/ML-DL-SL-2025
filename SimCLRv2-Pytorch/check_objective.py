import numpy as np
import torch
import tensorflow as tf

# === IMPORTS ===
import tf2.objective as tf_contrastive  # TensorFlow version file
import pytorch.objective as torch_contrastive  # PyTorch version file

# === CONFIG ===
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)

batch_size = 8
embedding_dim = 16
temperature = 0.5

# === SHARED INPUT DATA ===

# Create one-hot labels for classification (TF needs one-hot, PyTorch needs class indices)
class_indices = np.random.randint(0, 10, size=(batch_size,))
labels_one_hot = np.eye(10)[class_indices].astype(np.float32)

# Create random logits (for classification)
logits_np = np.random.randn(batch_size, 10).astype(np.float32)

# Create hidden vectors for contrastive loss (2 * batch_size, embedding_dim)
hidden_np = np.random.randn(2 * batch_size, embedding_dim).astype(np.float32)

# === TENSORFLOW TEST ===
logits_tf = tf.constant(logits_np)
labels_tf = tf.constant(labels_one_hot)
hidden_tf = tf.constant(hidden_np)

sup_loss_tf = tf_contrastive.add_supervised_loss(labels_tf, logits_tf)

contrast_loss_tf, logits_ab_tf, contrast_labels_tf = tf_contrastive.add_contrastive_loss(
    hidden=hidden_tf,
    hidden_norm=True,
    temperature=temperature,
    strategy=None
)

# === PYTORCH TEST ===
logits_torch = torch.tensor(logits_np)
labels_torch = torch.tensor(class_indices)  # class indices
hidden_torch = torch.tensor(hidden_np)

sup_loss_torch = torch_contrastive.add_supervised_loss(labels_torch, logits_torch)

contrast_loss_torch, logits_ab_torch, contrast_labels_torch = torch_contrastive.add_contrastive_loss(
    hidden=hidden_torch,
    hidden_norm=True,
    temperature=temperature,
    strategy=None
)

# === COMPARE RESULTS ===
def compare(name, tf_val, torch_val, atol=1e-5):
    tf_np = tf_val.numpy() if isinstance(tf_val, tf.Tensor) else tf_val
    torch_np = torch_val.detach().cpu().numpy() if isinstance(torch_val, torch.Tensor) else torch_val
    equal = np.allclose(tf_np, torch_np, atol=atol)
    print(f"{name}: {'OK' if equal else 'MISMATCH'}")
    if not equal:
        print(f"  TensorFlow: {tf_np}")
        print(f"  PyTorch:   {torch_np}")

print("=== Supervised Loss ===")
compare("Supervised loss", sup_loss_tf, sup_loss_torch)

print("\n=== Contrastive Loss ===")
compare("Contrastive loss", contrast_loss_tf, contrast_loss_torch)

print("\n=== Logits_ab ===")
compare("Logits_ab", logits_ab_tf, logits_ab_torch)

print("\n=== Contrastive Labels ===")
compare("Contrastive labels", contrast_labels_tf, contrast_labels_torch)
