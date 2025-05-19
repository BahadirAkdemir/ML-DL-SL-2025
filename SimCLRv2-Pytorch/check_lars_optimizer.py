import numpy as np
import torch

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf

import tensorflow.compat.v2 as tf


from pytorch.lars_optimizer import LARSOptimizer as LARSOptimizerTorch
from tf2.lars_optimizer import LARSOptimizer as LARSOptimizerTF  # Replace with actual paths

# Set fixed seed
np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)

# === Shared initial weights ===
init_weights = np.random.randn(2, 2).astype(np.float32)

# === TensorFlow Section ===
print("=== TensorFlow ===")
x_tf = tf.Variable(init_weights.copy(), name="x")

optimizer_tf = LARSOptimizerTF(learning_rate=0.1, weight_decay=0.01)

with tf.GradientTape() as tape:
    loss_tf = tf.reduce_sum(x_tf ** 2)
grads_tf = tape.gradient(loss_tf, [x_tf])
optimizer_tf.apply_gradients(zip(grads_tf, [x_tf]))

x_tf_result = x_tf.numpy()
v_tf = optimizer_tf.get_slot(x_tf, "Momentum").numpy()

print("x_tf:\n", x_tf_result)
print("Momentum_tf:\n", v_tf)

# === PyTorch Section ===
print("\n=== PyTorch ===")
x_torch = torch.nn.Parameter(torch.tensor(init_weights.copy()))
optimizer_torch = LARSOptimizerTorch([x_torch], learning_rate=0.1, weight_decay=0.01)

def loss_fn(x):
    return torch.sum(x ** 2)

optimizer_torch.zero_grad()
loss_torch = loss_fn(x_torch)
loss_torch.backward()
optimizer_torch.step()

x_torch_result = x_torch.detach().numpy()
v_torch = optimizer_torch.state[x_torch]['Momentum'].detach().numpy()

print("x_torch:\n", x_torch_result)
print("Momentum_torch:\n", v_torch)

# === Comparison ===
print("\n=== Comparison ===")
print("Weights equal:", np.allclose(x_tf_result, x_torch_result, rtol=1e-5, atol=1e-6))
print("Momentum equal:", np.allclose(v_tf, v_torch, rtol=1e-5, atol=1e-6))