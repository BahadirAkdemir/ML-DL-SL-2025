# Transformer Language Model Training

A clean, well-designed training pipeline for a transformer-based language model with comprehensive features for text generation and analysis.

## Features

### Training Pipeline
- ✅ **Data Loading & Splitting**: Automatic train/validation split from `TinyStories.txt`, `data.txt` or `example.txt`
- ✅ **Optimized Dataset**: Uses `GPTDatasetV1` from `data.py` with configurable overlap
- ✅ **Tokenization**: GPT-2 compatible tokenizer using tiktoken
- ✅ **Cross Entropy Loss**: Standard language modeling objective
- ✅ **Perplexity Calculation**: Automatic perplexity metrics during evaluation
- ✅ **AdamW Optimizer**: State-of-the-art optimizer with weight decay
- ✅ **Model Architecture**: Transformer decoder from `model.py` with GPT-2 parameters
- ✅ **Pre-trained Weights**: Load and fine-tune official OpenAI GPT-2 weights

### Training Features
- ✅ **Gradient Clipping**: Prevents exploding gradients
- ✅ **Periodic Evaluation**: Configurable evaluation frequency
- ✅ **Model Checkpointing**: Automatic model saving during training
- ✅ **Progress Tracking**: tqdm progress bars and detailed logging
- ✅ **Loss Plotting**: Automatic generation of training/validation loss plots

### Text Generation
- ✅ **Temperature Scaling**: Control randomness in generation
- ✅ **Top-K Sampling**: Limit sampling to top-k most likely tokens
- ✅ **Top-P (Nucleus) Sampling**: Dynamic vocabulary size based on cumulative probability
- ✅ **Multiple Decoding Strategies**: Compare different generation approaches

### Monitoring & Analysis
- ✅ **Real-time Generation**: Generate examples during training
- ✅ **Loss Visualization**: Matplotlib plots saved automatically
- ✅ **Model Statistics**: Parameter counting and model analysis
- ✅ **Reproducibility**: Fixed random seeds for consistent results

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Place your training text in `data.txt`, or the script will use `example.txt` as fallback.

### 3. Run Training
```bash
python train.py
```

## GPT-2 Pre-trained Weights

### Using Pre-trained Models

The training script now supports loading and fine-tuning official OpenAI GPT-2 weights! This provides several advantages:

- 🚀 **Faster convergence**: Start with learned representations
- 📈 **Better performance**: Leverage massive pre-training dataset
- 🎯 **Fine-tuning**: Adapt to your specific domain/task
- 🔄 **Multiple sizes**: Choose from 124M to 1.5B parameters

### Available Models

| Model | Parameters | Description |
|-------|------------|-------------|
| `gpt2-small (124M)` | 124M | Original GPT-2 model, good for experimentation |
| `gpt2-medium (355M)` | 355M | Larger model, better quality |
| `gpt2-large (774M)` | 774M | High-quality generation |
| `gpt2-xl (1558M)` | 1.5B | Largest model, best performance |

### How to Use

Simply modify the configuration in `train.py`:

```python
# Option 1: Train from scratch (default)
USE_PRETRAINED = False

# Option 2: Fine-tune pre-trained GPT-2 model
USE_PRETRAINED = True
MODEL_NAME = "gpt2-small (124M)"  # Choose your model size
```

### First Run Setup

On first run with pre-trained weights, the script will:

1. 📥 **Download**: Automatically download the GPT-2 weight loading script
2. 🔄 **Fetch Weights**: Download official OpenAI weights for your chosen model
3. 🗂️ **Cache**: Store weights locally in `gpt2/` directory for future use
4. 🔄 **Map Weights**: Automatically map OpenAI weights to your model architecture
5. ✅ **Ready**: Start fine-tuning immediately

### Weight Mapping

The script automatically handles the complex weight mapping between OpenAI's GPT-2 format and your model architecture:

- **Token Embeddings**: `wte` → `InputEmbeddings.embedding`
- **Position Embeddings**: `wpe` → `PositionalEncoding.pe`
- **Attention Weights**: `blocks[i].attn` → `MultiHeadAttentionBlock`
- **Feed-Forward**: `blocks[i].mlp` → `FeedForwardNetwork`
- **Layer Norms**: `ln_1/ln_2` → `LayerNormalization`

### Fine-tuning vs Training from Scratch

| Aspect | Pre-trained | From Scratch |
|--------|-------------|--------------|
| **Speed** | ⚡ Fast convergence | 🐌 Slower convergence |
| **Data Requirements** | 📊 Works with small datasets | 📈 Needs large datasets |
| **Quality** | 🎯 High quality from start | 📈 Improves gradually |
| **Learning Rate** | 🔽 Lower (1e-5) | ⬆️ Higher (3e-4) |
| **Epochs** | 🔢 Fewer needed (5-10) | 🔢 More needed (20-50) |
| **Use Case** | 🎯 Fine-tuning, adaptation | 🔬 Research, learning |

## Configuration

The training script includes a comprehensive configuration dictionary that you can modify:

```python
config = {
    'seq_len': 128,           # Sequence length
    'embedding_dim': 256,     # Model embedding dimension
    'num_layers': 4,          # Number of transformer layers
    'num_heads': 8,           # Number of attention heads
    'dropout': 0.1,           # Dropout rate
    'ffn_hidden_dim': 1024,   # Feed-forward network hidden dimension
    'batch_size': 8,          # Training batch size
    'learning_rate': 3e-4,    # Learning rate for AdamW
    'num_epochs': 5,          # Number of training epochs
    'eval_freq': 500,         # Evaluation frequency (steps)
    'save_freq': 1000,        # Model saving frequency (steps)
}
```

## Model Architecture

The model uses a transformer decoder architecture with:
- Multi-head self-attention
- Position-wise feed-forward networks
- Layer normalization
- Residual connections
- Learned positional encodings

## Text Generation Examples

The script demonstrates various decoding strategies:

### Temperature Scaling
- **Low temperature (0.5)**: More conservative, focused generation
- **Standard temperature (1.0)**: Balanced creativity and coherence
- **High temperature (1.5)**: More creative, diverse generation

### Top-K Sampling
- **Small K (10)**: Very focused vocabulary
- **Medium K (50)**: Balanced vocabulary size
- **Large K (100)**: Larger vocabulary, more diversity

### Top-P (Nucleus) Sampling
- Dynamic vocabulary size based on cumulative probability
- Adapts to model confidence

## Output Files

The training process generates several files:

1. **`training_losses.png`**: Loss plots showing training progress
2. **`model_checkpoint_step_X.pt`**: Periodic model checkpoints
3. **`final_trained_model.pt`**: Final trained model with full state
4. **Console output**: Real-time training metrics and generated examples

## Model Loading

To load a trained model for inference:

```python
import torch
from model import build_transformer
from tokenizer import GPT2Tokenizer

# Load the saved model
checkpoint = torch.load('final_trained_model.pt', map_location='cpu')
config = checkpoint['config']
tokenizer = GPT2Tokenizer()

# Rebuild model architecture
model = build_transformer(
    input_vocab_size=tokenizer.encoding.n_vocab,
    input_seq_len=config['seq_len'],
    embedding_dim=config['embedding_dim'],
    num_layers=config['num_layers'],
    num_heads=config['num_heads'],
    dropout=config['dropout'],
    ffn_hidden_dim=config['ffn_hidden_dim']
)

# Load trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Advanced Features

### Evaluation Metrics
- **Cross Entropy Loss**: Standard language modeling loss
- **Perplexity**: Intuitive metric for model performance (lower is better)
- **Per-step tracking**: Monitor training progress in real-time

### Generation Control
The `TextGenerator` class provides fine-grained control over text generation:
- Adjustable sequence length
- Multiple sampling strategies
- Early stopping on special tokens

### Training Monitoring
- Progress bars with ETA
- Real-time loss reporting
- Automatic validation evaluation
- Example generation during training

## Performance Tips

1. **GPU Usage**: The script automatically detects and uses CUDA if available
2. **Memory Management**: Adjust `batch_size` based on available GPU memory
3. **Sequence Length**: Longer sequences require more memory but may improve quality
4. **Evaluation Frequency**: More frequent evaluation provides better monitoring but slows training

## File Structure

```
.
├── train.py              # Main training script
├── model.py              # Transformer model architecture
├── tokenizer.py          # GPT-2 compatible tokenizer
├── data.txt              # Training data (create this)
├── example.txt           # Fallback training data
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` or `seq_len`
2. **No data file found**: Ensure `data.txt` exists or `example.txt` is present
3. **Import errors**: Install all requirements with `pip install -r requirements.txt`
4. **Slow training**: Consider using a GPU or reducing model size

### Performance Optimization

- Use mixed precision training for faster GPU training
- Implement gradient accumulation for larger effective batch sizes
- Consider using a learning rate scheduler
- Implement early stopping based on validation loss

## License

This implementation is for educational and research purposes.
