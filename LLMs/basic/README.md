# Transformer Language Model Components

A collection of transformer model components and utilities for building and experimenting with language models.

## Project Components

### Core Files
- ✅ **Model Architecture**: `model.py` - Complete transformer implementation with GPT-2 compatible architecture
- ✅ **Tokenization**: `tokenizer.py` - GPT-2 compatible tokenizer using tiktoken
- ✅ **Main Entry**: `main.py` - Basic entry point for the project
- ✅ **Configuration**: `pyproject.toml` - Project configuration and dependencies

### Model Architecture Features

The `model.py` file contains a comprehensive transformer implementation with:

- 🏗️ **Complete Transformer Stack**: Input embeddings, positional encoding, decoder blocks, and output projection
- 🎯 **Multi-Head Attention**: Efficient attention mechanism with configurable heads
- 🔄 **Feed-Forward Networks**: Position-wise feed-forward layers with GELU activation
- 📏 **Layer Normalization**: Pre-norm architecture for stable training
- 🔗 **Residual Connections**: Skip connections for gradient flow
- 🎭 **Causal Masking**: Proper autoregressive generation support
- 🎲 **Text Generation**: Built-in generation methods with temperature control

### Available Transformer Components

| Component | Description | Location |
|-----------|-------------|----------|
| `InputEmbeddings` | Token embedding layer | `model.py` |
| `PositionalEncoding` | Learnable position embeddings | `model.py` |
| `MultiHeadAttentionBlock` | Self-attention mechanism | `model.py` |
| `FeedForwardNetwork` | Position-wise FFN | `model.py` |
| `LayerNormalization` | Custom layer norm | `model.py` |
| `ResidualConnection` | Skip connections | `model.py` |
| `DecoderBlock` | Complete transformer block | `model.py` |
| `Decoder` | Stack of decoder blocks | `model.py` |
| `ProjectionLayer` | Output vocabulary projection | `model.py` |
| `Transformer` | Complete model assembly | `model.py` |

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Basic Usage
```bash
python main.py
```

### 3. Model Creation Example
```python
from model import build_transformer
from tokenizer import GPT2Tokenizer

# Initialize tokenizer
tokenizer = GPT2Tokenizer()
vocab_size = tokenizer.encoding.n_vocab

# Build a GPT-2 style model
model = build_transformer(
    input_vocab_size=vocab_size,
    input_seq_len=1024,
    embedding_dim=768,
    num_layers=12,
    num_heads=12,
    dropout=0.1,
    ffn_hidden_dim=3072
)

print(f"Model created with {model.count_parameters():,} parameters")
```

## Model Configurations

### GPT-2 Compatible Sizes

| Model Size | Parameters | Embedding Dim | Layers | Heads | Context Length |
|------------|------------|---------------|---------|-------|----------------|
| Small | 124M | 768 | 12 | 12 | 1024 |
| Medium | 355M | 1024 | 24 | 16 | 1024 |
| Large | 774M | 1280 | 36 | 20 | 1024 |
| XL | 1.5B | 1600 | 48 | 25 | 1024 |

### Custom Configuration Example
```python
# Custom smaller model for experimentation
model = build_transformer(
    input_vocab_size=50257,  # GPT-2 vocab size
    input_seq_len=512,       # Shorter context
    embedding_dim=256,       # Smaller embedding
    num_layers=6,            # Fewer layers
    num_heads=8,             # Fewer heads
    dropout=0.1,
    ffn_hidden_dim=1024      # Smaller FFN
)
```

## Text Generation

The model includes built-in text generation capabilities:

### Basic Generation
```python
# Generate text with temperature control
generated_text = model.generate_next(
    input_text="The future of AI is",
    max_length=50,
    temperature=0.8
)
```

### Advanced Generation Options
- **Temperature Scaling**: Control randomness (0.1 = focused, 2.0 = creative)
- **Causal Masking**: Proper autoregressive generation
- **Token-by-token**: Step-by-step generation process

## Tokenization

The project uses a GPT-2 compatible tokenizer:

```python
from tokenizer import GPT2Tokenizer

tokenizer = GPT2Tokenizer()

# Encode text to tokens
tokens = tokenizer.encode("Hello, world!")
print(f"Tokens: {tokens}")

# Decode tokens back to text
text = tokenizer.decode(tokens)
print(f"Decoded: {text}")

# Vocabulary info
print(f"Vocabulary size: {tokenizer.encoding.n_vocab}")
```

## Pre-trained Weights Support

The model architecture is designed to be compatible with official GPT-2 weights:

- ✅ **Weight Mapping**: Automatic mapping from OpenAI format
- ✅ **Layer Compatibility**: Matching layer structure and naming
- ✅ **Embedding Sharing**: Tied input/output embeddings
- ✅ **Attention Format**: Compatible attention weight structure

## Architecture Details

### Attention Mechanism
- Scaled dot-product attention
- Multi-head parallel computation
- Causal masking for autoregressive generation
- Configurable number of attention heads

### Feed-Forward Networks
- Two-layer MLP with GELU activation
- Typically 4x expansion ratio (hidden_dim = 4 * embedding_dim)
- Dropout for regularization

### Normalization and Residuals
- Pre-normalization (LayerNorm before attention/FFN)
- Residual connections around each sub-layer
- Stable gradient flow through deep networks

## Development Setup

### Project Structure
```
.
├── model.py              # Complete transformer implementation
├── tokenizer.py          # GPT-2 tokenizer wrapper
├── main.py              # Basic entry point
├── requirements.txt      # Dependencies
├── pyproject.toml       # Project configuration
├── TinyStories-train.txt # Training data
└── README.md            # This documentation
```

### Dependencies
The project requires:
- PyTorch (deep learning framework)
- tiktoken (tokenization)
- NumPy (numerical operations)
- Other utilities (see `requirements.txt`)

## Advanced Features

### Model Analysis
```python
# Count parameters
total_params = model.count_parameters()
print(f"Total parameters: {total_params:,}")

# Analyze model structure
for name, module in model.named_modules():
    if hasattr(module, 'weight'):
        print(f"{name}: {module.weight.shape}")
```

### Memory Efficient Operations
- Gradient checkpointing support
- Efficient attention computation
- Configurable dropout rates
- Device-aware tensor operations

## Use Cases

### Research and Experimentation
- Study transformer architectures
- Experiment with different configurations
- Understand attention mechanisms
- Compare model sizes and performance

### Educational Purposes
- Learn transformer implementation details
- Understand autoregressive language modeling
- Practice with PyTorch and deep learning
- Explore text generation techniques

### Development and Prototyping
- Build custom language models
- Integrate with larger systems
- Fine-tune for specific domains
- Experiment with architectural modifications

## Performance Considerations

### Memory Usage
- Model size scales quadratically with sequence length
- Attention computation is O(n²) in sequence length
- Larger models require more GPU memory
- Batch size affects memory usage significantly

### Computational Efficiency
- Use appropriate device (CUDA/MPS/CPU)
- Consider mixed precision training
- Optimize batch sizes for hardware
- Profile memory usage during development

## Troubleshooting

### Common Issues
1. **Import errors**: Ensure all dependencies are installed
2. **CUDA out of memory**: Reduce model size or batch size
3. **Shape mismatches**: Check input dimensions and model configuration
4. **Tokenization issues**: Verify text encoding and special tokens

### Performance Tips
- Use GPU acceleration when available
- Profile code to identify bottlenecks
- Consider model parallelism for very large models
- Implement gradient accumulation for large effective batch sizes

## License

This implementation is for educational and research purposes.

## Contributing

Feel free to explore, modify, and extend the transformer implementation for your specific needs. The modular design makes it easy to experiment with different architectural choices.
