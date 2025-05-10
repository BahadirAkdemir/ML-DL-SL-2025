import torch
import torch.nn as nn
import numpy as np
import random
import time
import math

d_model = 512
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size):
        # d_model: dimension of vector representing each word
        # vocab_size: how many words there are in
        super(InputEmbeddings, self).__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model ** 0.5)
        # math.sqrt(self.d_model**0.5): The last sentence of 3.4 Embeddings and Softmax paragraph


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Create a long enough PExD tensor
        pe = torch.zeros(seq_len, d_model)  # value for each dimensions of each word
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # pos* value (seq_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        # 3.5 Positional Encoding formula: To be more stability, log scaled

        pe[:, 0::2] = torch.sin(position * div_term)  # even positions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd positions

        pe = pe.unsqueeze(0)  # shape: (1, seq_len, d_model)

        self.register_buffer('pe', pe)
        # I don't know too much about it.
        # It saves the tensor as part of the model (moves with model but not trained)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:, :seq_len])

        # return x + self.pe[:, :seq_len].requires_grad(False) is not necessary
        # because we already registered as static value


class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        # eps: epsilon. Its for preventiong vanishing the denomiator while normalization.
        super(LayerNormalization, self).__init__()

        # alpha and bias: to introduce some fluctuations in data
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))  # per-feature scale
        self.bias = nn.Parameter(torch.zeros(d_model))  # per-feature bias

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model: int = 512, dff: int = 2048, dropout: float = 0):
        super(FeedForwardNetwork, self).__init__()

        self.d_model = d_model
        self.dff = dff

        self.layer1 = nn.Linear(self.d_model, self.dff)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(self.dff, self.d_model)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, dff) -> (batch, seq_len, d_model)
        return self.layer2(self.dropout(self.activation(self.layer1(x))))

        # also torch.relu() can be used directly, instead of assigning variable for relu.
        # Probably no important difference but using direct function may use less memory.
        # I assigned variable to make it more readable


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        # d_model: Dimension of word embedding vector
        # h: number of heads which will divide the embedding dimension

        super(MultiHeadAttentionBlock, self).__init__()

        self.d_model = d_model
        self.h = h

        assert (d_model % h == 0), "Word embedding must be divisible by number of heads (d_model / h)"

        self.d_k = self.d_q = self.d_v = self.d_model // self.h
        # To make it more readable.

        self.W_q = nn.Linear(d_model, d_model, bias=True)# , bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=True)# , bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=True)# , bias=False)
        # Only weights, biases werent mentioned in the paper
        # also OK to add bias. Your choice

        self.W_o = nn.Linear(d_model, d_model, bias=True)# , bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(q, k, v, mask, d_k, d_model, dropout):

        attention_scores = torch.einsum("nqhd,nkhd->nhqk", [q, k])

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(attention_scores / (d_k ** (1 / 2)), dim=3)
        if dropout is not None:
            attention = dropout(attention)
        attention_result = torch.einsum("nhql,nlhd->nqhd", [attention, v]).reshape(
            v.size(0), v.size(1), d_model
        )
        return attention_result

    def forward(self, q, k, v, mask):
        query = self.W_q(q)  # (batch, seq_len, d_model)
        key = self.W_k(k)  # (batch, seq_len, d_model)
        value = self.W_v(v)  # (batch, seq_len, d_model)

        # Split into heads

        query = query.reshape(q.size(0), q.size(1), self.h, self.d_q)  # (batch, seq_len, h, d_q)
        key = key.reshape(k.size(0), k.size(1), self.h, self.d_k)  # (batch, seq_len, h, d_k)
        value = value.reshape(v.size(0), v.size(1), self.h, self.d_v)  # (batch, seq_len, h, d_v)

        out = self.W_o(MultiHeadAttentionBlock.attention(query, key, value, mask, self.d_k, self.d_model, self.dropout))

        return out


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super(ResidualConnection, self).__init__()

        self.dropout = nn.Dropout(dropout)

        #self.norm = LayerNormalization(d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x, sublayer):
        #return x + self.dropout(sublayer(self.norm(x)))
        return x + self.dropout(self.norm(sublayer(x)))


class EncoderBlock(nn.Module):
    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardNetwork,
                 dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # lambda is a function taking input x and calling self.self_attention_block(x, x, x, src_mask)

        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super(Encoder, self).__init__()
        self.layer = layers
        #self.norm = LayerNormalization(d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x, mask):
        for layer in self.layer:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardNetwork,
                 dropout: float):
        super(DecoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block

        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])


    def forward(self, x, encoder_output, src_mask, trg_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, trg_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super(Decoder, self).__init__()
        self.layers = layers
        #self.norm = LayerNormalization(d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x, encoder_output, src_mask, trg_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, trg_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super(ProjectionLayer, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # Shape: (batch, seq_len, d_model) → (batch, seq_len, vocab_size)
        logits = self.linear(x)
        return self.softmax(logits)


class Transformer(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbeddings,
                 trg_embed: InputEmbeddings,
                 src_pos: PositionalEncoding,
                 trg_pos: PositionalEncoding,
                 projection: ProjectionLayer
                 ):
        super(Transformer, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.src_pos = src_pos
        self.trg_pos = trg_pos
        self.projection = projection

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, trg, trg_mask):
        trg = self.trg_embed(trg)
        trg = self.trg_pos(trg)
        return self.decoder(trg, encoder_output, src_mask, trg_mask)

    def project(self, x):
        return self.projection(x)


def build_transformer(src_vocab_size: int,
                      trg_vocab_size: int,
                      src_seq_len: int,
                      trg_seq_len: int,
                      d_model: int = 512,
                      Nx: int = 6,
                      h: int = 8,
                      dropout: float = 0.0,
                      d_ff: int = 2048):
    # Creating Input Embeddings: This part converts the text input/ground truth to meaningful float vector space
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    trg_embed = InputEmbeddings(d_model, trg_vocab_size)

    # Creating Positional Encoding : This part adds positional info to input embeddings since there are no recurrent
    # layer, positional info should be add with some way.
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    trg_pos = PositionalEncoding(d_model, trg_seq_len, dropout)

    # Creating Encoder Blocks
    encoder_blocks = []
    for _ in range(Nx):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardNetwork(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Creating Decoder Blocks
    decoder_blocks = []
    for _ in range(Nx):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardNetwork(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block,
                                     feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Creating a complete Encoder and Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Creating Projection Part
    projection_layer = ProjectionLayer(d_model, trg_vocab_size)

    # Creating the complete Transformer Mechanism
    transformer = Transformer(encoder, decoder, src_embed, trg_embed, src_pos, trg_pos, projection_layer)

    # Initializing
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


# ========== Utility Functions ==========

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generate_random_input(batch_size, seq_len, vocab_size):
    return torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)

def apply_custom_init(model):
    for name, param in model.named_parameters():
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)

def compare_models(custom_model, torch_model, src, tgt, src_mask, tgt_mask):
    with torch.no_grad():
        # Custom Transformer
        start = time.time()
        custom_output = custom_model.project(
            custom_model.decode(
                custom_model.encode(src, src_mask),
                src_mask,
                tgt,
                tgt_mask
            )
        )
        custom_time = time.time() - start

        # PyTorch Transformer
        start = time.time()
        torch_output = torch_model(src, tgt, src_mask, tgt_mask)
        torch_time = time.time() - start

        if custom_output.shape != torch_output.shape:
            raise ValueError(f"Shape mismatch: custom {custom_output.shape}, torch {torch_output.shape}")

        mae = torch.mean(torch.abs(custom_output - torch_output)).item()

    return mae, custom_time, torch_time

def measure_time(model, x, mask, label, device, n=100):
    model.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(n):
            if isinstance(model, nn.MultiheadAttention):
                model(x, x, x, need_weights=False)
            else:
                model(x, x, x, mask)
        elapsed = (time.time() - start) / n
    print(f"{label:<35}: {elapsed:.6f} seconds per forward pass")

def mae(x, y):
    return torch.mean(torch.abs(x - y)).item()

# ========== Attention Equivalence ==========

def set_shared_weights(attn_custom: nn.Module, attn_torch: nn.MultiheadAttention):
    with torch.no_grad():
        attn_torch.in_proj_weight.copy_(torch.cat([
            attn_custom.W_q.weight, attn_custom.W_k.weight, attn_custom.W_v.weight
        ], dim=0))
        attn_torch.in_proj_bias.copy_(torch.cat([
            attn_custom.W_q.bias, attn_custom.W_k.bias, attn_custom.W_v.bias
        ], dim=0))
        attn_torch.out_proj.weight.copy_(attn_custom.W_o.weight)
        attn_torch.out_proj.bias.copy_(attn_custom.W_o.bias)

def test_attention_equivalence():
    print("\n=== Testing Attention Equivalence ===")
    torch.set_printoptions(precision=16)
    device = torch.device("cpu")
    dtype = torch.float64

    x = torch.randn(10, 10, 512, dtype=dtype, device=device)
    mask = None

    attn_custom = MultiHeadAttentionBlock(512, 8, 0.0).to(dtype).to(device)
    attn_torch = nn.MultiheadAttention(512, 8, dropout=0.0, batch_first=True, bias=True).to(dtype).to(device)

    set_shared_weights(attn_custom, attn_torch)

    y_torch, _ = attn_torch(x, x, x, need_weights=False)
    y_custom = attn_custom(x, x, x, mask)

    print(f"✅ Matches PyTorch: {torch.allclose(y_custom, y_torch, atol=1e-10)}")
    print(f"🔍 MAE: {mae(y_custom, y_torch):.16f}")
    print(f"📊 Parameters - Custom: {count_parameters(attn_custom)}, PyTorch: {count_parameters(attn_torch)}")

    measure_time(attn_custom, x, mask, "Custom MultiHeadAttentionBlock", device)
    measure_time(attn_torch, x, mask, "PyTorch MultiheadAttention", device)

# ========== Full Weight Sharing ==========

def share_weights(custom_model, torch_model):
    with torch.no_grad():
        # Embeddings
        torch_model.src_embed.weight.copy_(custom_model.src_embed.embedding.weight)
        torch_model.trg_embed.weight.copy_(custom_model.trg_embed.embedding.weight)

        # Generator
        torch_model.generator.weight.copy_(custom_model.projection.linear.weight)

        # Encoder Layers
        for custom_layer, torch_layer in zip(custom_model.encoder.layer, torch_model.transformer.encoder.layers):
            c_attn = custom_layer.self_attention_block
            t_attn = torch_layer.self_attn

            t_attn.in_proj_weight.copy_(torch.cat([c_attn.W_q.weight, c_attn.W_k.weight, c_attn.W_v.weight], dim=0))
            t_attn.in_proj_bias.copy_(torch.cat([c_attn.W_q.bias, c_attn.W_k.bias, c_attn.W_v.bias], dim=0))
            t_attn.out_proj.weight.copy_(c_attn.W_o.weight)
            t_attn.out_proj.bias.copy_(c_attn.W_o.bias)

            c_ff = custom_layer.feed_forward_block
            torch_layer.linear1.weight.copy_(c_ff.layer1.weight)
            torch_layer.linear1.bias.copy_(c_ff.layer1.bias)
            torch_layer.linear2.weight.copy_(c_ff.layer2.weight)
            torch_layer.linear2.bias.copy_(c_ff.layer2.bias)

            torch_layer.norm1.weight.copy_(custom_layer.residual_connections[0].norm.alpha)
            torch_layer.norm1.bias.copy_(custom_layer.residual_connections[0].norm.bias)
            torch_layer.norm2.weight.copy_(custom_layer.residual_connections[1].norm.alpha)
            torch_layer.norm2.bias.copy_(custom_layer.residual_connections[1].norm.bias)

        # Decoder Layers
        for custom_layer, torch_layer in zip(custom_model.decoder.layers, torch_model.transformer.decoder.layers):
            # Self Attention
            c_attn = custom_layer.self_attention_block
            t_attn = torch_layer.self_attn
            t_attn.in_proj_weight.copy_(torch.cat([c_attn.W_q.weight, c_attn.W_k.weight, c_attn.W_v.weight], dim=0))
            t_attn.in_proj_bias.copy_(torch.cat([c_attn.W_q.bias, c_attn.W_k.bias, c_attn.W_v.bias], dim=0))
            t_attn.out_proj.weight.copy_(c_attn.W_o.weight)
            t_attn.out_proj.bias.copy_(c_attn.W_o.bias)

            # Cross Attention
            c_cross = custom_layer.cross_attention_block
            t_cross = torch_layer.multihead_attn
            t_cross.in_proj_weight.copy_(torch.cat([c_cross.W_q.weight, c_cross.W_k.weight, c_cross.W_v.weight], dim=0))
            t_cross.in_proj_bias.copy_(torch.cat([c_cross.W_q.bias, c_cross.W_k.bias, c_cross.W_v.bias], dim=0))
            t_cross.out_proj.weight.copy_(c_cross.W_o.weight)
            t_cross.out_proj.bias.copy_(c_cross.W_o.bias)

            # FeedForward
            c_ff = custom_layer.feed_forward_block
            torch_layer.linear1.weight.copy_(c_ff.layer1.weight)
            torch_layer.linear1.bias.copy_(c_ff.layer1.bias)
            torch_layer.linear2.weight.copy_(c_ff.layer2.weight)
            torch_layer.linear2.bias.copy_(c_ff.layer2.bias)

            # LayerNorms
            torch_layer.norm1.weight.copy_(custom_layer.residual_connections[0].norm.alpha)
            torch_layer.norm1.bias.copy_(custom_layer.residual_connections[0].norm.bias)
            torch_layer.norm2.weight.copy_(custom_layer.residual_connections[1].norm.alpha)
            torch_layer.norm2.bias.copy_(custom_layer.residual_connections[1].norm.bias)
            torch_layer.norm3.weight.copy_(custom_layer.residual_connections[2].norm.alpha)
            torch_layer.norm3.bias.copy_(custom_layer.residual_connections[2].norm.bias)

        # Final Norms
        torch_model.transformer.encoder.norm.weight.copy_(custom_model.encoder.norm.alpha)
        torch_model.transformer.encoder.norm.bias.copy_(custom_model.encoder.norm.bias)
        torch_model.transformer.decoder.norm.weight.copy_(custom_model.decoder.norm.alpha)
        torch_model.transformer.decoder.norm.bias.copy_(custom_model.decoder.norm.bias)



def test_encoder_equivalence():
    print("\n=== Testing Encoder Equivalence ===")
    torch.set_printoptions(precision=16)
    device = torch.device("cpu")
    dtype = torch.float64

    # --- Config ---
    vocab_size = 100
    seq_len = 50
    batch_size = 2
    d_model = 512
    nhead = 8
    num_layers = 6
    dropout = 0.0
    ff_dim = 2048

    # --- Inputs ---
    src = generate_random_input(batch_size, seq_len, vocab_size).to(device)
    src_mask = None

    # --- Build Custom Encoder ---
    custom_transformer = build_transformer(vocab_size, vocab_size, seq_len, seq_len,
                                           d_model, num_layers, nhead, dropout, ff_dim).to(dtype).to(device)
    apply_custom_init(custom_transformer)
    custom_transformer.eval()

    custom_encoder = custom_transformer.encoder
    custom_embed = custom_transformer.src_embed
    custom_pe = custom_transformer.src_pos
    custom_pe.eval()

    # --- Build PyTorch Encoder ---
    class PytorchEncoderWrapper(nn.Module):
        def __init__(self, embed, pos_encoding, encoder_layer, norm):
            super().__init__()
            self.embed = embed
            self.pos = pos_encoding
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
            self.norm = norm

        def forward(self, x, src_mask=None):
            x = self.embed(x) * math.sqrt(d_model)
            x = self.pos(x)
            return self.norm(self.encoder(x, src_key_padding_mask=src_mask))

    # Create compatible encoder layer
    encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, ff_dim, dropout, batch_first=True).to(dtype)
    torch_encoder = PytorchEncoderWrapper(
        embed=custom_embed.embedding,  # Share same embedding
        pos_encoding=custom_pe,        # Share exact same PE instance
        encoder_layer=encoder_layer,
        norm=nn.LayerNorm(d_model).to(dtype)
    ).to(dtype)

    apply_custom_init(torch_encoder)
    torch_encoder.eval()

    # --- Share Weights ---
    with torch.no_grad():
        # Encoder layers
        for c_layer, t_layer in zip(custom_encoder.layer, torch_encoder.encoder.layers):
            # Attention
            t_layer.self_attn.in_proj_weight.copy_(
                torch.cat([c_layer.self_attention_block.W_q.weight,
                           c_layer.self_attention_block.W_k.weight,
                           c_layer.self_attention_block.W_v.weight], dim=0))
            t_layer.self_attn.in_proj_bias.copy_(
                torch.cat([c_layer.self_attention_block.W_q.bias,
                           c_layer.self_attention_block.W_k.bias,
                           c_layer.self_attention_block.W_v.bias], dim=0))
            t_layer.self_attn.out_proj.weight.copy_(c_layer.self_attention_block.W_o.weight)
            t_layer.self_attn.out_proj.bias.copy_(c_layer.self_attention_block.W_o.bias)

            # Feedforward
            t_layer.linear1.weight.copy_(c_layer.feed_forward_block.layer1.weight)
            t_layer.linear1.bias.copy_(c_layer.feed_forward_block.layer1.bias)
            t_layer.linear2.weight.copy_(c_layer.feed_forward_block.layer2.weight)
            t_layer.linear2.bias.copy_(c_layer.feed_forward_block.layer2.bias)

            # Norms
            t_layer.norm1.weight.copy_(c_layer.residual_connections[0].norm.alpha)
            t_layer.norm1.bias.copy_(c_layer.residual_connections[0].norm.bias)
            t_layer.norm2.weight.copy_(c_layer.residual_connections[1].norm.alpha)
            t_layer.norm2.bias.copy_(c_layer.residual_connections[1].norm.bias)

        # Final norm
        torch_encoder.norm.weight.copy_(custom_encoder.norm.alpha)
        torch_encoder.norm.bias.copy_(custom_encoder.norm.bias)

    # --- Forward Pass ---
    with torch.no_grad():
        input_embedded = custom_pe(custom_embed(src) * math.sqrt(d_model))
        custom_encoded = custom_encoder(input_embedded, src_mask)
        torch_encoded = torch_encoder(src, src_mask)

    # --- Metrics ---
    error = mae(custom_encoded, torch_encoded)
    print("✅ Encoder outputs shape match:", custom_encoded.shape == torch_encoded.shape)
    print("🔍 Encoder MAE:", error)



# ========== Main ==========

if __name__ == "__main__":
    set_seed(42)

    vocab_size = 100
    batch_size = 2
    seq_len = 50
    d_model = 512
    nhead = 8
    num_layers = 6
    dropout = 0.0
    ff_dim = 2048

    # --- Build models ---
    custom_transformer = build_transformer(vocab_size, vocab_size, seq_len, seq_len, d_model, num_layers, nhead, dropout, ff_dim).double()
    apply_custom_init(custom_transformer)

    class BuiltInTransformerWrapper(nn.Module):
        def __init__(self, src_vocab_size, trg_vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout):
            super().__init__()
            self.d_model = d_model
            self.src_embed = nn.Embedding(src_vocab_size, d_model)
            self.trg_embed = nn.Embedding(trg_vocab_size, d_model)
            self.pos_encoder = PositionalEncoding(d_model, seq_len, dropout)
            self.transformer = nn.Transformer(d_model, nhead, num_layers, num_layers, dim_feedforward, dropout, batch_first=True)
            self.generator = nn.Linear(d_model, trg_vocab_size, bias=False)
            self.softmax = nn.Softmax(dim=-1)

        def forward(self, src, tgt, src_mask=None, tgt_mask=None):
            src = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
            tgt = self.pos_encoder(self.trg_embed(tgt) * math.sqrt(self.d_model))
            return self.softmax(self.generator(self.transformer(src, tgt, src_mask, tgt_mask)))

    torch_transformer = BuiltInTransformerWrapper(vocab_size, vocab_size, d_model, nhead, num_layers, ff_dim, dropout).double()
    apply_custom_init(torch_transformer)

    # --- Inputs ---
    src = generate_random_input(batch_size, seq_len, vocab_size)
    tgt = generate_random_input(batch_size, seq_len, vocab_size)
    src_mask = tgt_mask = None

    # --- Initial comparison ---
    mae1, t1_custom, t1_torch = compare_models(custom_transformer, torch_transformer, src, tgt, src_mask, tgt_mask)
    print("\n=== Before Weight Sharing ===")
    print("Custom Transformer parameters:", count_parameters(custom_transformer))
    print("PyTorch Transformer parameters:", count_parameters(torch_transformer))
    print("MAE:", mae1)
    print("Custom Time: {:.6f}s".format(t1_custom))
    print("Torch Time:  {:.6f}s".format(t1_torch))

    # --- Sync weights ---
    share_weights(custom_transformer, torch_transformer)
    custom_transformer.eval()
    torch_transformer.eval()

    # --- Post-sharing comparison ---
    mae2, t2_custom, t2_torch = compare_models(custom_transformer, torch_transformer, src, tgt, src_mask, tgt_mask)
    print("\n=== After Weight Sharing ===")
    print("MAE:", mae2)
    print("Custom Time: {:.6f}s".format(t2_custom))
    print("Torch Time:  {:.6f}s".format(t2_torch))

    # --- Run attention test ---
    test_attention_equivalence()

    test_encoder_equivalence()


