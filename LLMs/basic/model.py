import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import GPT2Tokenizer




class InputEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size):
        # d_model: dimension of vector representing each word
        # vocab_size: how many words there are in
        super(InputEmbeddings, self).__init__()

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.pred_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

    def forward(self, x):
        pred_token = self.pred_token.expand(x.size(0), 1, self.embedding_dim)
        x =self.embedding(x)
        #x = torch.cat([x, pred_token], dim=1)
        return x * math.sqrt(self.embedding_dim ** 0.5)
        # math.sqrt(self.d_model**0.5): The last sentence of 3.4 Embeddings and Softmax paragraph

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim: int, seq_len: int, dropout: float):
        super(PositionalEncoding, self).__init__()

        #learnable positional encoding. Add +1 to seq_len because of prediction token
        self.pe = nn.Embedding(seq_len, embedding_dim)

    def forward(self, x):
        # x: (batch_size, seq_len+1, embedding_dim)
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        pos_embed = self.pe(positions)
        return x + pos_embed


class LayerNormalization(nn.Module):
    def __init__(self, embedding_dim: int, eps: float = 1e-6):
        # eps: epsilon. Its for preventiong vanishing the denomiator while normalization.
        super(LayerNormalization, self).__init__()

        # alpha and bias: to introduce some fluctuations in data
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(embedding_dim))  # per-feature scale
        self.bias = nn.Parameter(torch.zeros(embedding_dim))  # per-feature bias

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim: int = 512, ffn_hidden_dim: int = 3072, dropout: float = 0):
        super(FeedForwardNetwork, self).__init__()

        self.embedding_dim = embedding_dim
        self.ffn_hidden_dim = ffn_hidden_dim

        self.layer1 = nn.Linear(self.embedding_dim, self.ffn_hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(self.ffn_hidden_dim, self.embedding_dim)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, dff) -> (batch, seq_len, d_model)
        return self.layer2(self.dropout(self.activation(self.layer1(x))))


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, dropout: float):
        # d_model: Dimension of word embedding vector
        # h: number of heads which will divide the embedding dimension

        super(MultiHeadAttentionBlock, self).__init__()

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        assert (embedding_dim % num_heads == 0), "Word embedding must be divisible by number of heads (embedding_dim / num_heads)"

        self.d_k = self.d_q = self.d_v = self.embedding_dim // self.num_heads
        # To make it more readable.

        self.W_q = nn.Linear(embedding_dim, embedding_dim, bias=True)# , bias=False)
        self.W_k = nn.Linear(embedding_dim, embedding_dim, bias=True)# , bias=False)
        self.W_v = nn.Linear(embedding_dim, embedding_dim, bias=True)# , bias=False)
        # Only weights, biases werent mentioned in the paper
        # also OK to add bias. Your choice

        self.W_o = nn.Linear(embedding_dim, embedding_dim, bias=True)# , bias=False)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(q, k, v, mask, d_k, embedding_dim, dropout):

        attention_scores = torch.einsum("nqhd,nkhd->nhqk", [q, k])

        if mask:
            mask_tmp = torch.triu(torch.ones(q.size(1), q.size(1)), diagonal=1).to(q.device)
            attention_scores = attention_scores.masked_fill(mask_tmp == 0, float("-1e20"))

        attention = torch.softmax(attention_scores / (d_k ** (1 / 2)), dim=3)
        if dropout is not None:
            attention = dropout(attention)
        attention_result = torch.einsum("nhql,nlhd->nqhd", [attention, v]).reshape(
            v.size(0), v.size(1), embedding_dim
        )
        return attention_result

    def forward(self, q, k, v, mask):
        query = self.W_q(q)  # (batch, seq_len, d_model)
        key = self.W_k(k)  # (batch, seq_len, d_model)
        value = self.W_v(v)  # (batch, seq_len, d_model)

        # Split into heads

        query = query.reshape(q.size(0), q.size(1), self.num_heads, self.d_q)  # (batch, seq_len, h, d_q)
        key = key.reshape(k.size(0), k.size(1), self.num_heads, self.d_k)  # (batch, seq_len, h, d_k)
        value = value.reshape(v.size(0), v.size(1), self.num_heads, self.d_v)  # (batch, seq_len, h, d_v)

        out = self.W_o(MultiHeadAttentionBlock.attention(query, key, value, mask, self.d_k, self.embedding_dim, self.dropout))

        return out


class ResidualConnection(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(embedding_dim)

    def forward(self, x, sublayer):
        #return x + self.dropout(sublayer(self.norm(x)))
        return x + self.dropout(self.norm(sublayer(x)))


class DecoderBlock(nn.Module):
    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_network: FeedForwardNetwork,
                 embedding_dim: int,
                 dropout: float):
        super(DecoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_network = feed_forward_network

        self.residual_connections = nn.ModuleList([ResidualConnection(embedding_dim, dropout) for _ in range(2)])


    def forward(self, x, input_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, input_mask))
        x = self.residual_connections[1](x, self.feed_forward_network)

        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, embedding_dim: int):
        super(Decoder, self).__init__()
        self.layers = layers
        self.norm = LayerNormalization(embedding_dim)

    def forward(self, x, input_mask):
        for layer in self.layers:
            x = layer(x, input_mask)
        
        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int):
        super(ProjectionLayer, self).__init__()
        self.linear = nn.Linear(embedding_dim, vocab_size, bias=True)

    def forward(self, x):
        # Shape: (batch, seq_len, d_model) → (batch, seq_len, vocab_size)
        return self.linear(x)

class Transformer(nn.Module):
    def __init__(self,
                 decoder_module: Decoder,
                 input_embedding: InputEmbeddings,
                 input_positional_encoding: PositionalEncoding,
                 projection_layer: ProjectionLayer
                 ):
        super(Transformer, self).__init__()

        self.decoder_module = decoder_module
        self.input_embedding = input_embedding
        self.input_positional_encoding = input_positional_encoding
        self.projection_layer = projection_layer

    def forward(self, input, input_mask):
        input = self.input_embedding(input)
        input = self.input_positional_encoding(input)
        input = self.decoder_module(input, input_mask)

        return self.projection_layer(input)



def build_transformer(
                      input_vocab_size: int,
                      input_seq_len: int,
                      embedding_dim: int = 512,
                      num_layers: int = 6,
                      num_heads: int = 8,
                      dropout: float = 0.0,
                      ffn_hidden_dim: int = 3072):
    # Creating Input Embeddings: This part converts the text input/ground truth to meaningful float vector space
    input_embedding = InputEmbeddings(embedding_dim, input_vocab_size)

    # Creating Positional Encoding : This part adds positional info to input embeddings since there are no recurrent
    # layer, positional info should be add with some way.
    input_positional_encoding = PositionalEncoding(embedding_dim=embedding_dim, seq_len=input_seq_len, dropout=dropout)


    # Creating Decoder Blocks
    decoder_block_list = []
    for _ in range(num_layers):
        self_attention_block = MultiHeadAttentionBlock(embedding_dim, num_heads, dropout)
        cross_attention_block = MultiHeadAttentionBlock(embedding_dim, num_heads, dropout)
        feed_forward_network = FeedForwardNetwork(embedding_dim, ffn_hidden_dim, dropout)
        decoder_layer = DecoderBlock(self_attention_block,
                                     feed_forward_network, embedding_dim, dropout)
        decoder_block_list.append(decoder_layer)

    # Creating a complete Encoder and Decoder
    decoder_module = Decoder(nn.ModuleList(decoder_block_list), embedding_dim)

    # Creating Projection Part
    projection_layer = ProjectionLayer(embedding_dim, input_vocab_size)

    # Creating the complete Transformer Mechanism
    transformer_model = Transformer(decoder_module, input_embedding, input_positional_encoding, projection_layer)

    # Weight tying: tie output projection weights to input embeddings
    transformer_model.projection_layer.linear.weight = transformer_model.input_embedding.embedding.weight

    # Initializing (AFTER tying!)
    for p in transformer_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer_model

def create_mask(input_size):
    return torch.triu(torch.ones(input_size, input_size), diagonal=1)

from tabulate import tabulate

def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def parameter_report(model):
    report = []
    total_params = 0

    # Embedding
    embedding_params = count_parameters(model.input_embedding)
    report.append(["Embedding", f"{embedding_params / 1e6:.1f}M"])
    total_params += embedding_params

    pos_embedding_params = count_parameters(model.input_positional_encoding)
    report.append(["Embedding", f"{pos_embedding_params / 1e6:.1f}M"])
    total_params += pos_embedding_params

    # Projection
    if model.projection_layer.linear.weight is model.input_embedding.embedding.weight:
        projection_params = 0
        report.append(["Projection", "tied (0 extra)"])
    else:
        projection_params = count_parameters(model.projection_layer)
        report.append(["Projection", f"{projection_params / 1e6:.1f}M"])
        total_params += projection_params

    # Decoder Blocks
    block_params = [count_parameters(block) for block in model.decoder_module.layers]
    block_total = sum(block_params)
    report.append([f"{len(block_params)} Decoder Blocks",
                   f"{block_params[0] / 1e6:.1f}M × {len(block_params)} = {block_total / 1e6:.1f}M"])
    total_params += block_total

    # Total
    report.append(["Total", f"{total_params / 1e6:.1f}M"])

    print(tabulate(report, headers=["Component", "Est. Parameters"], tablefmt="github"))


if __name__ == "__main__":
    GPT2_PARAMS = {
        "input_vocab_size": 50257,
        "input_seq_len": 1024,
        "embedding_dim": 768,
        "num_layers": 12,
        "num_heads": 12,
        "dropout": 0.1,
        "ffn_hidden_dim": 3072}

    
    transformer_model = build_transformer(**GPT2_PARAMS)
    transformer_model.eval()

    parameter_report(transformer_model)

    # print the model architecture
    tokenizer = GPT2Tokenizer()
    text = "Hello LLM! Do you know what is the capital of Turkiye?"

    
    temperature = 0.8

    # 6 iterations. concat input and output
    for _ in range(10):
        tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>","<|endofprompt|>"})
        tokens = torch.tensor(tokens).unsqueeze(0)
        tokens = tokenizer.encode(text, allowed_special={"<|endoftext|>","<|endofprompt|>"})
        tokens = torch.tensor(tokens).unsqueeze(0)
        # expand tokens like a batch of 5
        input = tokens.expand(5, -1)
        with torch.no_grad():
            logits = transformer_model(input, input_mask=True)  # See below for this change
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)  # Apply temperature
            next_token = torch.multinomial(probs, num_samples=1)  # Sample from distribution
            text += tokenizer.decode(next_token[0].tolist())

            print(text)
        
