import math
import torch
import torch.nn as nn
import einops
from config_args import args


# https://arxiv.org/abs/2010.11929

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

        self.W_q = nn.Linear(d_model, d_model, bias=True)  # , bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=True)  # , bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=True)  # , bias=False)
        # Only weights, biases werent mentioned in the paper
        # also OK to add bias. Your choice

        self.W_o = nn.Linear(d_model, d_model, bias=True)  # , bias=False)
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
    def __init__(self, d_model: int = 768, dff: int = 3072, dropout: float = 0):
        super(FeedForwardNetwork, self).__init__()

        self.d_model = d_model
        self.dff = dff

        self.layer1 = nn.Linear(self.d_model, self.dff)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.layer2 = nn.Linear(self.dff, self.d_model)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, dff) -> (batch, seq_len, d_model)
        return self.dropout(self.layer2(self.dropout(self.activation(self.layer1(x)))))

        # also torch.relu() can be used directly, instead of assigning variable for relu.
        # Probably no important difference but using direct function may use less memory.
        # I assigned variable to make it more readable


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float, d_model:int):
        super(ResidualConnection, self).__init__()

        self.dropout = nn.Dropout(dropout)

        # self.norm = LayerNormalization(d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x, sublayer):
        # return x + self.dropout(sublayer(self.norm(x)))
        return self.norm(x + self.dropout(sublayer(x)))


class InputEmbedding(nn.Module):
    def __init__(self, img_size: int, in_channels: int, patch_size: int, d_model: int):
        super().__init__()

        assert img_size % patch_size == 0, "Image size must be divisible by patch size"

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.d_model = d_model
        self.patch_dim = self.patch_size ** 2 * in_channels
        self.proj = nn.Linear(self.patch_dim, d_model)

        # CLS token initialized as a learnable parameter
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        x = einops.rearrange(x, "b c (h ph) (w pw) -> b (h w) (ph pw c)", ph=self.patch_size,
                             pw=self.patch_size)
        x = self.proj(x)

        cls_token = self.cls_token.expand(x.size(0), -1, -1)  # Shape: (b, 1, d_model)

        x = torch.cat((cls_token, x), dim=1)  # Shape: (b, 1 + num_patches, d_model)
        # Shape: (b, 1 + h * w, ph * pw * c)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super(PositionalEncoding, self).__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        self.positional_encoding = nn.Parameter(torch.randn(1, seq_len + 1, d_model))

    def forward(self, x):
        return self.dropout(x + self.positional_encoding)


class EncoderBlock(nn.Module):
    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardNetwork,
                 dropout: float,
                 d_model):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout,d_model) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        # lambda is a function taking input x and calling self.self_attention_block(x, x, x, src_mask)

        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model:int):
        super(Encoder, self).__init__()
        self.layer = layers
        # self.norm = LayerNormalization(d_model)
        self.norm = nn.LayerNorm(d_model, eps=1e-5)

    def forward(self, x, mask):
        for layer in self.layer:
            x = layer(x, mask)
        return self.norm(x)[:,0]


class ViT(nn.Module):
    def __init__(self,
                 encoder: Encoder,
                 src_embed: InputEmbedding,
                 src_pos: PositionalEncoding,
                 ):
        super(ViT, self).__init__()

        self.encoder = encoder
        self.src_embed = src_embed
        self.src_pos = src_pos

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def forward(self, img):
        return self.encode(img, None)


def build_transformer(img: torch.Tensor, patch_size= 16, d_model=768, dropout=0.1, Nx=12, h=16, d_ff=3072):


    # Creating Input Embeddings: This part converts the text input/ground truth to meaningful float vector space
    src_embed = InputEmbedding(img[3],img[1], patch_size, d_model)

    # Creating Positional Encoding : This part adds positional info to input embeddings since there are no recurrent
    # layer, positional info should be add with some way.
    src_seq_len = (img[3] // patch_size)**2
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)

    # Creating Encoder Blocks
    encoder_blocks = []
    for _ in range(Nx):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardNetwork(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout, d_model)
        encoder_blocks.append(encoder_block)



    # Creating a complete Encoder and Decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks), d_model)

    # Creating Projection Part

    # Creating the complete Transformer Mechanism
    vit = ViT(encoder, src_embed, src_pos)

    # Initializing
    for p in vit.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return vit

def vit():
    return build_transformer(args.size)


#img = torch.randn(7,3,224,224)
#vit_model = build_transformer(img)
#vit_model = vit()
#x = vit_model(img)

#print(x.shape)
