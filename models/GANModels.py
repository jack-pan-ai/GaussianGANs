import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import copy

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Generator(nn.Module):
    def __init__(self,
                 seq_len = 150,
                 channels = 3,
                 num_heads = 5, # num_head must be integer factor for seq_len
                 noise_dim = 20,  # the number of noise dimensions in the input for generator
                 depth = 4,  # the number of depth for transformer blocks
                 args = None
                ):
        super(Generator, self).__init__()
        self.channels = channels
        self.seq_len = seq_len # the sequence length of the whole time series
        self.depth = depth

        # model definition
        self.fc = FC(noise_dim=noise_dim, channels=channels, seq_len=seq_len)
        self.transformerSynthesis = TransformerSynthesis_generator(depth=depth, seq_len = seq_len, num_heads = num_heads)
        self.l1 = nn.Linear(in_features=seq_len, out_features=seq_len)
        self.args = args

    def forward(self, z):
        w = self.fc(z) # the latent space [batch_size, channels, seq]
        batch_size, channels, seq_len = w.shape
        if self.args.gpu is not None:
            x = torch.randn([1, channels, seq_len]).cuda(self.args.gpu)
        else:
            x = torch.randn([1, channels, seq_len], device='cuda')
        x = self.transformerSynthesis(x, w)  # [batch_size, channels, seq]
        x = self.l1(x)
        return x

class FC(nn.Module):
    '''
    The FC layers contain the positional embedding
    '''
    def __init__(self,
                 noise_dim, # the dimensions for random noise
                 channels,  # the dimension for multivariates
                 seq_len,  # the length of time series
                 **kwargs):
        super(FC, self).__init__()
        self.emb_size = noise_dim
        self.channels = channels
        self.seq_len = seq_len

        # model definition
        ## extend the emb_size to channels*seq_length
        self.l1 = nn.Linear(in_features=noise_dim, out_features=channels*seq_len)
        self.postional_embedding = nn.Parameter(torch.randn([1, channels, seq_len]))
        self.fc = nn.Sequential(
            nn.Linear(in_features=seq_len, out_features=seq_len),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=seq_len, out_features=seq_len),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=seq_len, out_features=seq_len),
            nn.LeakyReLU(0.2),
            nn.Linear(in_features=seq_len, out_features=seq_len),
            nn.LeakyReLU(0.2),
        )


    def forward(self, w):
        '''
        input: random noise [batch_size, emb_size]
        output: latent space [batch_size, channels, seq_length]
        '''
        w = self.l1(w).reshape([-1, self.channels, self.seq_len]) # [batch_size, emb_size] -> [batch_size, channels, seqlength]
        w = w + self.postional_embedding # postional embedding
        w = self.fc(w)  # [batch_size, channels, seqlength] -> [batch_size, channels, seqlength]

        return w

class TransformerSynthesisBlock_generator(nn.Module):
    def __init__(self,
                 seq_len, # length of sequence of time serie s
                 num_heads,
                 atten_drop_p=0.2,
                 forward_drop_p=0.2):
        super(TransformerSynthesisBlock_generator, self).__init__()

        self.atten_blocks = ResidualAdd(nn.Sequential(
            nn.LayerNorm(seq_len),
            MultiHeadAttention(in_dims=seq_len, num_heads=num_heads, dropout=atten_drop_p),
            nn.Dropout(atten_drop_p),
        ))
        self.feedforward_blocks = ResidualAdd(nn.Sequential(
            nn.LayerNorm(seq_len),
            FeedForwardBlock(in_dims=seq_len, drop_p=forward_drop_p),
            nn.Dropout(forward_drop_p),
        ))

    def forward(self, x, w):
        '''
        x, w : # [batch_size, channels, seq_len]
        '''
        x = x + w
        x = self.atten_blocks(x) # shape does not change: [batch_size, channels, seq_len]
        x = self.feedforward_blocks(x) # shape does not change: [batch_size, channels, seq_len]

        return x
        
class TransformerSynthesis_generator(nn.Module):
    def __init__(self, depth, **kwargs):
        super(TransformerSynthesis_generator, self).__init__()
        self.encoder = clones(TransformerSynthesisBlock_generator(**kwargs), N= depth)

    def forward(self, x, w):
        for module in self.encoder:
            x = module(x, w)
        return x
        
class MultiHeadAttention(nn.Module):
    def __init__(self, in_dims, num_heads, dropout):
        super().__init__()
        self.emb_size = in_dims
        self.num_heads = num_heads # must be integer multiple of seq_len: seq_len = integer * num_heads
        self.keys = nn.Linear(in_dims, in_dims)
        self.queries = nn.Linear(in_dims, in_dims)
        self.values = nn.Linear(in_dims, in_dims)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(in_dims, in_dims)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)

        return out
    
class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super(ResidualAdd, self).__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x

class FeedForwardBlock(nn.Module):
    def __init__(self, in_dims, drop_p):
        super(FeedForwardBlock ,self).__init__()
        self.seq_len = in_dims

        self.l1 = nn.Sequential(
            nn.Linear(in_dims, in_dims),
            nn.ReLU(),
        )
        self.d1 = nn.Dropout(drop_p)

    def forward(self, x):
        x = self.l1(x)
        x = self.d1(x)

        return x
        
class Discriminator(nn.Module):
    def __init__(self,
                 channels = 3,
                 seq_len = 150,
                 emb_size = 50, # transform the [batch_size, channels, seq_len] -> [batch_size, channels, emb_size]
                 num_heads = 5,
                 depth = 4,
                 n_classes=1,
                 ):
        super(Discriminator, self).__init__()
        # parameters setting
        self.channels = channels
        self.emb_size = emb_size
        self.seq_len = seq_len

        # layer initialization
        self.position_embedding = PositionEmbedding(channels = channels, emb_size = emb_size, seq_len = seq_len)
        self.transformer_discriminator = TransformerDecoder_discriminator(depth = depth, emb_size = emb_size, num_heads = num_heads)
        self.classification = ClassificationHead(emb_size = emb_size, n_classes = n_classes)

    def forward(self, x):
        x = self.position_embedding(x) # [batch_size, in_channels, seq_length] -> [batch_size, in_channels, emb_size]
        x = self.transformer_discriminator(x)
        x = self.classification(x)

        return x

class PositionEmbedding(nn.Module):
    '''
    Used to add the learnable positional embedding
    emb_size: the numebr of parameters in the middle layers
    seq_length: the length of the time series
    in_channels: the number of multivariates
    '''
    def __init__(self, channels, emb_size, seq_len):
        super(PositionEmbedding, self).__init__()
        self.projection = nn.Linear(seq_len, emb_size)
        self.positions = nn.Parameter(torch.randn(channels, seq_len))

    def forward(self, x: Tensor) -> Tensor:
        '''
        x shape: [batch_size, in_channels, seq_length]
        '''
        x = x + self.positions
        x = self.projection(x) # [batch_size, in_channels, seq_length] -> [batch_size, in_channels, emb_size]
        return x

class TransformerDecoder_discriminator(nn.Module):
    def __init__(self, depth, **kwargs):
        super(TransformerDecoder_discriminator, self).__init__()
        self.decoder = clones(TransformerDecoderblock_discriminator(**kwargs), depth)

    def forward(self, x):
        for module in self.decoder:
            x = module(x)

        return x

class TransformerDecoderblock_discriminator(nn.Module):
    def __init__(self,
                 emb_size,
                 num_heads,
                 drop_p=0.2,
                 ):
        super(TransformerDecoderblock_discriminator, self).__init__()
        self.atten_block = ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            MultiHeadAttention(in_dims=emb_size, num_heads=num_heads, dropout=drop_p),
            nn.Dropout(drop_p),
        ))
        self.feedforward_block = ResidualAdd(nn.Sequential(
            nn.LayerNorm(emb_size),
            FeedForwardBlock(in_dims=emb_size, drop_p=drop_p),
            nn.Dropout(drop_p),
        ))

    def forward(self,x):
        x = self.atten_block(x)
        x = self.feedforward_block(x)

        return x

class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        x = self.clshead(x)
        out = torch.sigmoid(x)
        return out

if __name__ == '__main__':
    d = Discriminator()
    a = d(torch.randn([16, 3, 150]))
    print(a)
    print(a.shape)