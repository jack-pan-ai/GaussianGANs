import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce


class inverseGenerator(nn.Module):
    def __init__(self, seq_len,  channels, num_heads, latent_dim, depth,
                 patch_size, embed_dim=10,
                 forward_drop_rate=0.5, attn_drop_rate=0.5):
        super(inverseGenerator, self).__init__()
        self.channels = channels
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.depth = depth
        self.attn_drop_rate = attn_drop_rate
        self.forward_drop_rate = forward_drop_rate

        self.l1 = nn.Linear(self.seq_len * self.embed_dim, self.latent_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, self.channels, self.seq_len))
        self.blocks = Gen_TransformerEncoder(
            depth=self.depth,
            num_heads = self.num_heads,
            emb_size=self.embed_dim,
            drop_p=self.attn_drop_rate,
            forward_drop_p=self.forward_drop_rate
        )

        self.reconv = nn.Sequential(
            nn.Conv2d(self.channels, self.embed_dim, 1)
        )

    def forward(self, z):
        z = z + self.pos_embed
        z = z.unsqueeze(2) # [batch_size, channels, seq_len] -> [batch_size, channels, 1, seq_len]
        z = self.reconv(z)  # [batch_size, channels, 1, seq_len] -> [batch_size, embed_dim, 1, seq_len]
        z = z.squeeze(2)  # [batch_size, embed_dim, 1, seq_len] -> [batch_size, embed_dim, seq_len]
        z = rearrange(z, 'b e s -> b s e')
        z = self.blocks(z).view(-1, self.embed_dim * self.seq_len) # [batch_size, embed_dim, seq_len] -> [batch_size, embed_dim * seq_len]
        z = self.l1(z).unsqueeze(1) # [batch_size, embed_dim * seq_len] -> [batch_size, 1, latent_dim]

        return z


class Gen_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads,
                 drop_p,
                 forward_drop_p,
                 forward_expansion=4,
                 ):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class Gen_TransformerEncoder(nn.Sequential):
    def __init__(self, depth=8, **kwargs):
        super().__init__(*[Gen_TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

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
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class Dis_TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads,
                 drop_p,
                 forward_drop_p,
                 forward_expansion=4,
                 ):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class Dis_TransformerEncoder(nn.Sequential):
    def __init__(self, depth, **kwargs):
        super().__init__(*[Dis_TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes):
        super().__init__()
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes),
        )

    def forward(self, x):
        out = self.clshead(x)
        return out

class PatchEmbedding_Linear(nn.Module):

    def __init__(self, in_channels, patch_size, emb_size, seq_length):
        # self.patch_size = patch_size
        super().__init__()
        # change the conv2d parameters here
        self.projection = nn.Sequential(
            # [batch_size, channels, (seq_len//patch_size, patch_size)] -> [batch_size, seq_len//patch_size, channels*patch_size]
            Rearrange('b c (w s2) -> b w (s2 c)', s2=patch_size),
            # [batch_size, seq_len//patch_size, channels*patch_size] -> [batch_size, seq_len//patch_size, emb_size]
            nn.Linear(patch_size * in_channels, emb_size)
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn(seq_length//patch_size + 1, emb_size))

    def forward(self, x: Tensor) -> Tensor:
        b, _, _ = x.shape
        x = self.projection(x) # [batch_size, channels, seq_len] -> [batch_size, seq_len//patch_size, emb_size]
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        # prepend the cls token to the input
        x = torch.cat([cls_tokens, x], dim=1) # [batch_size, seq_len//patch_size, emb_size] -> [batch_size, seq_len//patch_size + 1, emb_size]
        # position
        x += self.positions
        return x


class inverseDiscriminator(nn.Sequential):
    def __init__(self,
                 seq_len,
                 channels,
                 depth,
                 patch_size=8,
                 num_heads=5, # num_head * integer = emb_size
                 emb_size=50,
                 n_classes=1,
                 **kwargs):
        super(inverseDiscriminator, self).__init__(
            # [batch_size, channels, seq_len] -> [batch_size, seq_len//patch_size + 1, emb_size]
            PatchEmbedding_Linear(in_channels=channels, patch_size=patch_size,
                                  emb_size=emb_size, seq_length=seq_len),
            # [batch_size, seq_len//patch_size + 1, emb_size] -> [batch_size, seq_len//patch_size + 1, emb_size]
            Dis_TransformerEncoder(depth, emb_size=emb_size, num_heads = num_heads,
                                   drop_p=0.5, forward_drop_p=0.5,
                                   **kwargs),
            # [batch_size, seq_len//patch_size + 1, emb_size] -> [batch_size, 1]
            ClassificationHead(emb_size, n_classes)
        )

if __name__ == "__main__":
    gen = inverseGenerator(seq_len = 150,  channels=3, num_heads=5, latent_dim=64, depth=4, patch_size=15)
    out_gen = gen(torch.randn(128, 3, 150))
    print(out_gen.shape)
    dis = inverseDiscriminator(seq_len = 64, channels =1, depth=4, patch_size=8)
    out_gen = dis(torch.randn(128, 1 ,64))
    print(out_gen.shape)

