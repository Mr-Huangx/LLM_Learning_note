# 前馈神经网络（FNN）
每一个 Encoder Layer 都包含一个self-attention注意力机制和一个前馈神经网络。前馈神经网络的实现是较为简单的</font>

```python
class MLP(nn.Module):
    '''前馈神经网络'''
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        # 定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播函数
        # 首先，输入x通过第一层线性变换和RELU激活函数
        # 最后，通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.relu(self.w1(x))))


```

Transformer 的前馈神经网络是由两个线性层中间加一个 RELU 激活函数组成的，以及前馈神经网络还加入了一个 Dropout 层来防止过拟合。</font>

</font>

# 归一化
 层归一化，Layer Norm。神经网络主流的归一化一般有两种，批归一化（Batch Norm）和层归一化（Layer Norm）。</font>

归一化核心是为了让不同层输入的取值范围或者分布能够比较一致。由于深度神经网络中每一层的输入都是上一层的输出，因此多层传递下，对网络中较高的层，之前的所有神经层的参数变化会导致其输入的分布发生较大的改变。也就是说，随着神经网络参数的更新，各层的输出分布是不相同的，且差异会随着网络深度的增大而增大。但是，需要预测的条件分布始终是相同的，从而也就造成了预测的误差。</font>

## 2.1 Batch Normalization</font>
在mini-batch上进行归一化操作：

$ \mu_j=\frac{1}{m}\sum_{i=1}^{m}Z_j^i $

其中， </font>$ z_j^i $是样本 i 在第 j 个维度上的值，m 就是 mini-batch 的大小。</font>

再计算样本的方差:</font>

$ \sigma^2 = \frac{1}{m}\sum_{i=1}^{m}(Z_j^i - \mu_j)^2 $

最后，对每个样本的值减去均值再除以标准差来将这一个 mini-batch 的样本的分布转化为标准正态分布：</font>

$ \tilde{Z_j} = \frac{Z_j - \mu_j}{\sqrt{\sigma^2 + \varepsilon}} $

此处加上 </font>$ \varepsilon $ 是为了避免分母为0。</font>

但是，批归一化存在一些缺陷，例如：</font>

+ 当显存有限，mini-batch 较小时，Batch Norm 取的样本的均值和方差不能反映全局的统计分布信息，从而导致效果变差；</font>
+ 对于在时间维度展开的 RNN，不同句子的同一分布大概率不同，所以 Batch Norm 的归一化会失去意义；</font>
+ 在训练时，Batch Norm 需要保存每个 step 的统计信息（均值和方差）。在测试时，由于变长句子的特性，测试集可能出现比训练集更长的句子，所以对于后面位置的 step，是没有训练的统计量使用的；</font>
+ 应用 Batch Norm，每个 step 都需要去保存和计算 batch 统计量，耗时又耗力</font>

</font>

## 2.2 Layer Normalization</font>
Layer Norm 在每个样本上计算其所有层的均值和方差，从而使每个样本的分布达到稳定。Layer Norm 的归一化方式其实和 Batch Norm 是完全一样的，只是统计统计量的维度不同</font>

```python
class LayerNorm(nn.Module):
    ''' Layer Norm 层'''
    def __init__(self, features, eps=1e-6):
        super().__init__()
    # 线性矩阵做映射
    self.a_2 = nn.Parameter(torch.ones(features))
    self.b_2 = nn.Parameter(torch.zeros(features))
    self.eps = eps

    def forward(self, x):
    # 在统计每个样本所有维度的值，求均值和方差
        mean = x.mean(-1, keepdim=True) # mean: [bsz, max_len, 1]
    std = x.std(-1, keepdim=True) # std: [bsz, max_len, 1]
    # 注意这里也在最后一个维度发生了广播
    return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

```



# 残差神经网络
在神经网络中，随着层数的增加，为了避免模型退化，采用了residual思想，使用残差连接每一个子层。即下一层的输入不仅仅是上一层的输出，还包括上一层的输入。

$ x = x + MultiHeadSelfAttention(LayerNorm(x)) $

$ output = x + FNN(LayerNorm(x)) $

我们在代码实现中，通过在层的 forward 计算中加上原值来实现残差连接：</font>

```python
# 注意力计算
h = x + self.attention.forward(self.attention_norm(x))
# 经过前馈神经网络
out = h + self.feed_forward.forward(self.fnn_norm(h))
```

# Encoder
## Encoder Layer
Transformer的encoder是由N个重复的Encoer Layer组成，每一个Encoder Layer包括一个注意力层和一个前馈神经网络。因此，我们可以首先实现一个Encoder Layer，再完成Encoder。

```python
class EncoderLayer(nn.Module):
    '''Encoder层'''
    def __init__(self, args):
        super().__init__()
        # 一个 Layer 中有两个 LayerNorm，分别在 Attention 之前和 MLP 之前
        self.attention_norm = LayerNorm(args.n_embd)
        # Encoder 不需要掩码，传入 is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.fnn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x):
        # Layer Norm
        norm_x = self.attention_norm(x)
        # 自注意力
        h = x + self.attention.forward(norm_x, norm_x, norm_x)
        # 经过前馈神经网络
        out = h + self.feed_forward.forward(self.fnn_norm(h))
        return out
```

## Encoder
```python
class Encoder(nn.Module):
    '''Encoder 块'''
    def __init__(self, args):
        super(Encoder, self).__init__() 
        # 一个 Encoder 由 N 个 Encoder Layer 组成
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x):
        "分别通过 N 层 Encoder Layer"
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
```

通过 Encoder 的输出，就是输入编码之后的结果。</font>

# Decoder
## Decoder Layer
与Encoder类似，我们首先可以构建Decoder Layer然后组装Decoder。与Encoder不同，Decoder由两个注意力层和一个FNN组成。第一个注意力层是一个mask attention，使得每一个token只能关注该token之前的注意力分数；第二个注意力层是一个多头注意力层，该层将使用第一个注意力层的输出作为query，使用Encoder的输出作为Key、Value来计算注意力分数。最后，再经过FNN。

```python
class DecoderLayer(nn.Module):
  '''解码层'''
    def __init__(self, args):
        super().__init__()
        # 一个 Layer 中有三个 LayerNorm，分别在 Mask Attention 之前、Self Attention 之前和 MLP 之前
        self.attention_norm_1 = LayerNorm(args.n_embd)
        # Decoder 的第一个部分是 Mask Attention，传入 is_causal=True
        self.mask_attention = MultiHeadAttention(args, is_causal=True)
        self.attention_norm_2 = LayerNorm(args.n_embd)
        # Decoder 的第二个部分是 类似于 Encoder 的 Attention，传入 is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.ffn_norm = LayerNorm(args.n_embd)
        # 第三个部分是 MLP
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x, enc_out):
        # Layer Norm
        norm_x = self.attention_norm_1(x)
        # 掩码自注意力
        x = x + self.mask_attention.forward(norm_x, norm_x, norm_x)
        # 多头注意力
        norm_x = self.attention_norm_2(x)
        h = x + self.attention.forward(norm_x, enc_out, enc_out)
        # 经过前馈神经网络
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
```

## Decoder
```python
class Decoder(nn.Module):
    '''解码器'''
    def __init__(self, args):
        super(Decoder, self).__init__() 
        # 一个 Decoder 由 N 个 Decoder Layer 组成
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x, enc_out):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)
```

以上，就完成了Transformer的核心部分--Encoder、Decoder结构。

---

# Transformer
## Embedding 层
通过分词器tokenizer，将对应token转化为embedding。Embedding的输入通常为（batch_size, seq_len, 1)矩阵，输出为(batch_size, seq_len, emb_dim)。

这里，我们使用torch中的Embedding层进行搭建

```python
self.tok_embedding = nn.Embedding(arg.vocab_size, args.dim)
```

## 位置编码
attenion实现了：

+ 数据并行计算
+ 权重计算

与此同时，其失去了位置信息：在注意力机制的计算过程中，对于序列中的每一个 token，其他各个位置对其来说都是平等的，即“我喜欢你”和“你喜欢我”在注意力机制看来是完全相同的，但无疑这是注意力机制存在的一个巨大问题。</font>

为了使用序列的序列信息，保留序列的相对位置关系，Transformer采用了位置编码机制。</font>

位置编码，即根据序列中 token 的相对位置对其进行编码，再将位置编码加入词向量编码中。位置编码的方式有很多，Transformer 使用了正余弦函数来进行位置编码（绝对位置编码Sinusoidal），其编码方式为：</font>

$ PE(pos, 2i) = sin(pos/10000^{2i/d_model})
 $

$ PE(pos, 2i+1) = cos(pos/10000^{2i/d_model}) $

上述式子中，pos表示了token在句子中的位置（即token在sequence中的位置），$ 2i $和$ 2i+1
 $表示embedding的偶数还是奇数位置，可以看到对于一个token其embedding奇数位置和偶数位置，Transformer 采用了不同的函数进行编码。</font>

```python
class PositionalEncoding(nn.Module):
    '''位置编码模块'''

    def __init__(self, args):
        super(PositionalEncoding, self).__init__()
        # Dropout 层
        # self.dropout = nn.Dropout(p=args.dropout)

        # block size 是序列的最大长度
        pe = torch.zeros(args.block_size, args.n_embd)
        position = torch.arange(0, args.block_size).unsqueeze(1)
        # 计算 theta
        div_term = torch.exp(
            torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd)
        )
        # 分别计算 sin、cos 结果
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将位置编码加到 Embedding 结果上
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x

```

---

### register_buffer
1. **其中**`self.register_buffer("pe", pe)`**是干什么的？**

`register_buffer(name, tensor)` 会把一个 不会被训练（不需要梯度）但又属于模型状态 的张量加入到模块中。

也就是说：

+ 它会跟着模型一起保存 / 加载（state_dict里有它）
+ 它会被自动移动到 GPU / CPU（.to(device) 时自动迁移）
+ 不会被 optimizer 更新（不参与训练）
2. **如果不写**`register_buffer`**会怎么样？**

如果直接写：

```python
self.pe = pe
```

❌ **模型保存时，state_dict 里不会包含这个 pe。**

这会导致你 load 模型时缺少 pe 信息。

❌ `.to(device)` 不会自动把 pe 移动到 GPU。

你需要手动：

```python
self.pe = self.pe.to(device)
```

❌ 优化器有可能错误地把它当成模型参数

所以 **`**register_buffer**`** 用来定义：

✔ 固定、不可训练的数据  
✔ 在模型中扮演状态角色（如位置编码、mask、均值方差等）

> 小结： register_buffer = “不可训练但需要保存 & 自动迁移的模型状态”
>

---



# 完整的Transfomer
先上图：



 Transformer 模型结构</font>

上图为原论文《Attention is all you need》配图，LayerNorm 层放在了 Attention 层后面，也就是“Post-Norm”结构，但在其发布的源代码中，LayerNorm 层是放在 Attention 层前面的，也就是“Pre Norm”结构。考虑到目前 LLM 一般采用“Pre-Norm”结构（可以使 loss 更稳定），本文在实现时采用“Pre-Norm”结构。</font>

如图，经过 tokenizer 映射后的输出先经过 Embedding 层和 Positional Embedding 层编码，然后进入上一节讲过的 N 个 Encoder 和 N 个 Decoder</font>，最后经过一个线性层和一个 Softmax 层就得到了最终输出。</font>

基于之前所实现过的组件，我们可以实现完整的 Transformer 模型：</font>

```python
class Transformer(nn.Module):
   '''整体模型'''
    def __init__(self, args):
        super().__init__()
        # 必须输入词表大小和 block size
        assert args.vocab_size is not None
        assert args.block_size is not None
        self.args = args
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(args.vocab_size, args.n_embd),
            wpe = PositionalEncoding(args),
            drop = nn.Dropout(args.dropout),
            encoder = Encoder(args),
            decoder = Decoder(args),
        ))
        # 最后的线性层，输入是 n_embd，输出是词表大小
        self.lm_head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        # 初始化所有的权重
        self.apply(self._init_weights)

        # 查看所有参数的数量
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    '''统计所有参数的数量'''
    def get_num_params(self, non_embedding=False):
        # non_embedding: 是否统计 embedding 的参数
        n_params = sum(p.numel() for p in self.parameters())
        # 如果不统计 embedding 的参数，就减去
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    '''初始化权重'''
    def _init_weights(self, module):
        # 线性层和 Embedding 层初始化为正则分布
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    '''前向计算函数'''
    def forward(self, idx, targets=None):
        # 输入为 idx，维度为 (batch size, sequence length, 1)；targets 为目标序列，用于计算 loss
        device = idx.device
        b, t = idx.size()
        assert t <= self.args.block_size, f"不能计算该序列，该序列长度为 {t}, 最大序列长度只有 {self.args.block_size}"

        # 通过 self.transformer
        # 首先将输入 idx 通过 Embedding 层，得到维度为 (batch size, sequence length, n_embd)
        print("idx",idx.size())
        # 通过 Embedding 层
        tok_emb = self.transformer.wte(idx)
        print("tok_emb",tok_emb.size())
        # 然后通过位置编码
        pos_emb = self.transformer.wpe(tok_emb) 
        # 再进行 Dropout
        x = self.transformer.drop(pos_emb)
        # 然后通过 Encoder
        print("x after wpe:",x.size())
        enc_out = self.transformer.encoder(x)
        print("enc_out:",enc_out.size())
        # 再通过 Decoder
        x = self.transformer.decoder(x, enc_out)
        print("x after decoder:",x.size())

        if targets is not None:
            # 训练阶段，如果我们给了 targets，就计算 loss
            # 先通过最后的 Linear 层，得到维度为 (batch size, sequence length, vocab size)
            logits = self.lm_head(x)
            # 再跟 targets 计算交叉熵
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        else:
            # 推理阶段，我们只需要 logits，loss 为 None
            # 取 -1 是只取序列中的最后一个作为输出
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

```

