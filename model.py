import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, max_seq_length, pos_dropout=0.1,
                 trans_dropout=0.1):
        super(DecoderOnlyTransformer, self).__init__()

        self.d_model = d_model

        # 位置编码层
        self.positional_encoding = nn.Embedding(max_seq_length, d_model)

        # Token嵌入层
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=trans_dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)

        # Dropout层
        self.dropout = nn.Dropout(pos_dropout)

    def forward(self, src, src_mask):
        # 构建序列长度与batch size大小相同的位置编码
        seq_length, batch_size = src.size()
        position = torch.arange(seq_length, device=src.device).unsqueeze(1).repeat(1, batch_size)
        pos_encoding = self.positional_encoding(position)

        # 嵌入与位置编码相加
        src = self.dropout(pos_encoding + self.token_embedding(src))

        # 调整尺寸以符合Transformer预期的shape：S x B x E => B x S x E
        src = src.permute(1, 0, 2)

        # 解码器层
        memory = None  # Decoder-only模型中不需要encoder的输出作为memory
        output = self.transformer_decoder(src, memory, tgt_mask=src_mask)

        # 将输出通过线性层以生成预测
        output = self.output_layer(output)

        return output


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


# 假设一些超参数和模型参数
vocab_size = 1000
d_model = 512
nhead = 8
num_decoder_layers = 6
dim_feedforward = 2048
max_seq_length = 100
pos_dropout = 0.1
trans_dropout = 0.1

# 初始化模型
model = DecoderOnlyTransformer(vocab_size, d_model, nhead, num_decoder_layers, dim_feedforward, max_seq_length,
                               pos_dropout, trans_dropout)

# 假设一些输入
src = torch.randint(0, vocab_size, (35, 20))  # (seq_length, batch_size)

# 生成掩码
src_mask = generate_square_subsequent_mask(35)

# 前向传播
output = model(src, src_mask)

print(output.shape)  # 预期输出：(batch_size, seq_length, vocab_size)
