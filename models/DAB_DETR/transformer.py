# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

import math
import copy
from typing import Optional
from util.misc import inverse_sigmoid

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .attention import MultiheadAttention


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()

        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)

        return x


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)

    scale = 2 * math.pi
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

    if pos_tensor.size(-1) == 2:
        # (n_query,bs,128*2)
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        # (n_query,bs,128*4)
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))

    return pos


class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_queries=300, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=4,
                 keep_query_pos=False, query_scale_type='cond_elewise',
                 num_patterns=0,
                 modulate_hw_attn=True,
                 bbox_embed_diff_each_layer=False,
                 ):

        super().__init__()

        # Encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # Decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before, keep_query_pos=keep_query_pos)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos, query_scale_type=query_scale_type,
                                          modulate_hw_attn=modulate_hw_attn,
                                          bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)

        # 对多维(dim > 1)的参数使用 Xavier 初始化
        self._reset_parameters()
        # Decoder 中对位置 query 实施 scale(调整) 的方法
        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0
        if self.num_patterns > 0:
            self.patterns = nn.Embedding(self.num_patterns, d_model)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, refpoint_embed, pos_embed):
        # bs, c, h, w = src.shape
        bs = src.size(0)
        
        # Flatten NxCxHxW to HWxNxC
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        # Flatten NxHxW to NxHW
        mask = mask.flatten(1)
        # Encoder 对 Backbone 提取的特征进行编码，得到更“完善”的特征       
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        # 准备 Decoder 的 query(包含内容&位置部分)
        # 位置 query：每张图片共享 n_q 个位置先验，其中每个都是 4d anchor box(x,y,w,h)
        # (n_q,4)->(n_q,bs,4)
        refpoint_embed = refpoint_embed.unsqueeze(1).repeat(1, bs, 1)
        # query_embed = gen_sineembed_for_position(refpoint_embed)
        num_queries = refpoint_embed.shape[0]
        # tgt 是内容 query
        if self.num_patterns == 0:
            tgt = torch.zeros(num_queries, bs, self.d_model, device=refpoint_embed.device)
        else:
            # 若每个 query 需要关注多个对象，则将其内容与位置部分在对象之间进行重复，即多个对象会对应到同一个 query
            tgt = self.patterns.weight[:, None, None, :].repeat(1, self.num_queries, bs, 1).flatten(0, 1) # n_q*n_pat, bs, d_model
            refpoint_embed = refpoint_embed.repeat(self.num_patterns, 1, 1) # n_q*n_pat, bs, d_model

        # Decoder 基于位置先验与 Encoder 的编码特征做交互，
        # 解码得到隐层特征与校正后的参考点(4d anchor boxes)
        hs, references = self.decoder(
            tgt, memory, memory_key_padding_mask=mask,
            pos=pos_embed, refpoints_unsigmoid=refpoint_embed
        )
        
        return hs, references


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, d_model=256):
        super().__init__()

        # 实质上是 torch.nn.ModuleList()，其中每个子模块结构相同，但使用了 deepcopy，
        # 因此每个子模块不共享参数，是独立的实例
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        # 2层 MLP
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        
        output = src
        for _, layer in enumerate(self.layers):
            # rescale the content and pos sim
            # 基于当前层的 query 生成用于对位置嵌入做变换的张量
            # 相当于将内容与位置信息结合
            pos_scales = self.query_scale(output)
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos*pos_scales)

        if self.norm is not None:
            # LN
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(
        self, decoder_layer, num_layers, norm=None, return_intermediate=False, 
        d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise',
        modulate_hw_attn=False,
        bbox_embed_diff_each_layer=False,
    ):
        super().__init__()

        # 实质上是 torch.nn.ModuleList()，其中每个子模块结构相同，但使用了 deepcopy，
        # 所以每个子模块不共享参数，是独立的实例
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        # 传参进来是 LayerNorm
        self.norm = norm
        # 是否要返回每层的结果
        self.return_intermediate = return_intermediate
        assert return_intermediate
        # 传参进来是 4，代表 4d anchor box
        self.query_dim = query_dim

        # query_scale_type 默认是 cond_elewise
        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type

        # query_scale 的输入是 Decoder 每层的 output，
        # 输出的结果用于对参考点的位置嵌入(4d anchor box 经历了 正余弦位置编码后得到的向量)做变换
        # 对应于 paper 公式中的 PE(x)*PE(x_ref) 与 PE(y)*PE(y_ref)
        # 以下 MLP 的前三个参数分别代表：input_dim, hidden_dim, output_dim，
        # 最后一个参数代表 num_layers
        if query_scale_type == 'cond_elewise':
            # 代表每个元素都根据条件来生成，意思是每个 scale 向量都根据 Decoder 每层的 output 来生成(维度上一一对应)
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            # 代表根据条件来生成标量，即每个 scale 向量会根据 Decoder 每层的 output 来生成，但它是个标量
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            # 不根据 Decoder 的 output 来生成，而是用它来在 Embedding 中做 lookup
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))
        
        # self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        # ref_point_head 的输入是参考点的位置嵌入，因此输入维度就是位置嵌入的维度，输出用作位置 query
        # 由于位置先验是 4d(query_dim=4)的，于是由它经历正余弦编码后得到的 embedding 维度是 2 * d_model，
        # 因此这里用这个 ref_point_head 将其映射回 d_model 维度
        # 将以上改成以下这样较好理解：
        # 因为参考点(4d anchor box, query_dim=1)的位置嵌入是在各个维度上独立进行位置编码得到对应的向量，
        # 每个向量的维度是 d_model // 2。所以，将所有维度上的位置嵌入拼接后，维度就会是 query_dim * (d_model // 2)
        self.ref_point_head = MLP(query_dim * (d_model // 2), d_model, d_model, 2)
        
        # bbox 校正模块，输出 bbox 的偏移量(这里虽然设为 None，但其实会在外层模型中对这部分进行设置，通常是 MLP 结构)
        self.bbox_embed = None
        # 默认 256
        self.d_model = d_model
        # 是否使用尺度调制的自注意力，默认是
        self.modulate_hw_attn = modulate_hw_attn
        # bbox 校正模块在每层中是否不一样，默认 False，代表所有层共享
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        # 尺度调制的注意力(cross-attention)
        # 2层 MLP，输出维度：2，分别用于 x, y 坐标
        if modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)

        # 是否要在每层计算交叉注意力前将位置 query 与内容 query 结合(相加)
        # 相加前，会将位置 query 经过 projection(MLP)，即如下的 ca_qpos_proj
        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                # 第一层由于内容 query 没有位置信息，是必须会相加的，因此这里排除了第一层
                self.layers[layer_id + 1].ca_qpos_proj = None

    def forward(
        self, tgt, memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        refpoints_unsigmoid: Optional[Tensor] = None, # num_queries, bs, 4
    ):
        # (num_queries,bs,d_model)
        output = tgt

        # 收集每一层的输出结果
        intermediate = []
        # 将 x,y,w,h 缩放到 0~1
        reference_points = refpoints_unsigmoid.sigmoid()
        # 收集每层的参考点，除第一层外，每层均会进行校正
        ref_points = [reference_points] 

        for layer_id, layer in enumerate(self.layers):
            # i. 获得位置 query: 参考点->正余弦编码获得参考点的位置嵌入->2层 MLP 获得位置 query
            # (num_queries, batch_size, 4)
            obj_center = reference_points[..., :self.query_dim]
            # Get sine embedding for the query vector
            # 在4个维度(x,y,w,h)独立进行位置编码然后拼接在一起
            # (num_queries,bs,4*128=4*(d_model//2))
            query_sine_embed = gen_sineembed_for_position(obj_center)
            # 位置 query (num_queries,bs,d_model)
            query_pos = self.ref_point_head(query_sine_embed) 

            # ii. 基于当前层的 output 生成 transformation，对参考点的位置嵌入做变换，将内容与位置信息结合
            if self.query_scale_type != 'fix_elewise':
                # For the first decoder layer, we do not apply transformation over p_s
                # 因为在第一层中的 output 是初始化的 query(以上传参过来的 tgt)，
                # 所以基于它来生成对位置嵌入做变换的向量(pos_transformation)没有意义
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]
            # Apply transformation
            # 注意这里做了截断，在最后一维截取前 d_model 个维数
            # (num_queries,bs,d_model)
            query_sine_embed = query_sine_embed[..., :self.d_model] * pos_transformation

            # modulated HW attentions
            if self.modulate_hw_attn:
                # 基于当前层的 output 生成 x, y 坐标的调制参数(向量)，对应于 paper 公式中的 w_{q,ref} & h_{q,ref}
                # (num_queries,bs,2)
                refHW_cond = self.ref_anchor_head(output).sigmoid()
                # 分别调制 x, y 坐标并处以 w, h 归一化，从而将尺度信息注入到交叉注意力中
                # 后 self.d_model // 2 个维度对应 x 坐标，前 self.d_model // 2 个维度对应 y 坐标；
                # query_sine_embed[..., self.d_model // 2:] 对应 paper 公式的 PE(x)*PE(x_ref)
                # query_sine_embed[..., :self.d_model // 2] 对应 paper 公式的 PE(y)*PE(y_ref)
                # obj_center[..., 2] 是宽，对应 paper 公式的 w_q，obj_center[..., 3] 是高，对应 paper 公式的 h_q
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)

            # 解码
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0))

            # Iter update
            # 更新参考点
            if self.bbox_embed is not None:
                # 生成 bbox 的偏移量
                if self.bbox_embed_diff_each_layer:
                    tmp = self.bbox_embed[layer_id](output)
                else:
                    tmp = self.bbox_embed(output)

                # 由于参考点坐标是经过了 sigmoid，因此这里先反 sigmoid 再加上偏移量
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                # 更新参考点后重新经过 sigmoid 缩放
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
                # 最后一层的参考点会在外层模型中由整个 transformer 的输出经过 bbox_embed 得到偏移量，然后更新
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                
                # 作者说(本人说的哦) detach() 是因为让梯度的流通更友好，它想让每层的梯度仅受该层的输出影响
                reference_points = new_reference_points.detach()

                # 那么参考点(嵌入向量)如何被训练而学习？
                # 在 Decoder 第一层，参考点进来时，由其生成了 query 位置嵌入向量(query_sine_embed) & 位置 query(query_pos)，
                # 改层的 output 与它们都相关联。同时，由于每层的 input 都是前一层的 output，
                # 因此能够通过 loss 计算，最后反向传播得到梯度。

                # 另外，对于校正模块(box_embed)的优化，是通过隐层向量输入到它里面生成偏移量、结合最后一层解码出来的参考点得到
                # 对象的位置，最后与标签计算 loss 使得梯度得以传导过来。

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    # (num_layers,bs,num_queries,d_model)
                    torch.stack(intermediate).transpose(1, 2),
                    # (num_layers,bs,num_queries,4)
                    torch.stack(ref_points).transpose(1, 2),
                ]
            else:
                return [
                    # (num_layers,bs,num_queries,d_model)
                    torch.stack(intermediate).transpose(1, 2),
                    # (1,bs,num_queries,4) 
                    reference_points.unsqueeze(0).transpose(1, 2)
                ]

        # 这里增加一个维度是为了适配返回中间层(return_intermediate=True)的情况，
        # 使得当 return_intermediate=False 时，len(output)=0，代表仅返回最后一层的结果。
        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False):
        super().__init__()

        # 说明：对于以下变量命名，sa 即 self-attention 缩写，ca 即 cross-attention 缩写
        # 由以下参数可知，DAB-DETR 将 query & key 都拆解成内容(content)与位置(position)两部分来分别对应做交互，
        # 即 query content <-> key content, query position <-> key position

        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        # 这个 Linear 层用于对参考点的位置嵌入做映射
        # 由于在 DETR 中 query 的位置部分(learnable query) 没有像在 Encoder 那边一样是正余弦位置编码，
        # 因此这里引入了由参考点经历了正余弦位置编码得到的向量，也就是所说的参考点的位置嵌入
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        # 在计算交叉注意前，由于 q & k 是内容与位置部分的 concat，因此维度增加了一倍(d_model->d_model*2)
        self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of FeedForward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        # 是否要在计算交叉注意力前将内容 query 与位置 query 相加(对应地，内容 key 与位置 key 也会相加)
        self.keep_query_pos = keep_query_pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self, tgt, memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        query_sine_embed = None,
        is_first = False
):                     
        # ========== Begin of Self-Attention =============
        # attention->dropout->residual->norm
        if not self.rm_self_attn_decoder:
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt)  # target is the input of the first decoder layer, zero by default.
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            # num_queries, bs, n_model = q_content.shape
            # hw, _, _ = k_content.shape

            # 内容+位置
            q = q_content + q_pos
            k = k_content + k_pos

            # 取0是拿出注意力施加在 value 后的结果，1是注意力权重矩阵
            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            # ========== End of Self-Attention =============

            # 残差连接
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # attention->dropout->residual->norm
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)
        # 第一层由于没有足够的位置信息，因此默认要加上位置部分
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        # 将 query & key 的内容与通过正余弦位置编码得到的位置部分拼接，
        # 从而两者在交叉注意力中做交互是能够实现 内容与位置分别独立做交互，即：
        # q_content <-> k_content; q_position <-> k_position
        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        # query_sine_embed 由 4d anchor box 经历正余弦位置编码而来，实现了与 key 一致的位置编码方式
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        # 这里有个细节要注意，就是在拼接(concat)前要先将最后一维按注意力头进行划分，这样才能将各个头部的维度对应拼接
        # 否则，就会导致前面一些头部全部都是 q 的部分，而后面一些头部则全是 query_sine_embed 的部分。
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)

        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        # 取0是拿出注意力施加在 value 后的结果，1是注意力权重矩阵
        tgt2 = self.cross_attn(
            query=q,
            key=k,
            value=v, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )[0]               
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # FFN
        # linear->activation->dropout->linear
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        # dropout->residual->norm
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,  # 默认300
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,  # Transformer 有两种结构：pre-norm or post-norm
        return_intermediate_dec=True,  # True 表示要返回 Decoder 每层的结果
        query_dim=4,  # 对应 4d 的位置先验：anchor box
        activation=args.transformer_activation,  # 默认 prelu
        num_patterns=args.num_patterns,  # 每个 query 需要关注几个对象，默认是 0，代表每个 query 仅负责1个目标实例
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
        
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
