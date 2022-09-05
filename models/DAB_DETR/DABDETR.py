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

import torch
from torch import nn
import torch.nn.functional as F

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss)
from .transformer import build_transformer


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2.):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """

    # inputs 是 logits，targets 是 one-hot 形式，与 inputs 的 shape 一致

    # logit -> probability
    prob = inputs.sigmoid()
    # (bs,num_queries,num_classes) 交叉熵损失
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # re-weight，降低容易样本(prob 接近1的正样本 & prob 接近0的负样本)对 loss 的贡献
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    # 平衡正负样本对 loss 的贡献
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # 先对 query(dim 1 对应的就是 num_queries) 求平均，得到每张图片在各类别上均摊在每个 query 上的损失，
    # 然后在所有图片、所有类别上求和，最后计算这批图片平均在每个物体上的损失是多少
    # (num_boxes 代表 1 个 batch 的物体总数量)
    # Scalar tensor
    return loss.mean(1).sum() / num_boxes


class DABDETR(nn.Module):
    """ This is the DAB-DETR module that performs object detection """
    def __init__(
        self, backbone, transformer, num_classes, num_queries, 
        aux_loss=False,  # 是否对 Decoder 每层都进行监督
        iter_update=True,  # 在 Decoder 每层都会更新(校正)参考点(anchor boxes)位置
        query_dim=4,  # 4维分别对应 anchor box 的 x,y,w,h
        bbox_embed_diff_each_layer=False,  # 用于校正参考点位置的 MLP 层在 Decoder 各层是否不共享参数
        random_refpoints_xy=False,  # anchor box 的 x,y 是否采用随机位置而非可学习的(x,y 仅在 Decoder 第一层采用随机位置，在后面层会由于校正作用而更新)
    ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            iter_update: iterative update of boxes
            query_dim: query dimension. 2 for point and 4 for box.
            bbox_embed_diff_each_layer: dont share weights of prediction heads. Default for False. (shared weights.)
            random_refpoints_xy: random init the x,y of anchor boxes and freeze them. (It sometimes helps to improve the performance)
        """

        super().__init__()

        # DAB-DETR 是 300 个，DETR 是 100 个
        self.num_queries = num_queries

        self.transformer = transformer
        hidden_dim = transformer.d_model

        # 分类头
        # 注意：num_classes 不包含背景
        # 与原始的 DETR 不同，在原始的 DETR 中，分类头的输出维度是 num_classes + 1
        # 两者的分类 loss 也不同，这里是 BCE(focal loss)，相当于多个二分类，而 DETR 是 CE
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        # bbox 校正模块是否在 Transformer Decoder 每层都不一样(不共享参数)
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        if bbox_embed_diff_each_layer:
            # 原实现是直接 hard code 成 6 个 MLP 层，但实际这个数量要和 Decoder 的层数对齐，于是我进行了修改
            # self.bbox_embed = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(6)])
            self.bbox_embed = nn.ModuleList([MLP(hidden_dim, hidden_dim, 4, 3) for _ in range(self.transformer.decoder.num_layers)])
        else:
            # 3 层 MLP，输出4维，对应 bbox(xywh) 的偏移量
            self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # Setting query dim
        # 这里是4，代表 4d anchor box
        self.query_dim = query_dim
        # 2代表位置先验使用点的形式：x,y，而4则代表使用框的形式：x,y,w,h
        assert query_dim in [2, 4]

        # 4d anchor box 位置先验，注意，它并非位置 query，位置 query 是由它经过正余弦位置编码后再经过 MLP 层输出得到
        self.refpoint_embed = nn.Embedding(num_queries, query_dim)
        # 是否随机初始化anchor box的中心位置，并固定(即不训练)它
        self.random_refpoints_xy = random_refpoints_xy
        if random_refpoints_xy:
            # 均匀分布
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
            # 取消 x,y 的梯度，使得每张图片在输入到 Decoder 第一层时，使用的位置先验中心点(x,y)都是随机均匀分布的，
            # 而后每一层再由校正模块(bbox_embed)进行调整。
            # 这样可在一定程度上避免模型基于训练集而学到过份的归纳偏置(即过拟合)，更具泛化性
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

        # 将 backbone 输出的特征图维度映射到 transformer 的隐层维度，以便喂给 transformer
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone

        self.aux_loss = aux_loss
        self.iter_update = iter_update
        # 如果要在 Decoder 每层校正参考点，则要设置 Decoder 的 bbox_embed
        # (因为在 Decoder 的代码实现里，将其设置为了 None，所以需要在这里设置)
        if self.iter_update:
            self.transformer.decoder.bbox_embed = self.bbox_embed

        # Init prior_prob setting for focal loss
        prior_prob = 0.01
        # Sigmoid 函数：1/(1+e^(-x))
        # 因此，若 weight 初始化为0，则输出就是 bias_value，
        # 经过 Sigmoid 函数后(将 bias_value 代入 Sigmoid 公式)，最终结果就是 prior_prob
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        # Init bbox_embed
        # 注意，仅将最后一层 MLP 初始化为 0。
        # 这样，在第一次反向传播后，仅最后一层 MLP 有梯度，得以更新，使得后续反向传播有梯度传导到前面的 MLP 层。
        # 效果是，在第一次前向过程时，不会校正参考点，因为 bbox_embed 的最终输出是 0；
        # 并且这种做法可以避免过拟合到第一个 batch(因其前面层 MLP 不会就此更新)，一定程度上加速了收敛。
        if bbox_embed_diff_each_layer:
            for bbox_embed in self.bbox_embed:
                nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, width, height). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        # i. Backbone
        # 如果输入不是 NestedTensor 类型的对象，那么必须是1个 Tensor 列表(目前的实现仅支持这样)，
        # 然后通过 nested_tensor_from_tensor_list 将输入转换成 NestedTensor 类型的对象
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        # 输出金字塔特征和对应的位置嵌入向量(通过调整了温度参数的位置编码得到)
        features, pos = self.backbone(samples)
        # 取出最后一层的特征，feature 里每个都是 NestedTensor 对象，因此可以“分解”出对应的特征向量和 mask
        src, mask = features[-1].decompose()
        assert mask is not None

        # ii. Transformer
        # Default pipeline
        embedweight = self.refpoint_embed.weight
        # 返回 Decoder 所有层的输出(包含隐层向量 & 参考点)
        # (num_layers,bs,num_queries,d_model), (num_layers,bs,num_queries,4)
        hs, reference = self.transformer(self.input_proj(src), mask, embedweight, pos[-1])
        
        # iii. 预测每个对象(query)的位置：
        # 基于它们的参考点(位置先验)，然后将它们的隐层向量输入 bbox_embed 得到校正的偏移量，
        # 最后由参考点+偏移量得到位置：x,y,w,h
        if not self.bbox_embed_diff_each_layer:
            reference_before_sigmoid = inverse_sigmoid(reference)
            # (num_layers,bs,num_queries,4)
            tmp = self.bbox_embed(hs)
            tmp[..., :self.query_dim] += reference_before_sigmoid
            outputs_coord = tmp.sigmoid()
        else:
            reference_before_sigmoid = inverse_sigmoid(reference)

            outputs_coords = []
            for lvl in range(hs.shape[0]):
                # (bs,num_queries,4)
                tmp = self.bbox_embed[lvl](hs[lvl])
                tmp[..., :self.query_dim] += reference_before_sigmoid[lvl]
                outputs_coord = tmp.sigmoid()
                outputs_coords.append(outputs_coord)
            # (num_layers,bs,num_queries,4)
            outputs_coord = torch.stack(outputs_coords)

        # iv. 预测每个对象(query)的类别
        outputs_class = self.class_embed(hs)

        # v. “收集”预测结果
        # 最后一层解码出来的预测结果
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        # 中间层解码出来的预测结果
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # This is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
        """

        super().__init__()

        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """

        # indices: List[Tuple(Tensor, Tensor)], len(indices) = batch_size,
        # indices 是 query & target 匹配对索引，list 中的每个元组代表每张图片匹配的预测结果和目标物体索引，
        # 元组的第一个张量(记为 i)是 query indices，第二个(记为 j)是 target indices，并且，它们的数量相等，
        # len(i) == len(j) == min(num_queries, num_targets_a_picture)，以下记为 num_matched

        assert 'pred_logits' in outputs
        # (bs,num_queries,num_classes)
        src_logits = outputs['pred_logits']

        # 返回的是一个元组：(batch_indices, query_indices)，1-to-1
        idx = self._get_src_permutation_idx(indices)
        # (bs*num_matched,)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # (bs,num_queries)，其中每个元素的值都初始化为 num_classes(如果是 COCO 目标检测数据集的话这个值就是 91)
        # 于是，对于没有匹配到目标的 query，其被分配到的 class id 就会是 num_classes
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # 由于会在 one-hot 向量的最后一维中将 target_classes 的值设置进去，
        # 而 target_classes 的最大值是 num_classes，因此这个 one-hot 向量的最后一维要初始化成 num_classes + 1，
        # 也就是 src_logits.shape[2] + 1 
        # (bs,num_queries,num_classes+1)
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        # 因为 src_logits 的 dim 2 仅有 num_classes 个，所以将 one-hot 向量的 dim 2 最后一个维度截掉，
        # 从而 one-hot 向量与 src_logits 的维度完全一致，进而可以进行 binary focal loss 的计算
        # 注意，这样玩法，对于没有被匹配到目标的 query，其对应的 one-hot 为全0。
        # 因为 target_classes 初始化的值为 num_classes，所以没有匹配到目标的 query 会被“分配”到类别 id: num_classes，
        # 而现在 one-hot 向量将这个维度截掉了，于是使得这些没有匹配到目标的 query 对应的 one-hot 向量为全0
        # (bs,num_queries,num_classes)
        target_classes_onehot = target_classes_onehot[:, :, :-1]
        # 在 sigmoid_focal_loss 中计算的是平均在每个 query 上的损失，
        # 这里乘 query 数量(src_logits.shape[1]) 得到在所有 queries 上的损失
        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot, num_boxes,
            alpha=self.focal_alpha, gamma=2
        ) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]

        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """

        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        
        # Count the number of predictions that are NOT "no-object" (which is the last class)

        # 注意：这里作者直接沿用了原始 DETR 的实现，但是用在这里会出错
        # 因为在原始 DETR 中，logits 最后一维是 num_classes + 1，包含了背景；
        # 而在这里的 DAB-DETR 实现中，logits 的最后一维是 num_classes，
        # 因此 pred_logits.shape[-1] - 1 并非是背景类

        # (bs,)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        # |pred_num_objects - num_objects|
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}

        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """

        assert 'pred_boxes' in outputs
        # 返回的是一个元组：(batch_indices, query_indices)
        idx = self._get_src_permutation_idx(indices)
        # (bs*num_queries,4)
        src_boxes = outputs['pred_boxes'][idx]
        # (bs*num_queries,4)
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # src_boxes & target_boxes 是经过排序的，因此用 torch.diag() 取出 Iou 矩阵的对角线元素，
        # 表示取出的是匹配的预测与标签对，仅在这些匹配对之间计算 loss
        loss_giou = 1 - torch.diag(
            box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(src_boxes),
                box_ops.box_cxcywh_to_xyxy(target_boxes)
            )
        )
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        # Calculate the x,y and h,w loss
        with torch.no_grad():
            losses['loss_xy'] = loss_bbox[..., :2].sum() / num_boxes
            losses['loss_hw'] = loss_bbox[..., 2:].sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """

        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # Permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])

        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # Permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])

        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets, return_indices=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
             return_indices: used for vis. if True, the layer 0-5 indices will be returned as well.
        """

        # outputs: {'pred_logits': xxx, 'pred_boxes': xxx, 'aux_outputs': xxx}
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)
        if return_indices:
            indices0_copy = indices
            indices_list = []

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                if return_indices:
                    indices_list.append(indices)

                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue

                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}

                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if return_indices:
            indices_list.append(indices0_copy)
            return losses, indices_list

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self, num_select=100) -> None:
        super().__init__()
        self.num_select = num_select

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """

        num_select = self.num_select
        # (bs,num_queries,num_classes), (bs,num_queries,4)
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        # logit -> probability
        prob = out_logits.sigmoid()
        # 每张图片预测分数(预测是物体的概率)最高的 topk 类别，
        # 以下 topk_indexes 是在 0 ~ num_queries * num_classes - 1 范围中选择 topk 计算出来的
        # (bs,num_queries,num_classes)->(bs,num_queries*num_classes)->(bs,num_select)
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), num_select, dim=1)
        
        scores = topk_values
        # 其中每个值代表 query index(0~num_queries-1)
        # 注意，这里可能会多次选到同一个 query，因为有可能同一个 query 在多个类别中的分数都排在了 topk 中
        topk_boxes = topk_indexes // out_logits.shape[2]
        # 每张图片的 topk queries 对应的物体类别(0~num_classes-1)
        labels = topk_indexes % out_logits.shape[2]

        # 筛选出最有可能是物体的 topk 个 bounding boxes
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # (bs,num_select,4)
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        # (bs,4)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        # 每个预测出来的 bounding box 都乘以其对应图片的宽、高以还原尺度
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results


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


def build_DABDETR(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    # 注意，这个 num_classes 是不包含背景(no-obj)的。
    # 对于 COCO 目标检测数据集，由于其最大的物体类别 id 为 90(0~90)，于是这里 num_classes 需要设置为 91
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    # i. 模型
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    model = DABDETR(
        backbone,
        transformer,
        num_classes=num_classes,  # 目标类别数，不包括背景
        num_queries=args.num_queries,  # 默认 300
        aux_loss=args.aux_loss,  # 是否要在 Transformer(Decoder) 每层都做监督(计算 loss)，默认 True
        iter_update=True,  # 迭代(在 Transformer Decoder 每层)更新 anchor box(x,y,w,h)
        query_dim=4,  # 对应 4d 的位置先验：anchor box
        random_refpoints_xy=args.random_refpoints_xy,  # 是否随机初始化 anchor box 的中心位置，若是，则每来一批数据时，anchor 的位置都是随机分布的，避免过拟合到训练集
    )
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
    
    # ii. 损失函数相关
    # 标签分配策略：基于匈牙利算法的二分匹配
    matcher = build_matcher(args)

    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    # TODO this is a hack
    # 对 Decoder 中间层进行监督
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)

    # cardinality 计算的是预测出来的物体数量与图片中真实物体数量的差值，仅用作 log，不涉及梯度的反向传播
    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]

    # 损失函数
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha, losses=losses)
    criterion.to(device)

    # iii. 后处理
    postprocessors = {'bbox': PostProcess(num_select=args.num_select)}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
