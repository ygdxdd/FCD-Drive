import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from typing import Callable, Optional
from torch import Tensor
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
# from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss
# from mmdet.models.losses import FocalLoss

def reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def weight_reduce_loss(loss: Tensor,
                       weight: Optional[Tensor] = None,
                       reduction: str = 'mean',
                       avg_factor: Optional[float] = None) -> Tensor:
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Optional[Tensor], optional): Element-wise weights.
            Defaults to None.
        reduction (str, optional): Same as built-in losses of PyTorch.
            Defaults to 'mean'.
        avg_factor (Optional[float], optional): Average factor when
            computing the mean of losses. Defaults to None.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def py_sigmoid_focal_loss(pred,
                          target,
                          weight=None,
                          gamma=2.0,
                          alpha=0.25,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float, optional): A balanced form for Focal Loss.
            Defaults to 0.25.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    # Actually, pt here denotes (1 - pt) in the Focal Loss paper
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    # Thus it's pt.pow(gamma) rather than (1 - pt).pow(gamma)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    if weight is not None:
        if weight.shape != loss.shape:
            if weight.size(0) == loss.size(0):
                # For most cases, weight is of shape (num_priors, ),
                #  which means it does not have the second axis num_class
                weight = weight.view(-1, 1)
            else:
                # Sometimes, weight per anchor per class is also needed. e.g.
                #  in FSAF. But it may be flattened of shape
                #  (num_priors x num_class, ), while loss is still of shape
                #  (num_priors, num_class).
                assert weight.numel() == loss.numel()
                weight = weight.view(loss.size(0), -1)
        assert weight.ndim == loss.ndim
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss


class LossComputer(nn.Module):
    def __init__(self,config: TransfuserConfig):
        self._config = config
        super(LossComputer, self).__init__()
        # self.focal_loss = FocalLoss(use_sigmoid=True, gamma=2.0, alpha=0.25, reduction='mean', loss_weight=1.0, activated=False)
        self.cls_loss_weight = config.trajectory_cls_weight
        self.reg_loss_weight = config.trajectory_reg_weight
        self.aux_reg_weight = config.aux_reg_weight
    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        #odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = 2*(odo_info_fut_x + 1.2)/56.9 -1
        odo_info_fut_y = 2*(odo_info_fut_y + 20)/46 -1
        #odo_info_fut_head = 2*(odo_info_fut_head + 2)/3.9 -1
        return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)
        
    def aux_loss(self, num_mode, poses_reg, mode_targets, cls_idx):
        # 5. 辅助回归损失：其余模式拟合锚点偏移的个性化目标（修复归一化+精准跳过最优模式）
        aux_reg_loss = torch.tensor(0.0, device=poses_reg.device)
        if mode_targets is not None:
            bs = poses_reg.shape[0]  # 获取批量大小，适配批量维度的cls_idx判断
            
            aux_count = 0
            
            # 遍历所有非最优模式，计算辅助损失
            for m in range(num_mode):
                is_best_mode = (cls_idx == m)  # (bs,) → 每个样本是否以m为最优模式   
                # 非最优模式：仅取当前模式下，非最优样本的轨迹（避免重复计算核心损失）
                pred_aux = poses_reg[~is_best_mode, m, :, :2]  # 过滤掉以m为最优模式的样本
                target_aux = mode_targets[~is_best_mode, m, :, :2]
                if len(pred_aux) == 0:
                    continue
                pred_aux_norm = self.norm_odo(pred_aux)  # 辅助预测轨迹归一化
                target_aux_norm = self.norm_odo(target_aux)  # 辅助目标轨迹归一化
        
                # 计算单模式辅助损失（仅针对非最优样本）
                aux_reg_loss += F.l1_loss(pred_aux_norm, target_aux_norm, reduction='mean')
                aux_count += 1
    
            # 平均辅助损失（避免模式数变化影响损失幅度）
            if aux_count > 0:
                aux_reg_loss = aux_reg_loss / aux_count  # 按有效模式数平均
                # 乘以辅助权重（远小于核心损失，避免稀释核心精度）
                aux_reg_loss = self.aux_reg_weight * aux_reg_loss
        return aux_reg_loss
    
    def forward(self, poses_reg, poses_cls, targets, classification_labels, mode_targets=None):
        """
        pred_traj: (bs, 20, 8, 3)
        pred_cls: (bs, 20)
        plan_anchor: (bs,20, 8, 2)
        targets['trajectory']: (bs, 8, 3)
        """
        bs, num_mode, ts, d = poses_reg.shape  
        
            
        target_traj = self.norm_odo(targets["trajectory"])
        #print(f"target_traj 形状: {target_traj}")
        #print(f"plan_anchor 形状: {plan_anchor}")
        batch_idx = torch.arange(bs, device=poses_reg.device)

        # 确保classification_labels是整数类型
        cls_idx = classification_labels.long()


        # Calculate cls loss using focal loss
        target_classes_onehot = torch.zeros([bs, num_mode],
                                            dtype=poses_cls.dtype,
                                            layout=poses_cls.layout,
                                            device=poses_cls.device)
        target_classes_onehot.scatter_(1, cls_idx.unsqueeze(1), 1)

        
        # Use py_sigmoid_focal_loss function for focal loss calculation
        #print(f"poses_cls 形状: {poses_cls}")
        #print(f"target_classes_onehot 形状: {target_classes_onehot}")
        loss_cls = self.cls_loss_weight * py_sigmoid_focal_loss(
            poses_cls,
            target_classes_onehot,
            weight=None,
            gamma=2.0,
            alpha=0.75,
            reduction='mean',
            avg_factor=None
        )

        '''
        # Calculate regression loss
            # === 添加分类准确率评估 ===
        with torch.no_grad():
            # 1. 计算概率
            probs = torch.sigmoid(poses_cls) + 1e-8  # (bs, 20)
            
            # 2. 各种准确率
            # Top-1准确率
            _, top1_idx = poses_cls.max(dim=1)
            top1_acc = (top1_idx == cls_idx).float().mean()
            
            # Top-3准确率
            _, top3_idx = poses_cls.topk(3, dim=1)
            top3_acc = (top3_idx ==         cls_idx.unsqueeze(1)).any(dim=1).float().mean()
            
            # Top-5准确率
            _, top5_idx = poses_cls.topk(5, dim=1)
            top5_acc = (top5_idx ==     cls_idx.unsqueeze(1)).any(dim=1).float().mean()
            
            # 3. 正确模态的概率统计
            correct_probs = probs[torch.arange(bs), cls_idx]
            mean_correct_prob = correct_probs.mean()
            median_correct_prob = correct_probs.median()
            min_correct_prob = correct_probs.min()
            max_correct_prob = correct_probs.max()
            
            # 4. 预测置信度分析
            top1_probs = probs[torch.arange(bs), top1_idx]
            confidence_mean = top1_probs.mean()
            
            print(f"\n=== 分类性能评估  ===")
            print(f"加权分类损失: {loss_cls.item():.6f} | Top1准确率: {top1_acc.item():.2%}")
            print(f"正确模态平均概率: {mean_correct_prob.item():.4f} | 预测置信度平均: {confidence_mean.item():.4f}")
            # 概率分布只打印前10步
            prob_bins = {
                '<0.3': (correct_probs < 0.3).float().mean(),
                '0.3-0.5': ((correct_probs >= 0.3) & (correct_probs < 0.5)).float().mean(),
                '0.5-0.7': ((correct_probs >= 0.5) & (correct_probs < 0.7)).float().mean(),
                '0.7-0.9': ((correct_probs >= 0.7) & (correct_probs < 0.9)).float().mean(),
                '>=0.9': (correct_probs >= 0.9).float().mean()
            }
            print(f"正确模态概率分布: {[(k, f'{v.item():.2%}') for k, v in prob_bins.items()]}")

        '''
        # 4. 核心回归损失：最优模式拟合GT（保留原有逻辑）
        best_reg = poses_reg[batch_idx, cls_idx, :, :2]  # (bs, 8, 2) → 仅取x,y
        best_reg_norm = self.norm_odo(best_reg) 
        # 新增：核心L2正则（轻量）
        core_l2_reg = 5e-4 * torch.norm(best_reg_norm, p=2, dim=-1).mean()
        core_reg_loss = self.reg_loss_weight * (F.l1_loss(best_reg_norm, target_traj[..., :2]) + core_l2_reg)

        
        # 5. 辅助回归损失：其余模式拟合锚点偏移的个性化目标（新增核心逻辑）
        aux_reg_loss = self.aux_loss(num_mode, poses_reg, mode_targets, cls_idx)

        # 6. 总损失：分类损失 + 核心回归损失 + 辅助回归损失
        total_reg_loss = core_reg_loss + aux_reg_loss
        ret_loss = loss_cls + total_reg_loss
        # import ipdb; ipdb.set_trace()
        # Combine classification and regression losses
        trajectory_loss_dict = {
            # 原始无权重损失（用于监控/调参）
            'traj_cls_loss_raw': loss_cls / self.cls_loss_weight,
            'traj_core_reg_loss_raw': core_reg_loss / self.reg_loss_weight,
            'traj_aux_reg_loss_raw': aux_reg_loss / self.aux_reg_weight,
        }
        return ret_loss, trajectory_loss_dict
