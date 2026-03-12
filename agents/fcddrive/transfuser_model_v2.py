from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import copy 
import random
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter   #修改
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_backbone import TransfuserBackbone
from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
#from diffusers.schedulers import DDIMScheduler
from navsim.agents.diffusiondrive.modules.conditional_unet1d import ConditionalUnet1D,SinusoidalPosEmb
import torch.nn.functional as F
from navsim.agents.diffusiondrive.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention
from navsim.agents.diffusiondrive.modules.multimodal_loss import LossComputer
from torch.nn import TransformerDecoder,TransformerDecoderLayer
from typing import Any, List, Dict, Optional, Union

from .dic_models import DiC_models as DiT_models
from .diffusion import create_diffusion
#from diffusers.models import AutoencoderKL

class V2TransfuserModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._query_splits = [
            1,
            config.num_bounding_boxes,
        ]

        self._config = config
        self._backbone = TransfuserBackbone(config)

        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)  # 8x8 feature grid + trajectory
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        # usually, the BEV features are variable in size.
        self._bev_downscale = nn.Conv2d(512, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear(4 + 2 + 2, config.tf_d_model)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        self._trajectory_head = TrajectoryHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            plan_anchor_path=config.plan_anchor_path,
            config=config,
        )
        self.bev_proj = nn.Sequential(
            *linear_relu_ln(256, 1, 1,320),
        )
        self._perception_modules = nn.ModuleList([
            self._backbone, 
            self._bev_semantic_head, 
            self._agent_head,
            # 如果有其他冻结模块，可以加在这里
        ])
        self._planner_modules = nn.ModuleList([
            # 如果有其他冻结模块，可以加在这里
        ])
        self._diff_modules = nn.ModuleList([
            self._trajectory_head.dit,
            # 如果有其他冻结模块，可以加在这里
        ])
        self.freeze_perception_modules = False
        self.freeze_planner_modules = False
        self.freeze_diff_modules = False

    def freeze_perception_layers(self):
        """冻结所有指示层参数"""
        for module in self._perception_modules:
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        
        print("perception modules frozen:")
        for i, module in enumerate(self._perception_modules):
            print(f"  - {module.__class__.__name__}")
        
        self.freeze_perception_modules = True
        
    def unfreeze_perception_layers(self):
        """解冻指示层参数"""
        for module in self._perception_modules:
            for param in module.parameters():
                param.requires_grad = True
            module.train()
        
        print("perception modules unfrozen.")
        self.freeze_perception_modules = False
        
    def freeze_planner_layers(self):
        """冻结所有指示层参数"""
        for module in self._planner_modules:
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        
        print("Planner modules frozen:")
        for i, module in enumerate(self._planner_modules):
            print(f"  - {module.__class__.__name__}")
        
        self.freeze_planner_modules = True
        
    def unfreeze_planner_layers(self):
        """解冻指示层参数"""
        for module in self._planner_modules:
            for param in module.parameters():
                param.requires_grad = True
            module.train()
        
        print("Planner modules unfrozen.")
        self.freeze_planner_modules = False
    
    def freeze_diff_layers(self):
        """冻结所有指示层参数"""
        for module in self._diff_modules:
            for param in module.parameters():
                param.requires_grad = False
            module.eval()
        
        print("diff modules frozen:")
        for i, module in enumerate(self._diff_modules):
            print(f"  - {module.__class__.__name__}")
        
        self.freeze_diff_modules = True
        
    def unfreeze_diff_layers(self):
        """解冻指示层参数"""
        for module in self._diff_modules:
            for param in module.parameters():
                param.requires_grad = True
            module.train()
        
        print("diff modules unfrozen.")
        self.freeze_diff_modules = False
        
    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        if self.freeze_perception_modules:
            for module in self._perception_modules:
                module.eval()
        else:
            for module in self._perception_modules:
                module.train() 
        if self.freeze_planner_modules:
            for module in self._planner_modules:
                module.eval()
        else:
            for module in self._planner_modules:
                module.train() 
        if self.freeze_diff_modules:
            for module in self._diff_modules:
                module.eval()
        else:
            for module in self._diff_modules:
                module.train() 
        camera_feature: torch.Tensor = features["camera_feature"]
        lidar_feature: torch.Tensor = features["lidar_feature"]
        status_feature: torch.Tensor = features["status_feature"]

        batch_size = status_feature.shape[0]

        bev_feature_upscale, bev_feature, _ = self._backbone(camera_feature, lidar_feature)#bev_festure_upscale用于检测框和语义分割
        cross_bev_feature = bev_feature_upscale
        bev_spatial_shape = bev_feature_upscale.shape[2:]
        concat_cross_bev_shape = bev_feature.shape[2:]
        bev_feature = self._bev_downscale(bev_feature).flatten(-2, -1)
        bev_feature = bev_feature.permute(0, 2, 1)
        status_encoding = self._status_encoding(status_feature)              # [B, 1, 256]

        keyval = torch.concatenate([bev_feature, status_encoding[:, None]], dim=1)
        keyval += self._keyval_embedding.weight[None, ...]

        concat_cross_bev = keyval[:,:-1].permute(0,2,1).contiguous().view(batch_size, -1, concat_cross_bev_shape[0], concat_cross_bev_shape[1])
        # upsample to the same shape as bev_feature_upscale

        concat_cross_bev = F.interpolate(concat_cross_bev, size=bev_spatial_shape, mode='bilinear', align_corners=False)
        # concat concat_cross_bev and cross_bev_feature
        cross_bev_feature = torch.cat([concat_cross_bev, cross_bev_feature], dim=1)

        cross_bev_feature = self.bev_proj(cross_bev_feature.flatten(-2,-1).permute(0,2,1))
        cross_bev_feature = cross_bev_feature.permute(0,2,1).contiguous().view(batch_size, -1, bev_spatial_shape[0], bev_spatial_shape[1]) # [B, 320, 64, 64]
        query = self._query_embedding.weight[None, ...].repeat(batch_size, 1, 1)
        query_out = self._tf_decoder(query, keyval)

        bev_semantic_map = self._bev_semantic_head(bev_feature_upscale)                      # 获取 [64, 64]
        trajectory_query, agents_query = query_out.split(self._query_splits, dim=1)                  # [B, 1, 256] 和 [B, 30, 256]

        output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}

        trajectory = self._trajectory_head(trajectory_query,agents_query, cross_bev_feature,bev_spatial_shape,status_encoding[:, None],targets=targets,global_img=None)
        # 输入参数的维度                     [B, 1, 256] 和 [B, 30, 256]     [B, 320, 64, 64]   [64, 64]         [B, 1, 256]
        output.update(trajectory)

        agents = self._agent_head(agents_query)
        output.update(agents)

        return output

class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}

class ModeEmbedding(nn.Module):
    def __init__(self, num_mode, d_model):
        super().__init__()
        self.mode_emb = nn.Embedding(num_mode, d_model)
        # 初始化embedding，保证不同模式的特征区分度
        nn.init.normal_(self.mode_emb.weight, mean=0.0, std=0.02)
    
    def forward(self, bs, num_mode, device):
        # 生成模式embedding: [bs, num_mode, d_model]
        mode_ids = torch.arange(num_mode, device=device)  # [num_mode]
        mode_emb = self.mode_emb(mode_ids).unsqueeze(0).repeat(bs, 1, 1)
        return mode_emb


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_head, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_head, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_head, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, cross_kv):
        # 自注意力（捕捉轨迹/模式内部依赖）
        x = x + self.dropout(self.self_attn(x, x, x)[0])
        x = self.norm1(x)
        # 交叉注意力（融合外部特征）
        x = x + self.dropout(self.cross_attn(x, cross_kv, cross_kv)[0])
        x = self.norm2(x)
        # 前馈网络（特征映射）
        x = x + self.dropout(self.ffn(x))
        x = self.norm3(x)
        return x

class Plan_cls_Decoder(nn.Module):
    def __init__(
        self, 
        config, 
    ):
        super().__init__()
        num_poses = config.trajectory_sampling.num_poses
        d_model = config.tf_d_model
        self.num_mode = config.num_mode  # 新增：从config中读取模式数
        self.d_model = d_model
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(d_model, 1, 1, 512),
            nn.Linear(d_model, d_model),
        )
        self.mode_embedding = ModeEmbedding(self.num_mode, d_model)
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
        )
        self.bev_norm = nn.LayerNorm(d_model)
        self.bev_dropout = nn.Dropout(0.1)
        self.agent_transformer = TransformerBlock(d_model, config.tf_num_head, config.tf_dropout)
        self.ego_transformer = TransformerBlock(d_model, config.tf_num_head, config.tf_dropout)

        self.plan_cls_branch = nn.Sequential(
            *linear_relu_ln(config.tf_d_model, 1, 2),
            nn.Dropout(0.1), 
            nn.Linear(config.tf_d_model, 1),
        )
    
    def forward(self, 
                traj_feature, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                global_img=None):
        
        x = traj_feature
        bs, num_mode, ts, d = x.shape
        device = x.device
        #print(f"x.shape: {x.shape}")
        noisy_traj_points = x               #维度[bs,num_mode,ts,d ]
        # proj noisy_traj_points to the query
        traj_pos_embed = gen_sineembed_for_position(noisy_traj_points, hidden_dim=64)    #[bs, num_mode, 8, 64]
        traj_pos_embed = traj_pos_embed.flatten(-2)
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)        # 输出: [bs, num_mode, 256]
        mode_emb = self.mode_embedding(bs, num_mode, device)     # [bs, num_mode, d_model]
        traj_feature = traj_feature + mode_emb  # 融合模式专属特征
        "轨迹信息处理"
        #bev信息
        bev_feat = self.cross_bev_attention(traj_feature,noisy_traj_points,bev_feature,bev_spatial_shape)
        traj_feature = traj_feature + self.bev_dropout(bev_feat)  # 残差连接
        traj_feature = self.bev_norm(traj_feature)
        #agent信息
        traj_feature = self.agent_transformer(traj_feature, agents_query)
        #ego信息
        traj_feature = self.ego_transformer(traj_feature, ego_query)
        
        "轨迹分类结果"
        # Step5: 分类头（增强后的特征映射）
        plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)  # [bs, num_mode]

        return plan_cls
        
class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, plan_anchor_path: str,config: TransfuserConfig):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()

        #self.turn_optimizer = TurnAwareTrajectoryOptimizer()            #添加
        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = 20

        #self.diffusion_scheduler = DDIMScheduler(
        #    num_train_timesteps=1000,
        #    beta_schedule="scaled_linear",
        #    prediction_type="sample",
        #)


        plan_anchor = np.load(plan_anchor_path)

        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        ) # 20,8,2

        ######################################修改##################################################
        self.dit = DiT_models['DiC-XL'](
            config=config,                     #注意修改###############################
            input_size=20,
            #num_classes=args.num_classes,
            #**(opts['network_g'] if opts.get('network_g') is not None else dict())
        )
        self.diffusion_model = create_diffusion("100", config)
        self.loss_computer = LossComputer(config)
        self.traj_cls = Plan_cls_Decoder(config)
    
        #self.vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    
    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        #odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = 2*(odo_info_fut_x + 1.2)/56.9 -1
        odo_info_fut_y = 2*(odo_info_fut_y + 20)/46 -1
        #odo_info_fut_head = 2*(odo_info_fut_head + 2)/3.9 -1
        return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)
    
    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        #odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = (odo_info_fut_x + 1)/2 * 56.9 - 1.2
        odo_info_fut_y = (odo_info_fut_y + 1)/2 * 46 - 20
        #odo_info_fut_head = (odo_info_fut_head + 1)/2 * 3.9 - 2
        return torch.cat([odo_info_fut_x, odo_info_fut_y], dim=-1)
    
    def compute_headings_spline(self, trajectory_xy: torch.Tensor) -> torch.Tensor:
        """
        使用样条插值获得平滑轨迹，然后计算切线方向（航向角）
    
        Args:
            trajectory_xy: 输入轨迹点，形状为 (batch_size, num_poses, 2) 或 (batch_size, num_modes, num_poses, 2)
    
        Returns:
            headings: 航向角，形状与输入匹配，但最后一个维度为1，最终输出为 (batch_size, num_poses, 3) 或 (batch_size, num_modes, num_poses, 3)
        """
    
        # 保存原始设备
        device = trajectory_xy.device
    
        # 将数据移到CPU并转换为numpy
        trajectory_np = trajectory_xy.detach().cpu().numpy()
        original_shape = trajectory_np.shape
        
        # 处理不同的输入维度
        if trajectory_np.ndim == 3:
            # 形状: (batch_size, num_poses, 2)
            batch_size, num_poses, _ = trajectory_np.shape
            num_modes = 1
            trajectory_np = trajectory_np[:, np.newaxis, :, :]  # 增加modes维度
        elif trajectory_np.ndim == 4:
            # 形状: (batch_size, num_modes, num_poses, 2)
            batch_size, num_modes, num_poses, _ = trajectory_np.shape
        else:
            raise ValueError(f"不支持的输入维度: {trajectory_np.ndim}")
    
        # 初始化结果数组
        headings_np = np.zeros((batch_size, num_modes, num_poses, 1))
    
        # 时间点（假设等间隔）
        t = np.linspace(0, 1, num_poses)
    
        for b in range(batch_size):
            for m in range(num_modes):
                # 获取当前轨迹
                traj = trajectory_np[b, m]  # (num_poses, 2)
            
                # 检查轨迹是否有效（没有NaN或Inf）
                if np.any(np.isnan(traj)) or np.any(np.isinf(traj)):
                    # 无效轨迹，使用简单差分法
                    if num_poses > 1:
                        dx = np.diff(traj[:, 0])
                        dy = np.diff(traj[:, 1])
                        # 填充边界
                        dx = np.concatenate([dx[:1], dx])
                        dy = np.concatenate([dy[:1], dy])
                        headings = np.arctan2(dy, dx)
                    else:
                        headings = np.zeros(1)
                else:
                    try:
                        # 对x, y分量分别进行样条插值
                        spline_x = CubicSpline(t, traj[:, 0])
                        spline_y = CubicSpline(t, traj[:, 1])
                    
                        # 计算一阶导数（切线方向）
                        dx_dt = spline_x.derivative()(t)  # x方向速度
                        dy_dt = spline_y.derivative()(t)   # y方向速度
                    
                        # 计算航向角
                        headings = np.arctan2(dy_dt, dx_dt)
                    
                    except Exception as e:
                        # 样条插值失败，回退到简单差分法
                        print(f"样条插值失败，使用差分法: {e}")
                        if num_poses > 1:
                            dx = np.diff(traj[:, 0])
                            dy = np.diff(traj[:, 1])
                            dx = np.concatenate([dx[:1], dx])
                            dy = np.concatenate([dy[:1], dy])
                            headings = np.arctan2(dy, dx)
                        else:
                            headings = np.zeros(1)
            
                headings_np[b, m, :, 0] = headings
    
        # 转换回torch tensor
        headings_tensor = torch.from_numpy(headings_np).float().to(device)
    
        # 恢复原始形状
        if trajectory_xy.ndim == 3:
            headings_tensor = headings_tensor.squeeze(1)  # 移除modes维度
    
        return headings_tensor

    def add_headings_to_trajectory(self, trajectory_xy: torch.Tensor) -> torch.Tensor:
        """
        将航向角添加到轨迹中，生成完整的 (x, y, heading) 轨迹
    
        Args:
            trajectory_xy: 输入轨迹点，形状为 (batch_size, num_poses, 2)
    
        Returns:
            full_trajectory: 完整轨迹，形状为 (batch_size, num_poses, 3)
        """
        headings = self.compute_headings_spline(trajectory_xy)
        full_trajectory = torch.cat([trajectory_xy, headings], dim=-1)
        return full_trajectory
    
    def generate_mode_targets(self, plan_anchor, target_traj, closest_idx, device):
        """
        核心：生成每个模式的个性化目标（锚点偏移=模式锚点-GT偏移）
        参数：
            plan_anchor: [B, 20, 8, 2] → 处理后的锚点轨迹
            target_traj: [B, 1, 8, 2] → GT轨迹
            closest_idx: [B] → 每个样本的最优锚点索引
            device: 设备
        返回：
            mode_targets: [B, 20, 8, 2] → 每个模式的个性化目标
        """
        bs, num_mode, ts, d = plan_anchor.shape
    
        # 1. 提取每个样本的最优锚点轨迹 [B, 8, 2]
        best_anchor = torch.gather(plan_anchor, 1, closest_idx.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, ts, d)).squeeze(1)
    
        # 2. 计算最优锚点与GT的偏移 [B, 8, 2]
        gt_traj = target_traj.squeeze(1)  # [B, 8, 2]
        best_anchor_offset = best_anchor - gt_traj  # 最优锚点相对GT的偏移
    
        # 3. 为每个模式生成个性化目标：模式锚点 - 最优锚点偏移 = 模式锚点 - (最优锚点 - GT) = GT + (模式锚点 - 最优锚点)
        # 广播偏移到所有模式 [B, 20, 8, 2]
        best_anchor_offset_expand = best_anchor_offset.unsqueeze(1).repeat(1, num_mode, 1, 1)
        mode_targets = plan_anchor - best_anchor_offset_expand
    
        return mode_targets
    
    def forward(self, ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding, targets=None,global_img=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        if self.training:
            return self.forward_train(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,targets,global_img)
        else:
            return self.forward_test(ego_query, agents_query, bev_feature,bev_spatial_shape,status_encoding,global_img)


    def forward_train(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding, targets=None,global_img=None) -> Dict[str, torch.Tensor]:
        # 输入参数的维度   [B, 1, 256] 和 [B, 30, 256] [B, 320, 64, 64] [64, 64]     [B, 1, 256]
        bs = ego_query.shape[0]
        device = ego_query.device
        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs,1,1,1)

        ################################################################################################################
        #                                                  修改部分                                                      #
        ################################################################################################################
        bs, num_mode, num_poses, d = plan_anchor.shape
        target_traj = targets["trajectory"]
        target_traj = target_traj.unsqueeze(1)[..., :2]
        #print(target_traj.shape)
        #dist = torch.linalg.norm(target_traj - plan_anchor, dim=-1)
        #dist = dist.mean(dim=-1)
        #mode_idx = torch.argmin(dist, dim=-1)
        #for i in range(bs):
        #    plan_anchor[i, mode_idx[i]] = target_traj[i,0]
        # === 改进的多样化训练策略 ===
        classification_labels = torch.zeros(bs, dtype=torch.long, device=device)
        mode_targets = None  # 初始化模式个性化目标

        # 为整个batch选择策略
        strategy = random.choices(
            ['original', 'all_anchors', 'mixed'],
            weights=[0.7, 0.2, 0.1],
            k=1
        )[0]

        if strategy == 'original':
            # 1. 为每个样本找到最近的锚点（向量化）
            # target_traj: [B, 1, 8, 2], plan_anchor: [B, 20, 8, 2]
            dist = torch.linalg.norm(target_traj - plan_anchor, dim=-1)  # [B, 20, 8]
            dist = dist.mean(dim=-1)  # [B, 20]
            closest_idx = torch.argmin(dist, dim=1)  # [B]

            # 2. 用目标轨迹替换最近的锚点
            for i in range(bs):
                plan_anchor[i, closest_idx[i]] = target_traj[i, 0]  # target_traj[i, 0] shape: [8, 2]

            # 3. 分类标签
            classification_labels = closest_idx

        elif strategy == 'all_anchors':
            # 计算哪个锚点最接近目标（作为伪标签）
            dist = torch.linalg.norm(target_traj - plan_anchor, dim=-1)  # [B, 20, 8]
            dist = dist.mean(dim=-1)  # [B, 20]
            closest_idx = torch.argmin(dist, dim=1)  # [B]

            classification_labels = closest_idx
            # plan_anchor 保持不变

        elif strategy == 'mixed':
            # 为每个样本创建2-3个"好"轨迹
            for i in range(bs):
                num_good = random.randint(2, 3)
                good_indices = random.sample(range(plan_anchor.size(1)), num_good)

                for j, idx in enumerate(good_indices):
                    if j == 0:
                        # 第一个是真实目标
                        plan_anchor[i, idx] = target_traj[i, 0]
                        classification_labels[i] = idx
                    else:
                        # 其他是加噪的目标
                        noise = torch.randn_like(target_traj[i, 0]) * 0.15
                        plan_anchor[i, idx] = target_traj[i, 0] + noise

        ################################################################################################################
        #                             完结                               #        ################################################################################################################
        mode_targets = self.generate_mode_targets(plan_anchor, target_traj, classification_labels, device)
        timesteps = torch.randint(
            0, self.diffusion_model.num_timesteps,
            (bs,), device=device
        )
        

        poses_reg = self.diffusion_model.training_losses(self.dit, plan_anchor, timesteps, bev_feature=bev_feature, bev_spatial_shape=bev_spatial_shape, agents_query=agents_query, ego_query=ego_query, status_encoding=status_encoding, global_img=global_img, compute_cls=True, plan_anchor=plan_anchor, targets=targets, num_poses=self._num_poses, classification_labels=classification_labels)


        poses_cls = self.traj_cls(plan_anchor,bev_feature,bev_spatial_shape,agents_query,ego_query,global_img)
        
        trajectory_loss, trajectory_loss_dict = self.loss_computer(poses_reg, poses_cls, targets, classification_labels, mode_targets=mode_targets)
        
        mode_idx = poses_cls.argmax(dim=-1)
        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, num_poses, 2) #####num_poses是时间戳的数量
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
        return {"trajectory": best_reg,"trajectory_loss":trajectory_loss, "trajectory_loss_dict": trajectory_loss_dict}

    def forward_test(self, ego_query,agents_query,bev_feature,bev_spatial_shape,status_encoding,global_img) -> Dict[str, torch.Tensor]:

        bs = ego_query.shape[0]
        device = ego_query.device
        # 1. add truncated noise to the plan anchor
        plan_anchor = self.plan_anchor.unsqueeze(0).repeat(bs, 1, 1, 1)
        bs, num_mode, num_poses, d = plan_anchor.shape
        x_norm = self.norm_odo(plan_anchor)
        noise = torch.randn_like(x_norm)*0.08
        #t = torch.randint(
        #    0, self.diffusion_model_infer.num_timesteps,
        #    (bs,), device=device
        #)
        t = torch.full((bs,), self.diffusion_model.num_timesteps-1, device=device, dtype=torch.long)
        noisy_traj_points = self.diffusion_model.q_sample(x_norm, t, noise=noise)
        noisy_traj_points = torch.clamp(noisy_traj_points, min=-1, max=1)
        x_t = self.denorm_odo(noisy_traj_points)
        poses_reg = self.diffusion_model.ddim_sample_loop(
            self.dit, x_t.shape, x_t, bev_feature=bev_feature,
            bev_spatial_shape=bev_spatial_shape,
            agents_query=agents_query,
            ego_query=ego_query,
            status_encoding=status_encoding,
            global_img=global_img,compute_cls=False, clip_denoised=True
        )
        #poses_reg = self.diffusion_model.p_sample_loop1(
        #    self.dit, x_t.shape, x_t, bev_feature=bev_feature,
        #    bev_spatial_shape=bev_spatial_shape,
        #    agents_query=agents_query,
        #    ego_query=ego_query,
        #    status_encoding=status_encoding,
        #    global_img=global_img,compute_cls=False, clip_denoised=True
        #)[0]
        poses_cls = self.traj_cls(plan_anchor,bev_feature,bev_spatial_shape,agents_query,ego_query,global_img)

        mode_idx = poses_cls.argmax(dim=-1)
        mode_idx = mode_idx[..., None, None, None].repeat(1, 1, num_poses, 2)  #####num_poses是时间戳的数量
        topk_idx = torch.topk(poses_cls, k=5, dim=-1).indices
        mode_idx_2 = topk_idx[:,1][...,None,None,None].repeat(1,1,num_poses,2)
        best_reg_2 = self.add_headings_to_trajectory(torch.gather(poses_reg,1,mode_idx_2).squeeze(1))
        mode_idx_3 = topk_idx[:,2][...,None,None,None].repeat(1,1,num_poses,2)
        best_reg_3 = self.add_headings_to_trajectory(torch.gather(poses_reg,1,mode_idx_3).squeeze(1))
        mode_idx_4 = topk_idx[:,3][...,None,None,None].repeat(1,1,num_poses,2)
        best_reg_4 = self.add_headings_to_trajectory(torch.gather(poses_reg,1,mode_idx_3).squeeze(1))
        mode_idx_5 = topk_idx[:,4][...,None,None,None].repeat(1,1,num_poses,2)
        best_reg_5 = self.add_headings_to_trajectory(torch.gather(poses_reg,1,mode_idx_3).squeeze(1))
        best_reg = torch.gather(poses_reg, 1, mode_idx).squeeze(1)
        best_reg = self.add_headings_to_trajectory(best_reg)
        best_reg = torch.cat([best_reg, best_reg_2, best_reg_3, best_reg_4, best_reg_5], dim=0)
        return {"trajectory": best_reg}
