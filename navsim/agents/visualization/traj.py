import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
matplotlib.use('Agg')  # 在导入pyplot之前设置
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List, Dict, Any
import hydra
# 导入您的模块
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import Scene, Frame, Camera, Trajectory
from navsim.visualization.bev import add_configured_bev_on_ax, add_trajectory_to_bev_ax
from navsim.visualization.camera import add_camera_ax
from navsim.visualization.config import BEV_PLOT_CONFIG, TRAJECTORY_CONFIG, CAMERAS_PLOT_CONFIG
from navsim.visualization.plots import configure_ax, plot_bev_with_agent, configure_bev_ax, plot_cameras_frame, configure_all_ax
import logging
from pathlib import Path
from omegaconf import DictConfig
from hydra.utils import instantiate
from navsim.common.dataloader import SceneLoader, SceneFilter
from navsim.common.dataclasses import SensorConfig

logger = logging.getLogger(__name__)

CONFIG_PATH = "/media/clw/data/xdd/DiffusionDrive/navsim/planning/script/config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"

def load_scene_by_token(
    cfg: DictConfig, 
    token: str
) -> Optional[Scene]:
    """
    根据token加载场景（基于您提供的SceneLoader结构）
    
    Args:
        token: 场景token
        navsim_log_path: 数据路径
        scene_filter_config: 场景过滤器配置
    
    Returns:
        Scene对象或None
    """
    try:

        # 创建SceneLoader（使用您提供的参数结构）
        scene_loader = SceneLoader(
            sensor_blobs_path=Path(cfg.sensor_blobs_path), 
            data_path=Path(cfg.navsim_log_path),
            scene_filter=instantiate(cfg.train_test_split.scene_filter),
            sensor_config=SensorConfig.build_all_sensors(),  
        )
        if hasattr(scene_loader, 'get_scene_from_token') and callable(scene_loader.get_scene_from_token):
            try:
                scene = scene_loader.get_scene_from_token(token)
                if scene is not None:
                    print(f"✅ 使用get_scene_from_token方法成功加载场景")
                    return scene
                else:
                    print(f"❌ get_scene_from_token返回None")
            except Exception as e:
                print(f"❌ get_scene_from_token方法失败: {e}")
    except Exception as e:
        print(f"❌ 加载场景失败: {e}")
        return None

def project_trajectory_to_camera(trajectory_3d: np.ndarray, camera: Camera) -> np.ndarray:
    """
    将3D轨迹点投影到相机图片坐标系
    """
    try:
        # 获取相机参数（需要根据您的Camera数据结构调整）
        if hasattr(camera, 'intrinsics') and hasattr(camera, 'extrinsics'):
            K = camera.intrinsics  # 相机内参
            T = camera.extrinsics  # 相机外参
        else:
            # 使用默认相机参数（需要替换为真实值）
            K = np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]])  # 内参矩阵
            T = np.eye(4)  # 外参矩阵
        
        # 添加齐次坐标
        N = trajectory_3d.shape[0]
        points_3d_homo = np.hstack([trajectory_3d, np.ones((N, 1))])
        
        # 世界坐标系 -> 相机坐标系
        points_camera = (T @ points_3d_homo.T).T
        
        # 过滤掉相机后面的点
        valid_mask = points_camera[:, 2] > 0.1  # z > 0.1m
        points_camera_valid = points_camera[valid_mask]
        
        if len(points_camera_valid) == 0:
            return np.array([])
        
        # 相机坐标系 -> 像素坐标系
        points_image_homo = (K @ points_camera_valid[:, :3].T).T
        
        # 归一化
        points_image = points_image_homo[:, :2] / points_image_homo[:, 2:3]
        
        return points_image
        
    except Exception as e:
        print(f"⚠️ 3D投影失败: {e}")
        return np.array([])


def plot_trajectory_on_front_camera(
    scene: Scene, 
    agent: AbstractAgent
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots trajectory on front camera visualization with correct coordinate alignment
    """
    # 获取最后一帧
    frame_idx = scene.scene_metadata.num_history_frames - 1
    frame = scene.frames[frame_idx]
    front_camera = frame.cameras.cam_f0
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=BEV_PLOT_CONFIG["figure_size"])
    
    # 1. 显示前视相机图片
    add_camera_ax(ax, front_camera)
    
    # 2. 获取轨迹
    agent_input = scene.get_agent_input()
    trajectory = agent.compute_trajectory(agent_input)
    
    # 3. 正确的3D到2D投影
    if trajectory is not None and len(trajectory.poses) > 0:
        # 将2D轨迹转换为3D（假设在地平面上，z=0）
        trajectory_3d = np.hstack([
            trajectory.poses[:, :2],  # x, y
            np.zeros((len(trajectory.poses), 1))  # z=0 (地面高度)
        ])
        
        # 3D到2D投影
        projected_points = project_trajectory_to_camera(trajectory_3d, front_camera)
        
        if len(projected_points) > 0:
            # 画真实的投影轨迹
            ax.plot(
                projected_points[:, 0], projected_points[:, 1],
                color=TRAJECTORY_CONFIG["agent"]["line_color"],
                alpha=TRAJECTORY_CONFIG["agent"]["line_color_alpha"],
                linewidth=TRAJECTORY_CONFIG["agent"]["line_width"],
                linestyle=TRAJECTORY_CONFIG["agent"]["line_style"],
                marker=TRAJECTORY_CONFIG["agent"]["marker"],
                markersize=TRAJECTORY_CONFIG["agent"]["marker_size"],
                markeredgecolor=TRAJECTORY_CONFIG["agent"]["marker_edge_color"],
                zorder=TRAJECTORY_CONFIG["agent"]["zorder"],
                label='Planned Trajectory (3D Projection)'
            )
            
            print(f"  ✅ 成功投影 {len(projected_points)}/{len(trajectory.poses)} 个轨迹点")
    
    # 4. 添加标题
    ax.set_title(f"Front Camera View with Trajectory Projection", 
                 fontsize=14, fontweight='bold', pad=20)
    
    # 5. 添加图例
    ax.legend(loc='upper right', fontsize=10)
    
    # 6. 关闭坐标轴
    ax.set_xticks([])
    ax.set_yticks([])
    
    return fig, ax

def plot_trajectory_on_bev_camera(
    scene: Scene, 
    agent: AbstractAgent
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plots trajectory on front camera and bev visualization 
    """
    human_trajectory = scene.get_future_trajectory()
    agent_trajectory = agent.compute_trajectory(scene.get_agent_input())
    frame_idx = scene.scene_metadata.num_history_frames - 0
    frame = scene.frames[frame_idx]
    fig, ax = plt.subplots(3, 3, figsize=CAMERAS_PLOT_CONFIG["figure_size"])

    add_camera_ax(ax[0, 0], frame.cameras.cam_l0)
    add_camera_ax(ax[0, 1], frame.cameras.cam_f0)
    add_camera_ax(ax[0, 2], frame.cameras.cam_r0)

    add_camera_ax(ax[1, 0], frame.cameras.cam_l1)
    add_configured_bev_on_ax(ax[1, 1], scene.map_api, frame)
    #add_trajectory_to_bev_ax(ax[1, 1], human_trajectory, TRAJECTORY_CONFIG["human"])
    #add_trajectory_to_bev_ax(ax[1, 1], agent_trajectory, TRAJECTORY_CONFIG["agent"])
    add_camera_ax(ax[1, 2], frame.cameras.cam_r1)

    add_camera_ax(ax[2, 0], frame.cameras.cam_l2)
    add_camera_ax(ax[2, 1], frame.cameras.cam_b0)
    add_camera_ax(ax[2, 2], frame.cameras.cam_r2)

    configure_all_ax(ax)
    configure_bev_ax(ax[1, 1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.99, bottom=0.01)
    
    return fig, ax


def save_figure(fig: plt.Figure, output_path: str, dpi: int = 300) -> str:
    """
    Saves matplotlib figure to file
    :param fig: matplotlib figure
    :param output_path: output file path
    :param dpi: image resolution
    :return: saved file path
    """
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=dpi,
        bbox_inches='tight',
        facecolor='white'
    )
    plt.close(fig)
    return output_path

#def plot_mutli() -> Tuple[plt.Figure, plt.Axes]:
    


def generate_two_visualizations(
    scene: Scene, 
    agent: AbstractAgent, 
    output_dir: str = "./visualizations"
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generates two visualizations: front camera trajectory and BEV trajectory
    :param scene: navsim scene dataclass
    :param agent: navsim agent
    :param output_dir: output directory
    :return: tuple of saved file paths (camera_path, bev_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    token = getattr(scene.scene_metadata, 'token', 'unknown')
    
    print(f"📊 Generating visualizations for token: {token}")
    
    # 1. Front camera trajectory
    camera_path = os.path.join(output_dir, f"{token}_front_camera.png")
    try:
        fig, ax = plot_trajectory_on_bev_camera(scene, agent)
        save_figure(fig, camera_path)
        print(f"✅ Front camera trajectory saved: {camera_path}")
    except Exception as e:
        print(f"❌ Front camera visualization failed: {e}")
        camera_path = None
    
    # 2. BEV trajectory
    bev_path = os.path.join(output_dir, f"{token}_bev_trajectory.png")
    try:
        fig, ax = plot_bev_with_agent(scene, agent)
        save_figure(fig, bev_path)
        print(f"✅ BEV trajectory saved: {bev_path}")
    except Exception as e:
        print(f"❌ BEV visualization failed: {e}")
        bev_path = None
    
    return camera_path, bev_path


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """
    token = "647a42b4e5075f16"
     # 调用普通函数，手动传入cfg
    scene = load_scene_by_token(cfg, token)
    agent: AbstractAgent = instantiate(cfg.agent)
    
    # 生成可视化
    camera_path, bev_path = generate_two_visualizations(
        scene=scene,
        agent=agent,
        output_dir="./my_visualizations"    
    )


if __name__ == "__main__":
    main()
    

