# get_tokens.py - 获取可用的token列表
import os
import hydra
from omegaconf import DictConfig
from pathlib import Path
from hydra.utils import instantiate
CONFIG_PATH = "/media/clw/data/xdd/DiffusionDrive/navsim/planning/script/config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"
@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def debug_sceneloader_behavior(cfg: DictConfig):
    """调试SceneLoader的实际行为"""
    from navsim.common.dataloader import SceneLoader, SceneFilter
    from navsim.common.dataclasses import SensorConfig
    from hydra.utils import instantiate
    
    print("🔍 调试SceneLoader行为...")
    
    # 创建SceneLoader
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path), 
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_all_sensors(),  
    )
    
    print(f"📊 SceneLoader类型: {type(scene_loader)}")
    print(f"📊 SceneLoader属性: {[attr for attr in dir(scene_loader) if not attr.startswith('_')]}")
    
    if hasattr(scene_loader, 'tokens'):
        tokens = scene_loader.tokens
        print(f"📊 tokens列表长度: {len(tokens)}")
        print(f"📋 前5个tokens: {tokens[:5]}")
    else:
        print("❌ SceneLoader没有tokens属性")
        return
    
    # 测试迭代器行为
    print(f"\n🔍 测试SceneLoader迭代器...")
    scene_count = 0
    
    for item in scene_loader:
        scene_count += 1
        print(f"  场景{scene_count}: 类型={type(item)}, 值={item}")
        
        # 检查前几个项目的类型
        if scene_count <= 3:
            print(f"     详细检查:")
            print(f"       类型: {type(item)}")
            print(f"       值: {item}")
            print(f"       长度: {len(item) if hasattr(item, '__len__') else 'N/A'}")
            
            # 如果是字符串，检查内容
            if isinstance(item, str):
                print(f"       字符串内容: {item[:100]}...")
            
            # 检查是否有scene_metadata
            if hasattr(item, 'scene_metadata'):
                print(f"       ✅ 有scene_metadata属性")
            else:
                print(f"       ❌ 没有scene_metadata属性")
        
        if scene_count >= 5:  # 只检查前5个
            break
    
    print(f"\n📊 统计: 迭代器返回了{scene_count}个项目")

if __name__ == "__main__":
    debug_sceneloader_behavior()

