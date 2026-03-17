import torch
import json
from model.v2 import MoGeModel

# ==========================================
# 这里填入你想验证的配置 (以 Base 为例)
# ==========================================
config_base = {
    "encoder": {
        "backbone": "dinov2_vitb14",       # Base
        "intermediate_layers": [2, 5, 8, 11],
        "dim_out": 768
    },
    "neck": {
        "dim_in": [770, 2, 2, 2, 2],       # 768 + 2
        "dim_out": None,
        "dim_res_blocks": [768, 256, 128, 64, 32],
        "num_res_blocks": [0, 2, 2, 2, 0],
        "res_block_in_norm": "none",
        "res_block_hidden_norm": "none",
        "resamplers": ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"]
    },
    "points_head": {
        "dim_in": [768, 256, 128, 64, 32],
        "dim_out": [None, None, None, None, 3],
        "dim_res_blocks": [768, 256, 128, 64, 32],
        "num_res_blocks": [0, 1, 1, 1, 0],
        "res_block_in_norm": "none",
        "res_block_hidden_norm": "none",
        "resamplers": ["conv_transpose", "conv_transpose", "conv_transpose", "bilinear"]
    },
    # ... 其他 head 类似，省略 ...
    "scale_head": {
        "dims": [768, 768, 768, 1]
    },
    "remap_output": "exp",
    "num_tokens_range": [1024, 2048]
}

def test_model_build():
    print("🚀 正在尝试构建 MoGe v2 (Base) 模型...")
    
    try:
        # 1. 尝试实例化
        model = MoGeModel(
            encoder=config_base['encoder'],
            neck=config_base['neck'],
            points_head=config_base['points_head'],
            scale_head=config_base['scale_head'],
            # 如果你要测 mask_head 和 normal_head，也要把它们的配置放进去
            mask_head=None, 
            normal_head=None
        )
        print("✅ 模型实例化成功！配置参数格式正确。")
        
        # 2. 尝试 Forward (假数据)
        print("🔄 正在进行一次 Forward 测试...")
        dummy_input = torch.randn(1, 3, 518, 518) # 模拟一张图
        num_tokens = 1024
        
        # 如果配置的维度不对，这里一定会报 RuntimeError: mat1 and mat2 shapes cannot be multiplied
        output = model.forward(dummy_input, num_tokens=num_tokens)
        
        print("✅ Forward 成功！输出形状检查：")
        if 'points' in output:
            print(f"   Points: {output['points'].shape}")
        if 'metric_scale' in output:
            print(f"   Scale:  {output['metric_scale'].shape}")
            
        print("\n🎉 结论：该配置是有效的，可以用于训练。")

    except Exception as e:
        print("\n❌ 测试失败！配置有误。")
        print(f"错误详情: {e}")

if __name__ == "__main__":
    test_model_build()