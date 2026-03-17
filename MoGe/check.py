import torch
ckpt = torch.load("/home/data1/szq/Megadepth/benchmarkmodel/Moge2/MoGe/vits-normal.pt", map_location="cpu")
print("Config in Checkpoint:", ckpt.get('model_config', 'No config found'))