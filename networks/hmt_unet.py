from .mamba_vision import MambaVision, mamba_vision_T, MambaVision_sim
import torch
from torch import nn

# by mingya zhang  dg20330034@smail.nju.edu.cn 2024 08 16

class HMTUNet(nn.Module):
    
    def __init__(self, 
                    input_channels=1,
                    num_classes=1,
                    depths=[1, 3, 3, 2],
                    num_heads=[2, 4, 4, 8],
                    window_size=[4, 4, 7, 7],
                    dim=96,
                    in_dim=64,
                    mlp_ratio=4,
                    resolution=128,
                    drop_path_rate=0.2,
                    load_ckpt_path=None,
                    **kwargs):
        
        super().__init__()
        
        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        
        self.hmtunet = MambaVision_sim(
                in_chans=input_channels,
                depths=depths,
                num_heads=num_heads,
                window_size=window_size,
                dim=dim,
                num_classes=num_classes,
                in_dim=in_dim,
                mlp_ratio=mlp_ratio,
                resolution=resolution,
                drop_path_rate=drop_path_rate,
        )
        
        
        
    
    def forward(self, x):
        return self.hmtunet(x)
    
    
    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.hmtunet.state_dict()
            model_checkpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = model_checkpoint['state_dict']
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict) 
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.hmtunet.load_state_dict(model_dict)
            
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            model_dict = self.hmtunet.state_dict()
            model_checkpoint = torch.load(self.load_ckpt_path)
            pretrained_order_dict = model_checkpoint['state_dict']
            pretrained_dict = {}
            for k,v in pretrained_order_dict.items():
                if 'levels.0' in k:
                    new_k = k.replace('levels.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'levels.1' in k: 
                    new_k = k.replace('levels.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'levels.2' in k: 
                    new_k = k.replace('levels.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'levels.3' in k: 
                    new_k = k.replace('levels.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
                    
            # decoder 
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.hmtunet.load_state_dict(model_dict)
            
            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")
            
def main():
    # 假设输入图像大小为 224x224，3通道
    input_tensor = torch.randn(1, 3, 224, 224).cuda()  # batch_size=1

    # 实例化模型
    model = HMTUNet(
        input_channels=1,
        num_classes=1,
        resolution=224
    ).cuda()

    # 打印网络结构
    print(model)

    # 前向传播测试
    output = model(input_tensor)
    print("Output shape:", output.shape)

if __name__ == '__main__':
    main()

        
    