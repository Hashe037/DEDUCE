import torch
import os
import torch.nn as nn
from torchvision import models, transforms


class EmbeddingExtractor(nn.Module):
    """Extract embeddings from images using a backbone."""
    
    def __init__(self, model_name='resnet18', model_path=None, device= torch.device('cuda')):
        super().__init__()

        if model_name=='resnet18':
            self.model = models.resnet18(pretrained=False)
            if model_path:
                # Load finetuned weights
                original_fc = self.model.fc
                self.model.fc = nn.Linear(self.model.fc.in_features, 2)
                try:
                    self.model.load_state_dict(torch.load(model_path, map_location=device))
                except:  #backbone issue
                    checkpoint = torch.load(model_path)
                    state_dict = checkpoint['state_dict']
                    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
                    self.model.load_state_dict(state_dict, strict=False)  #how to incoproate device?
        elif model_name=='enet_b0':
            self.model = models.efficientnet_b0(weights='IMAGENET1K_V1')
            if model_path:
                # Load finetuned weights
                num_features = self.model.classifier[1].in_features
                self.model.classifier[1] = nn.Linear(num_features, 2)
                # import pdb
                # pdb.set_trace()
                try:
                    self.model.load_state_dict(torch.load(model_path, map_location=device))
                except:  #state dict is a variable
                    self.model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])


        elif model_name == 'dinov2':
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')  #do i need device here?
            return  #don't need to remove final layer
        else:
            raise ValueError
        
        # Remove the final classification layer to get embeddings        
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        
    def forward(self, x):
        x = self.model(x)
        return torch.flatten(x, 1)
    
    @staticmethod
    def get_transforms():
        """Standard ImageNet transforms."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    

    @staticmethod
    def get_transforms_np():
        """Standard ImageNet transforms."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            # transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    