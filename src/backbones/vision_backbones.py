import torch.nn as nn
import torchvision.models as models
import torch
import os 
def get_vision_backbone(model_name, pretrained=True,weights=None):
    """
    Load a vision model from torchvision and adapt it for multilabel classification.
    """
    # Load a pretrained model
    
    
    
    
    # If a weights file is provided, load the checkpoint from the ckpt folder nan
    if weights is None :
        raise ValueError("Weights should be provided (default is imagenet)")
    elif weights == 'random':
        model = getattr(models, model_name)(pretrained=False)
        
    elif weights == 'imagenet':
        model = getattr(models, model_name)(pretrained=True)
    else:
        weights_path = os.path.join('ckpt', weights)
        print(f"===== Loading weights from {weights_path} =====")
        checkpoint = torch.load(weights_path)
        
        # If the checkpoint contains a 'state_dict', use it
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load the weights into the model
        model.load_state_dict(state_dict, strict=False)  # strict=False allows partial loading

    # Replace the last fully connected layer
    in_features = model.fc.in_features
    # model.fc = nn.Linear(in_features, num_labels)
    model.fc = nn.Identity() 
    print(f"===== Loading Vision Backbone {model_name} from torchvision =====")
    #Display number of parameters trainable and not trainable in the model
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Model has {trainable_params/10**6}M trainable parameters and {non_trainable_params} non-trainable parameters")

    return model, in_features