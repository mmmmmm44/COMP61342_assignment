from torchvision.models import efficientnet_b0, efficientnet_b4, resnet50, efficientnet_v2_s
import torch
from torchvision.transforms import v2 as T
from enum import Enum

class ModelType(Enum):
    EFFICIENTNET_B0 = "efficientnet-b0"
    EFFICIENTNET_B4 = "efficientnet-b4"
    RESNET50 = "resnet50"
    EFFICIENTNET_V2_S = "efficientnet-v2-s"

    def __str__(self):
        return self.value
    
    def available_models():
        return [model.value for model in ModelType]


def load_model(model_type: ModelType, out_features:int):
    if model_type == ModelType.EFFICIENTNET_B0:
        model = efficientnet_b0(weights="DEFAULT")

        model.classifier[-1] = torch.nn.Linear(in_features=1280, out_features=out_features)

        for param in model.parameters():
            param.requires_grad = True
        
    elif model_type == ModelType.RESNET50:
        model = resnet50(weights="DEFAULT")

        model.fc = torch.nn.Linear(in_features=2048, out_features=out_features)

        for param in model.parameters():
            param.requires_grad = True

    elif model_type == ModelType.EFFICIENTNET_B4:
        model = efficientnet_b4(weights="DEFAULT")

        model.classifier[-1] = torch.nn.Linear(in_features=1792, out_features=out_features)

        for param in model.parameters():
            param.requires_grad = True

    elif model_type == ModelType.EFFICIENTNET_V2_S:
        model = efficientnet_v2_s(weights="DEFAULT")

        model.classifier[-1] = torch.nn.Linear(in_features=1280, out_features=out_features)

        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return model

def get_default_transforms(model_type:ModelType):
    if model_type == ModelType.EFFICIENTNET_B0 or model_type == ModelType.RESNET50:
        return T.Compose([
            T.Resize((224, 224)),
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.ToPureTensor()
        ])
    elif model_type == ModelType.EFFICIENTNET_B4:
        return T.Compose([
            T.Resize((380, 380)),
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.ToPureTensor()
        ])
    elif model_type == ModelType.EFFICIENTNET_V2_S:
        return T.Compose([
            T.Resize((384, 384)),
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.ToPureTensor()
        ])
    else:
        raise ValueError(f"Unsupported model type: {model_type}")