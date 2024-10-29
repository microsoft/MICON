import torch
from torch import nn
from hydra.utils import instantiate
from unimol_tools import UniMolRepr

from model.MPN import MPN
from model.MLP import MLP
from model.vit import VisionTransformer



def get_img_encoder(args):
    if args.img_encoder.from_pretrained is True:
        img_encoder = instantiate(args.img_encoder.model)
        if not args.img_encoder.finetune:
            for param in img_encoder.parameters():
                param.requires_grad = False
        else:
            for param in img_encoder.parameters():
                param.requires_grad = False
            for param in img_encoder.layer4.parameters():
                param.requires_grad = True
            # for param in img_encoder.parameters():
            #     param.requires_grad = True
        if args.img_encoder.model_type == 'resnet':
            in_features = img_encoder.fc.in_features
            img_encoder.fc = nn.Linear(in_features, args.img_encoder.output_size // 5)
        elif args.img_encoder.model_type == 'vit':
            in_features = img_encoder.heads.head.in_features
            img_encoder.heads.head = nn.Linear(in_features, args.img_encoder.output_size // 5)
        else:
            raise NotImplementedError
    else:
        img_encoder = VisionTransformer(224, 16, 1024, 8, 16, 1000)
    return img_encoder

def get_mol_encoder(args):
    if args.mol_encoder.model == "Unimol":
        mol_encoder = UniMolRepr(data_type='molecule')
    elif args.mol_encoder.model == "MLP":
        mol_encoder = MLP(input_dim=1024, hidden_dim=1024, output_dim=512, n_layers=4)
    elif args.mol_encoder.model == "MPN":
        raise NotImplementedError
    return mol_encoder

def param_count(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.
    :param model: An nn.Module.
    :return: The number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)