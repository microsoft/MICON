import torch
from torch import nn
from hydra.utils import instantiate

from model.MPN import MPN
from model.Resnet import ResNet34

def get_img_encoder(args):
    if args.img_encoder.from_pretrained is True:
        img_encoder = instantiate(args.img_encoder.model)
        if not args.img_encoder.finetune:
            for param in img_encoder.parameters():
                param.requires_grad = False
        if args.img_encoder.model_type == 'resnet':
            in_features = img_encoder.fc.in_features
            img_encoder.fc = nn.Linear(in_features, args.img_encoder.output_size // 5)
        elif args.img_encoder.model_type == 'vit':
            in_features = img_encoder.heads.head.in_features
            img_encoder.heads.head = nn.Linear(in_features, args.img_encoder.output_size // 5)
        else:
            raise NotImplementedError
    else:
        img_encoder = ResNet34()
    return img_encoder

def get_mol_encoder(args):
    if args.mol_encoder.from_pretrained == False:
        mol_encoder = MPN(args)
    return mol_encoder

def param_count(model: nn.Module) -> int:
    """
    Determines number of trainable parameters.
    :param model: An nn.Module.
    :return: The number of trainable parameters.
    """
    return sum(param.numel() for param in model.parameters() if param.requires_grad)