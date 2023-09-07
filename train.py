import os
import torch
import hydra
import pdb
import pickle as pkl
import numpy as np

from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import normalize

from model.micon import MICON
from utils import Averager, Logger, get_lr, get_img_encoder
from data.dataset import MoleculeImageDataset
from unimol_tools import UniMolRepr



def build_loaders(dataset, batch_size, collate_fn=None, mode='train', drop_last=False):
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=20,
        shuffle=True if mode == "train" else False,
        drop_last=drop_last
    )
    return dataloader

def get_weights_l2(model):
    weights_l2 = sum(p.detach().norm(2).item() ** 2 for p in model.parameters()) ** 0.5
    return weights_l2

def train(args, model, train_loader, valid_dataloader, optimizer, lr_scheduler, writer, logger):
    model.train()
    averager = Averager()
    current_step = 0
    current_epochs = 0
    total_steps = args.train.steps if args.train.epochs == -1 else args.train.epochs * len(train_loader)
    pbar = tqdm(total = total_steps)
    while current_step < total_steps:
        # logger.log_message(f"Epoch: {current_epochs + 1}")
        for _batch in train_loader:
            current_step += 1
            batch = {}
            for k, v in _batch.items():
                batch[k] = v.to(args.device) if torch.is_tensor(v) else v
            batch_dict = model.training_step(batch)
            loss = batch_dict['loss']
            logs = batch_dict['log']
            optimizer.zero_grad()
            loss.backward()
            # if step % 10 == 0:
            #     plot_grad_flow_v2(model.named_parameters())
            optimizer.step()
            lr_scheduler.step(loss)
            logs["weights_l2"] = get_weights_l2(model)
            logs["lr"] = get_lr(optimizer)
            averager.update({"loss": loss.item()})
            averager.update(logs)
            pbar.update(1)
            pbar.set_postfix(train_loss=loss.item(), lr=logs["lr"])
            if current_step % args.train.log_every == 0:
                averaged_stats = averager.average()
                writer.add_scalar("Train/loss", averaged_stats["loss"], current_step)
                for k,v in sorted(averaged_stats.items()):
                    writer.add_scalar(f"Train/{k}", v, current_step)
                logger.log_stats(
                stats=averaged_stats,
                step=current_step,
                prefix='train/'
            )
                averager.reset()
            if current_step % args.valid.per_steps == 0:
                eval(args, model, valid_dataloader, writer, logger, current_step)
                model.train()
            if args.train.save and current_step % args.train.save_per_steps == 0:
                save_state(model, optimizer, current_step, args)
        current_epochs += 1
        
    return 

def eval(args, model, valid_dataloader, writer, logger, current_step):
    model.eval()
    averager = Averager()
    _step = 0
    while _step < args.valid.steps:
        for _, _batch in enumerate(valid_dataloader):
            if _step >= args.valid.steps:
                break
            batch = {}
            for k, v in _batch.items():
                batch[k] = v.to(args.device) if torch.is_tensor(v) else v
            batch_dict = model.validation_step(batch)
            loss = batch_dict['loss']
            logs = batch_dict['log']
            averager.update({"loss": loss.item()})
            averager.update(logs)
            _step += 1
    averaged_stats = averager.average()
    writer.add_scalar("Valid/loss", averaged_stats["loss"], current_step)
    for k,v in sorted(averaged_stats.items()):
        writer.add_scalar(f"Valid/{k}", v, current_step)
    logger.log_stats(
    stats=averaged_stats,
    step=current_step,
    prefix='valid/'
    )
    
    return

def generate_embeddings(args, model, generate_dataloader):
    total_valid_steps = len(generate_dataloader)
    img_emb, img_generated_emb, f_name = [], [], []
    pbar = tqdm(total = total_valid_steps)
    for _step, _batch in enumerate(generate_dataloader):
        batch = {}
        for k, v in _batch.items():
            batch[k] = v.to(args.device) if torch.is_tensor(v) else v
        _img_emb, _img_generated_emb, _f_name = model.generate_image_embeddings(**batch)
        img_generated_emb.append(_img_generated_emb.detach().cpu().numpy())
        img_emb.append(_img_emb.detach().cpu().numpy())
        f_name.extend(_f_name)
        pbar.update(1)
    return np.concatenate(img_emb, axis=0), np.concatenate(img_generated_emb, axis=0), f_name

def generate_retreival(args, model, generate_dataloader, logger, step=10):
    pbar = tqdm(total = step)
    averager = Averager()
    _step = 0
    smis = []
    for _ in range(step):
        for _, _batch in enumerate(generate_dataloader):
            batch = {}
            for k, v in _batch.items():
                batch[k] = v.to(args.device) if torch.is_tensor(v) else v
            batch_dict = model.get_retrieval_scores(**batch)
            logs = batch_dict['log']
            smis.append(batch_dict['smis'])
            averager.update(logs)
        pbar.update(1)
    averaged_stats = averager.average()
    logger.log_stats(
    stats=averaged_stats,
    step=_step,
    prefix='test/'
    )
    return smis
        
def save_state(model, optimizer, step, args):
    if os.path.exists(args.train.save_dir) == False:
        os.makedirs(args.train.save_dir)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        f"{args.train.save_dir}/model_{step}.pt",
    )
    return  

def load_state(model, optimizer, args):
    checkpoint = torch.load(args.load.load_path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer
        
    

@hydra.main(config_path="config", config_name="default", version_base='1.1')
def main(args):
    writer = SummaryWriter()
    logger = Logger()
    img_enc = get_img_encoder(args)
    # mol_enc = get_mol_encoder(args)
    pretrained_mol_enc = UniMolRepr(data_type='molecule')
    
    OmegaConf.save(config=args, f=".hydra/config.yaml")
    
    model = MICON(args, img_enc, mol_enc=None).to(args.device)
    optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.train.optim.lr, weight_decay=args.train.optim.weight_decay
    )
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.train.optim.lr, momentum=0.9, weight_decay=args.train.optim.weight_decay)
    # lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode="min", patience=500, factor=0.9
    # )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=500, factor=0.9, min_lr=1e-6,
    )
    
    if args.mode == "train":
        # model, optimizer = load_state(model, optimizer, args)
        train_set = MoleculeImageDataset(args.data.train_path, args=args, sample_strategy=args.data.train_sample_strategy, \
                                        from_pretrained=pretrained_mol_enc, mode='train')
        valid_set = MoleculeImageDataset(args.data.valid_path, args=args, sample_strategy=args.data.valid_sample_strategy, \
                                        from_pretrained=pretrained_mol_enc, mode='test')

        train_dataloader = build_loaders(train_set, batch_size=args.train.batch_size, mode="train")
        valid_dataloader = build_loaders(valid_set, batch_size=args.valid.batch_size, mode="valid")
        train(args, model, train_dataloader, valid_dataloader, optimizer, lr_scheduler, writer=writer, logger=logger)
        
    if args.mode == "generate_embedding":
        generate_set_train = MoleculeImageDataset(args.data.generate_path, args=args, sample_strategy='img_only', \
                            from_pretrained=pretrained_mol_enc, mode='test', metadata="metadata_train.csv")
        generate_set_test = MoleculeImageDataset(args.data.generate_path, args=args, sample_strategy='img_only', \
                            from_pretrained=pretrained_mol_enc, mode='test', metadata="metadata_test.csv")
        # generate_set_test_ood = MoleculeImageDataset(args.data.generate_path, args=args, sample_strategy='img_only', \
        #                     from_pretrained=pretrained_mol_enc, mode='test', metadata="metadata_test_unseen.csv")
        generate_train_dataloader = build_loaders(generate_set_train, batch_size=64, mode="valid")
        generate_test_dataloader = build_loaders(generate_set_test, batch_size=64, mode="valid")
        # generate_test_ood_dataloader = build_loaders(generate_set_test_ood, batch_size=64, mode="valid")
        model, _ = load_state(model, optimizer, args)
        model.eval()
        img_emb, img_generated_emb, f_name = generate_embeddings(args, model, generate_train_dataloader)
        with open(args.data.generate_path + f"_new_embeddings_supcon_freeze_img_train_{args.ckpt}.pkl", 'wb') as f:
            logger.log_message("Saving embeddings... Size: {}".format(img_emb.shape))
            pkl.dump((img_emb, f_name), f)
        with open(args.data.generate_path + f"_new_embeddings_supcon_freeze_img_train_{args.ckpt}_generated.pkl", 'wb') as f:
            logger.log_message("Saving generated embeddings... Size: {}".format(img_generated_emb.shape))
            pkl.dump((img_generated_emb, f_name), f)
            
        img_emb, img_generated_emb, f_name = generate_embeddings(args, model, generate_test_dataloader)
        with open(args.data.generate_path + f"_new_embeddings_supcon_freeze_img_test_{args.ckpt}.pkl", 'wb') as f:
            logger.log_message("Saving embeddings... Size: {}".format(img_emb.shape))
            pkl.dump((img_emb, f_name), f)
        with open(args.data.generate_path + f"_new_embeddings_supcon_freeze_img_test_{args.ckpt}_generated.pkl", 'wb') as f:
            logger.log_message("Saving generated embeddings... Size: {}".format(img_generated_emb.shape))
            pkl.dump((img_generated_emb, f_name), f)
            
            
    if args.mode == "test":
        test_set = MoleculeImageDataset(args.data.valid_path, args=args, sample_strategy='retreival', \
                                        from_pretrained=pretrained_mol_enc)
        test_dataloader = build_loaders(test_set, batch_size=64, mode="train", drop_last=False)
        model, _ = load_state(model, optimizer, args)
        model.eval()
        test_step = 200
        smis = generate_retreival(args, model, test_dataloader, logger, step=test_step)
        smi_stats = {}
        for k in args.train.top_k:
            smi_stats[k] = dict(zip(test_set.unique_smiles, np.zeros(len(test_set.unique_smiles))))
        for smi in smis:
            for k in args.train.top_k:
                correct_smi = smi[f"o_top{k}_mol"]
                for s in correct_smi:
                    smi_stats[k][s] += 1
        with open(args.data.valid_path + "/test_retrieval_score_unfreeze_200steps.pkl", "wb") as f:
            pkl.dump(smi_stats, f)
        
        
if __name__ == "__main__":
    main()