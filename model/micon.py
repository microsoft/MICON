import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from functools import lru_cache

from .loss import SupConLoss


class ProjectionHead(nn.Module):
    def __init__(
        self,
        hidden_size,
        projection_dim,
        dropout,
    ):
        super().__init__()
        self.projection = nn.Linear(hidden_size, projection_dim)
        self.bn = nn.BatchNorm1d(projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.bn(projected)
        x = self.gelu(x)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        return x
    
class GenerationHead(nn.Module):
    def __init__(
        self,
        hidden_size,
        projection_dim
    ):
        super().__init__()
        self.projection = nn.Linear(hidden_size, projection_dim)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        
    def forward(self, x):
        x = self.projection(x)
        x = self.relu(x)
        x = self.fc(x)
        return x

class LinearProjectionHead(nn.Module):
    def __init__(
        self,
        hidden_size,
        projection_dim
    ):
        super().__init__()
        self.projection = nn.Linear(hidden_size, projection_dim)
    
    def forward(self, x):
        projected = F.normalize(self.projection(x))
        return projected
    
class MICON(nn.Module):
    def __init__(self, args, img_enc, mol_enc=None):
        super(MICON, self).__init__()
        self.img_enc = img_enc
        self.mol_enc = mol_enc
        self.args = args
        self.pre_mol_projector = LinearProjectionHead(512, args.micon.hidden_size)
        self.original_img_projector = ProjectionHead(args.micon.hidden_size, args.micon.projection_dim, args.micon.dropout)
        if args.micon.generation_head:
            self.task_embedding_size = 64
            self.generation_task_embedding = nn.Embedding(2, self.task_embedding_size)
            self.generation_head_image = nn.Linear(args.micon.hidden_size, args.micon.hidden_size)
            self.generation_head_perturbation = nn.Linear(args.micon.hidden_size +  self.task_embedding_size, args.micon.hidden_size)
        self.temperature = args.micon.temperature
        self.supcon_criterion = SupConLoss(temperature=self.temperature)
        self.device = args.device
        self.gen_lambda = 1.0
        

    def forward(self, i_control_1, i_control_2, i_treated_1, i_treated_2, m_treated, m_label):
        # Getting all the hidden representation
        i_control_1 = self.interleave_channels(i_control_1)
        i_control_2 = self.interleave_channels(i_control_2)
        i_treated_1 = self.interleave_channels(i_treated_1)
        i_treated_2 = self.interleave_channels(i_treated_2)
        m_treated = self.pre_mol_projector(m_treated)
    
        
        # if not self.args.micon.self_supervised:
        # # Projection
        #     i_original= F.normalize(self.original_img_projector(torch.cat((i_treated, i_control), 0)))
        #     i_generated =  F.normalize(self.original_img_projector(torch.cat((i_treated_gen, i_control_gen), 0)))

        #     # Loss
        #     logits = (i_original @ i_generated.T) / self.temperature
        #     num_of_img = i_treated.shape[0]
        #     label_mask = torch.cat((torch.cat((torch.eye(num_of_img), torch.zeros(num_of_img,num_of_img)), dim=-1), \
        #                             torch.cat((torch.zeros(num_of_img, num_of_img), torch.ones(num_of_img, num_of_img)/num_of_img), dim=-1)), \
        #                             dim=0).to(i_original.device)

        #     # targets = F.softmax(i_generated @ i_generated.T * label_mask / self.temperature, dim=-1)
        #     # targets_t = F.softmax(i_original @ i_original.T * label_mask / self.temperature, dim=0).T
        #     loss_original = cross_entropy(logits, label_mask, reduction='none')
        #     loss_generated = cross_entropy(logits.T, label_mask, reduction='none')
        #     loss = (loss_original + loss_generated) / 2
        #     return logits, label_mask, loss_original.mean(), loss_generated.mean(), loss.mean()
        if self.args.micon.loss == 'supervised':
            # labels = torch.vstack((m_label, torch.zeros_like(m_label).to(self.device))).repeat(2,1)
            # view_1 = F.normalize(self.original_img_projector(torch.vstack((i_treated_1, i_control_1, i_treated_gen_1, i_control_gen_1))))
            # view_2 = F.normalize(self.original_img_projector(torch.vstack((i_treated_2, i_control_2, i_treated_gen_2, i_control_gen_2))))
            
            # loss, logits = self.supcon_criterion(torch.stack((view_1, view_2), dim=1), labels)
            
            # Original vs Original
            labels = torch.vstack((m_label, torch.zeros_like(m_label).to(self.device)))
            view_1_orig = F.normalize(self.original_img_projector(torch.vstack((i_treated_1, i_control_1))))
            view_2_orig = F.normalize(self.original_img_projector(torch.vstack((i_treated_2, i_control_2))))
            loss, logits = self.supcon_criterion(torch.stack((view_1_orig, view_2_orig), dim=1), labels)

            if self.args.micon.generation_head:
                # Original vs Generated
                i_treated_1_for_generation = i_treated_1.detach().clone()
                i_treated_2_for_generation = i_treated_2.detach().clone()
                i_control_1_for_generation = i_control_1.detach().clone()
                i_control_2_for_generation = i_control_2.detach().clone()
                _bsz_treated = i_treated_1.size(0)
                _bsz_control = i_control_1.size(0)
                generation_treated_task_embedding = self.generation_task_embedding(torch.zeros(_bsz_control, dtype=torch.int64).to(self.device))
                generation_control_task_embedding = self.generation_task_embedding(torch.ones(_bsz_treated,dtype=torch.int64).to(self.device))
                generation_treated_perturbation = self.generation_head_perturbation(torch.cat((generation_treated_task_embedding, m_treated), dim=-1))
                generation_control_perturbation = self.generation_head_perturbation(torch.cat((generation_control_task_embedding, m_treated), dim=-1))
                
                i_control_gen_1 = torch.add(self.generation_head_image(i_treated_1_for_generation), generation_control_perturbation)
                i_control_gen_2 = torch.add(self.generation_head_image(i_treated_2_for_generation), generation_control_perturbation)
                i_treated_gen_1 = torch.add(self.generation_head_image(i_control_1_for_generation), generation_treated_perturbation)
                i_treated_gen_2 = torch.add(self.generation_head_image(i_control_2_for_generation), generation_treated_perturbation)
                
                view_1_gen = F.normalize(self.original_img_projector(torch.vstack((i_treated_1_for_generation, i_control_1_for_generation))))
                view_2_gen = F.normalize(self.original_img_projector(torch.vstack((i_treated_gen_1, i_control_gen_1))))
                loss_gen_1, logits_gen_1 = self.supcon_criterion(torch.stack((view_1_gen, view_2_gen), dim=1), labels)
            
                if not self.args.micon.generation_double_align:
                    return logits, loss.mean() + self.gen_lambda * loss_gen_1.mean()
                else:
                    view_3_gen = F.normalize(self.original_img_projector(torch.vstack((i_treated_2_for_generation, i_control_2_for_generation))))
                    view_4_gen = F.normalize(self.original_img_projector(torch.vstack((i_treated_gen_2, i_control_gen_2))))
                    loss_gen_2, logits_gen_2 = self.supcon_criterion(torch.stack((view_3_gen, view_4_gen), dim=1), labels)
                    return logits, loss.mean() + self.gen_lambda * 0.5 * (loss_gen_1.mean() + loss_gen_2.mean())
                                      
        elif self.args.micon.loss == 'self-supervised':
            i_original_generated_1 = F.normalize(self.original_img_projector(torch.cat((i_treated_1, i_control_1, i_treated_gen_1, i_control_gen_1), 0)))
            i_original_generated_2 = F.normalize(self.original_img_projector(torch.cat((i_treated_2, i_control_2, i_treated_gen_2, i_control_gen_2), 0)))
            logits = (i_original_generated_1 @ i_original_generated_2.T) / self.temperature
            
            num_of_img = i_treated_1.shape[0]
            quadrant_mask = torch.cat((torch.cat((torch.eye(num_of_img), torch.zeros(num_of_img, num_of_img)), dim=-1), \
                                    torch.cat((torch.zeros(num_of_img, num_of_img), torch.ones(num_of_img, num_of_img)/(num_of_img)), dim=-1)), \
                                    dim=0).to(self.device)

            label_mask = quadrant_mask.repeat(1,2)
            loss_original_1 = cross_entropy(logits[:num_of_img*2, :], label_mask, reduction='none')
            loss_generated_1 = cross_entropy(logits[num_of_img*2:, :], label_mask, reduction='none')
            loss_original_2 = cross_entropy(logits[:, :num_of_img*2].T, label_mask, reduction='none')
            loss_generated_2 = cross_entropy(logits[:, num_of_img*2:].T, label_mask, reduction='none')
            loss = (loss_original_1 + loss_generated_1 + loss_original_2 + loss_generated_2) / 2

        return logits, loss.mean()

    def _step(self, batch, step_name):
        logits, loss = self.forward(**batch)
        logits = logits.detach().cpu()
        if self.args.micon.loss == 'supervised':
            n = int(logits.shape[0]/8)
            original_logits = torch.cat((logits[:n,:4*n], logits[n*2:n*3,:4*n]), dim=0)
            generated_logits = torch.cat((logits[4*n:5*n,4*n:], logits[6*n:7*n:,4*n:]), dim=0)
            labels = torch.cat((torch.arange(2*n, 3*n).long(), torch.arange(n).long()))
   
        elif self.args.micon.loss == 'self-supervised':
            n = int(logits.shape[0]/4)
            original_logits = logits[:n, 2*n:]
            generated_logits = logits[2*n:3*n, :2*n]
            labels = torch.arange(n).long()
        else:
            # Remove ori_vs_ori and gen_vs_gen diagonal from logits (always 1)
            n = int(logits.shape[0]/4)
            ori_vs_ori = logits[:n,:n].flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)
            ori_vs_gen = logits[:n,n*2:n*3]
            gen_vs_ori = logits[n*2:n*3, :n]
            gen_vs_gen = logits[n*2:n*3, n*2:n*3].flatten()[1:].view(n-1, n+1)[:,:-1].reshape(n, n-1)
        
            original_logits =  torch.cat((ori_vs_gen, ori_vs_ori), dim=-1)
            
            generated_logits = torch.cat((gen_vs_ori, gen_vs_gen), dim=-1)
        
        logs = {}
        topk = self.args.train.top_k
        acc_a, _ = accuracy(original_logits, labels, topk=topk)
        acc_b, _ = accuracy(generated_logits, labels, topk=topk)
        for k, acc_ak, acc_bk in zip(topk, acc_a, acc_b):
            suffix = f"_top{k}"
            acc_ak = acc_ak.cpu().item()
            acc_bk = acc_bk.cpu().item()
            logs.update(
                {
                    f"o{suffix}" : acc_ak*100,
                    f"g{suffix}" : acc_bk*100,
                    # f"{step_name}_acc_all{suffix}": (acc_ak + acc_bk) / 2,
                }
            )
        batch_dictionary = {
            "loss": loss,
            "log": logs,
        }
            
        return batch_dictionary

    def get_retrieval_scores(self, i_treated_1, i_treated_2, m_treated):
        i_treated_1 = self.interleave_channels(i_treated_1)
        i_treated_2 = self.interleave_channels(i_treated_2)
        i_treated_1 = F.normalize(self.original_img_projector(i_treated_1))
        i_treated_2 = F.normalize(self.original_img_projector(i_treated_2))        
        logits = (i_treated_1 @ i_treated_2.T) / self.temperature
        logits = logits.detach().cpu()
        labels = torch.arange(int(logits.shape[0])).long()
        logs = {}
        smis = {}
        topk = self.args.train.top_k
        acc_all, acc_ind_all = accuracy(logits, labels, topk=topk)
        for k, acc, acc_ind, in zip(topk, acc_all, acc_ind_all):
            suffix = f"_top{k}"
            acc = acc.cpu().item()
            logs.update(
                {
                    f"o{suffix}" : acc*100,
                    # f"{step_name}_acc_all{suffix}": (acc_ak + acc_bk) / 2,
                }
            )
            smis.update(
                {
                    f"o{suffix}_mol" : np.take(m_treated, acc_ind)
                }
            )
        batch_dictionary = {
            "log": logs,
            "smis": smis
        }
            
        return batch_dictionary
        
        
    def generate_image_embeddings(self, label, img, img_for_generation, m_treated, f_name, type="projected"):
        img_emb  = self.interleave_channels(img)
        # If label is 1, generate treated image, else label is -1, generate control image
        if self.args.micon.generation_head:
            m_treated = self.pre_mol_projector(m_treated)
            img_for_generation = self.interleave_channels(img_for_generation)
            img_generated_emb = self.generation_head(torch.cat((img_for_generation, m_treated, label), dim=-1))
        else:
            img_generated_emb = self.interleave_channels(img_for_generation) + torch.mul(label, self.pre_mol_projector(m_treated))

        if type == "projected":
            img_emb = F.normalize(self.original_img_projector(img_emb))
            img_generated_emb = F.normalize(self.original_img_projector(img_generated_emb))
        elif type == "raw":
            img_emb = F.normalize(img_emb)
            img_generated_emb = F.normalize(img_generated_emb)
        
        return img_emb, img_generated_emb, f_name
        
    def training_step(self, train_batch):
        batch_dictionary = self._step(
            batch=train_batch, step_name="train"
        )

        return batch_dictionary

    def validation_step(self, val_batch):
        batch_dictionary = self._step(
            batch=val_batch, step_name="val"
        )
        
        return batch_dictionary
    
    def interleave_channels(self, img):
        '''
            Whether to interleave the channels of 5-channel images to 5 3-channel images to fit the input of ResNet.
            
        '''
        if self.args.micon.interleave_image == True:
            batch_size = img.shape[0]
            img = self.img_enc(img.repeat_interleave(3, dim=1).view(-1, 3, 224, 224)).view(batch_size, -1)
        else:
            img = self.img_enc(img)
        
        return F.normalize(img)

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk=(1,)
) -> List[torch.FloatTensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.
    
    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        maxk = max(
            topk
        )  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = (
            y_pred.t()
        )  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.
        
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(
            y_pred
        )  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (
            y_pred == target_reshaped
        )  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        list_topk_accs = []
        list_topk_accs_index = []
        for k in topk:
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            flattened_indicator_which_topk_matched_truth = (
                ind_which_topk_matched_truth.float()
            ) 
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(
                dim=0, keepdim=False
            )
            topk_accs_index = torch.nonzero(tot_correct_topk).flatten().numpy()
            list_topk_accs_index.append(topk_accs_index)
            topk_acc = tot_correct_topk.sum() / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs, list_topk_accs_index   # list of topk accuracies for entire batch [topk1, topk2, ... etc]