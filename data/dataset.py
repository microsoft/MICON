import random
import os
import cv2
import h5py
import torch
import numpy as np
import pandas as pd
import albumentations as A
import pickle as pkl

from tqdm import tqdm
from argparse import Namespace
from typing import Callable, List, Union
from collections import defaultdict
from functools import lru_cache

from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from unimol_tools import UniMolRepr

POS_CONTROL = set(['c1ccc(-c2nn3c(c2-c2ccnc4cc(OCCN5CCOCC5)ccc24)CCC3)nc1', \
                  'COc1ncc2cc(C(=O)Nc3cc(C(O)=NCc4cccc(Cl)c4)ccc3Cl)c(O)nc2n1', \
                  'CC1CC2C3CC=C4CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO', \
                  'C=CC1CN2CCC1CC2C(O)c1ccnc2ccc(OC)cc12', \
                  'CCOC(=O)C1OC1C(O)=NC(CC(C)C)C(O)=NCCC(C)C', \
                  'Cc1csc(-c2nnc(Nc3ccc(Oc4ncccc4-c4cc[nH]c(=N)n4)cc3)c3ccccc23)c1', \
                  'O=C(c1ccccc1)N1CCC(CCCCN=C(O)C=Cc2cccnc2)CC1', \
                  'CC(C)N=C(O)N1CCC(N=C2Nc3cc(F)ccc3N(CC(F)F)c3ccc(Cl)cc32)C1'])

CONTROL = 'CS(C)=O'

def to_hdf5(data: np.ndarray, name: str):
    with h5py.File(name, "w") as f:
        f.create_dataset("raw_image", data=data)
    return

def load_hdf5(name: str) -> np.ndarray:
    with h5py.File(name, "r") as f:
        data = f["raw_image"][:]/255.
    return data

def get_all_data(root: os.PathLike, metadata_fname: os.PathLike):
    
    def get_file_name(metadata, type):
        source = metadata["Metadata_Source"].astype(str).values.tolist()
        batch = metadata["Metadata_Batch"].astype(str).values.tolist()
        plate = metadata["Metadata_Plate"].astype(str).values.tolist()
        well = metadata["Metadata_Well"].astype(str).values.tolist()
        fov = metadata["FOV"].astype(str).values.tolist()
        fnames = [os.path.join(root, type, '$'.join(x) + '.h5') for x in zip(source, batch, plate, well, fov)]
        return fnames
    
    metadata = pd.read_csv(os.path.join(root, metadata_fname))

    metadata_treated = metadata.loc[metadata["Metadata_SMILES"] != CONTROL]
    metadata_control = metadata.loc[metadata["Metadata_SMILES"] == CONTROL]  
    img_treated = get_file_name(metadata_treated, type="treated")
    img_control = get_file_name(metadata_control, type="control")
    
    treated_molecules = metadata_treated["Metadata_SMILES"].tolist()
    
    mol2img = defaultdict(list)
    img2mol = dict()
    for m,t in zip(treated_molecules, img_treated):
        mol2img[m].append(t)
        img2mol[t] = m
    
    # Class 0 is reserved for control label
    mol2label = {s:i+1 for i,s in enumerate(sorted(list(set(treated_molecules))))}
    mol2label[CONTROL] = 0
    
    return img_treated, img_control, treated_molecules, mol2img, img2mol, mol2label
    

class MoleculeImageDataset(Dataset):
    """A MoleculeImageDataset contains a list of molecules, their corresponding treated image view, and control image."""

    def __init__(self, root: os.PathLike, args: Namespace = None, sample_strategy: str = 'molecule', \
        from_pretrained = False, mode='train', metadata=None):
        """
        Initializes the MoleculeDataset.
        """
        self.args = args
        self.sample_strategy = sample_strategy
        self.from_pretrained = from_pretrained
        self.mode = mode
        
        if self.mode == 'train':
            metadata_fname = "metadata_train.csv" if not metadata else metadata
        elif self.mode == 'test':
            metadata_fname = "metadata_test.csv" if not metadata else metadata
        else:
            raise ValueError("mode must be either train or test")
        
        self.img_treated, self.img_control, self.molecules, self.mol2img, self.img2mol, self.mol2label = get_all_data(root, metadata_fname)
        self.unique_smiles = list(self.mol2img.keys())
        
        # Unimol representation.
        if self.args.mol_encoder.model == "Unimol":
            if not os.path.exists(root + "/unimol_repr.pkl"):    
                self.unique_smiles_dict = dict(zip(self.unique_smiles, \
                                                            from_pretrained.get_repr(self.unique_smiles)['cls_repr']))
                with open(root + "/unimol_repr.pkl", "wb") as f:
                    pkl.dump(self.unique_smiles_dict, f)
            else:
                with open(root + "/unimol_repr.pkl", "rb") as f:
                    self.unique_smiles_dict = pkl.load(f)
                for m in self.unique_smiles:
                    if m not in self.unique_smiles_dict:
                        self.unique_smiles_dict[m] = from_pretrained.get_repr(m)['cls_repr'][0]
        elif self.args.mol_encoder.model == "MLP":
            self.unique_smiles_dict = dict()
            for m in self.unique_smiles:
                self.unique_smiles_dict[m] = np.array(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(m), radius=2, nBits=1024)).astype(np.float32)
            
        assert len(self.molecules) == len(self.img_treated)
        # assert len(self.img_treated) == len(self.img_control)
        
        # Generate meta data for images. Including plate, batch, and source to image.
        self.plate_to_treated, self.plate_to_control, self.batch_to_treated, self.batch_to_control, \
            self.source_to_treated, self.source_to_control = self.get_meta_to_image(self.img_treated, self.img_control)
           
        if self.sample_strategy == 'sequential':
            self.max_num_batch = min([len(v) for v in self.mol2img.values()]) # the max number of unique mol batches for the dataset
            self.sequential_data = [[self.mol2img[k][i] for k in sorted(self.mol2img.keys())] for i in range(self.max_num_batch)]

        if self.sample_strategy == 'img_only':
            self.img_treated = list(set(self.img_treated))
            self.img_control = list(set(self.img_control))
            self.total_images = self.img_treated + self.img_control
            self.total_images_label = [0] * len(self.img_treated) + [1] * len(self.img_control) # Used only for generating embedding during inference
            
        # If the mean and std of the image set is already calculated, load them.
        if os.path.exists(root + "/mean.pkl"):
            with open(root + "/mean.pkl", "rb") as f:
                self.img_mean = pkl.load(f)
            with open(root + "/std.pkl", "rb") as f:
                self.img_std = pkl.load(f) 
        else:   
            self.get_image_norms(self.img_treated, self.img_control)
            with open(root + "/mean.pkl", "wb") as f:
                pkl.dump(self.img_mean, f)
            with open(root + "/std.pkl", "wb") as f:
                pkl.dump(self.img_std, f)    

        self.transforms = A.Compose(
        [
            A.Resize(224, 224, always_apply=True),
            A.Normalize(mean=self.img_mean, std=self.img_std, always_apply=True),
        ]
        ) 
        if self.mode == 'train':
            self.transforms = A.Compose(
                [
                    # A.RandomResizedCrop(224, 224, scale=(0.5, 1.)),
                    A.Resize(224, 224, always_apply=True),
                    # A.HorizontalFlip(),
                    # A.VerticalFlip(),
                    A.Normalize(mean=self.img_mean, std=self.img_std, max_pixel_value=1.0, always_apply=True),
                ]
            )

    def prepare_sequential_batch(batch: dict, device) -> dict:
        """
        Collate function used for sequential sampling.
        """
        batch = {
            'i_control' : torch.vstack(batch['i_control']).to(device),
            'i_treated' : torch.vstack(batch['i_treated']).to(device),
            'm_treated' : [x[0] for x in batch['m_treated']]
        }
        return batch

    def get_treated_classes(self):
        """
        Get the number of total treated_classes
        """
        return len(self.mol2label)

    def shuffle(self, seed: int = None):
        """
        Shuffles the dataset.

        :param seed: Optional random seed.
        """
        if seed is not None:
            random.seed(self.args.seed)

        if self.sample_strategy == 'molecule' or self.sample_strategy == 'random':
            random.shuffle(self.unique_smiles)
            random.shuffle(self.img_control)
            random.shuffle(self.img_treated)
        else:
            random.shuffle(self.sequential_data)

    def get_image_norms(self, img_treated: List[str], img_control: List[str]):
        """
        Calculate the mean and std of the image set.
        """
        mean = 0.
        std = 0.
        for f_path in tqdm(img_treated + img_control):
            image = load_hdf5(f_path)
            import pdb
            pdb.set_trace()
            image = cv2.resize(image, (224, 224))
            mean += np.mean(image.astype(float), axis=(0,1))
            std += np.std(image.astype(float), axis=(0,1))

        mean /= len(img_treated + img_control)
        std /= len(img_treated + img_control)
        self.img_mean = tuple(mean)
        self.img_std = tuple(std)

    def get_meta_to_image(self, img_treated: List[str], img_control: List[str]):
        """
        Get a dictionary that maps plate to image.
        """
        plate_to_treated = defaultdict(list)
        plate_to_control = defaultdict(list)
        batch_to_treated = defaultdict(list)
        batch_to_control = defaultdict(list)
        source_to_treated = defaultdict(list)
        source_to_control = defaultdict(list)
        for img in img_treated:
            source, batch, plate = img.split('/')[-1].split('$')[:3]
            source_to_treated[source].append(img)
            batch_to_treated[batch].append(img)
            plate_to_treated[plate].append(img)
        for img in img_control:
            source, batch, plate = img.split('/')[-1].split('$')[:3]
            source_to_control[source].append(img)
            batch_to_control[batch].append(img)
            plate_to_control[plate].append(img)

        return plate_to_treated, plate_to_control, batch_to_treated, batch_to_control, source_to_treated, source_to_control

    def sample_control_with_meta(self, source, batch, plate):
        if plate in self.plate_to_control:
            control_img_name = random.sample(self.plate_to_control[plate], 1)[0]
        elif batch in self.batch_to_control:
            control_img_name = random.sample(self.batch_to_control[batch], 1)[0]
        elif source in self.source_to_control:
            control_img_name = random.sample(self.source_to_control[source], 1)[0]
        else:
            control_img_name = random.sample(self.img_control, 1)[0]

        return control_img_name
    
    def sample_treated_with_meta(self, source, batch, plate):
        if plate in self.plate_to_treated:
            treated_img_name = random.sample(self.plate_to_treated[plate], 1)[0]
            smi = self.img2mol[treated_img_name]
        elif batch in self.batch_to_treated:
            treated_img_name = random.sample(self.batch_to_treated[batch], 1)[0]
            smi = self.img2mol[treated_img_name]
        elif source in self.source_to_treated:
            treated_img_name = random.sample(self.source_to_treated[source], 1)[0]
            smi = self.img2mol[treated_img_name]
        else:
            treated_img_name = random.sample(self.img_treated, 1)[0]
            smi = self.img2mol[treated_img_name]

        return treated_img_name, smi

    def __len__(self) -> int:
        """
        Returns the length of the dataset (i.e. the number of molecules/treated images).

        :return: The length of the dataset.
        """
        if self.sample_strategy == 'random':
            return len(self.molecules)
        elif self.sample_strategy in ['molecule', 'retrieval']:
            return len(self.unique_smiles)
        elif self.sample_strategy == 'sequential':
            return len(self.sequential_data)
        elif self.sample_strategy == 'img_only':
            return len(self.total_images)

    def __getitem__(self, idx) -> dict:
        """
        Gets a batch of treated images and their corresponding molecule structures, and control images. 
        Selecting unique molecules for a single batch.
        """
        if self.sample_strategy == 'molecule' or self.sample_strategy == 'random': 
            if self.sample_strategy == 'molecule':
                smi = self.unique_smiles[idx]

            elif self.sample_strategy == 'random':
                smi = self.molecules[idx]
            treated_img_1_name, treated_img_2_name = random.sample(self.mol2img[smi], 2)
            source_1, batch_1, plate_1 = treated_img_1_name.split('/')[-1].split('$')[:3]
            source_2, batch_2, plate_2 = treated_img_2_name.split('/')[-1].split('$')[:3]
            treated_img_1 = load_hdf5(treated_img_1_name)
            treated_img_2 = load_hdf5(treated_img_2_name)
            labels = self.mol2label[smi]
            
            # Sample with batch
            control_img_1_name = self.sample_control_with_meta(source_1, batch_1, plate_1)
            control_img_2_name = self.sample_control_with_meta( source_2, batch_2, plate_2)

            control_img_1 = load_hdf5(control_img_1_name)
            control_img_2 = load_hdf5(control_img_2_name)

            treated_img_1 = self.transforms(image=treated_img_1)['image']
            treated_img_2 = self.transforms(image=treated_img_2)['image']
            control_img_1 = self.transforms(image=control_img_1)['image']
            control_img_2 = self.transforms(image=control_img_2)['image']

            if self.from_pretrained:
                smi = torch.tensor(self.unique_smiles_dict[smi])
            batch = {
                'i_control_1' : torch.tensor(control_img_1).permute(2, 0, 1).float(),
                'i_control_2' : torch.tensor(control_img_2).permute(2, 0, 1).float(),
                'i_treated_1' : torch.tensor(treated_img_1).permute(2, 0, 1).float(),
                'i_treated_2' : torch.tensor(treated_img_2).permute(2, 0, 1).float(),
                'm_treated' : smi,
                'm_label': torch.tensor([labels])
            }
        # elif self.sample_strategy == 'random':
        #     smi = self.unique_smiles[idx]
        #     treated_img, _ = random.sample(self.mol2img[smi], 1)[0]
        #     control_img = random.sample(self.img_control, 1)[0]
        #     treated_img = self.transforms(image=load_hdf5(treated_img))['image']
        #     control_img = self.transforms(image=load_hdf5(control_img))['image']
        #     batch = {
        #         'i_control' : torch.tensor(control_img).permute(2, 0, 1).float(),
        #         'i_treated' : torch.tensor(treated_img).permute(2, 0, 1).float(),
        #         'm_treated' : smi
        #     }
        elif self.sample_strategy == 'sequential':
            batch = {
                'i_control' : [torch.tensor(self.transforms(image=load_hdf5(x[1]))['image']).permute(2, 0, 1).float() for x in self.sequential_data[idx]],
                'i_treated' : [torch.tensor(self.transforms(image=load_hdf5(x[0]))['image']).permute(2, 0, 1).float() for x in self.sequential_data[idx]],
                'm_treated' : sorted(list(self.mol2img.keys()))
            }

        elif self.sample_strategy == 'img_only':
            img_name = self.total_images[idx]
            label = self.total_images_label[idx]
            source, batch, plate = img_name.split('/')[-1].split('$')[:3]
            if label == 0:
                # original image is control, we sample a treated image for generation (Deprecated)
                img_for_generation_name, smi = self.sample_treated_with_meta(source, batch, plate)
            else:
                # original image is treated, we sample an control image for generation
                img_for_generation_name = self.sample_control_with_meta(source, batch, plate)
                    
            img_for_generation  = self.transforms(image=load_hdf5(img_for_generation_name))['image']
            img = self.transforms(image=load_hdf5(img_name))['image']
            if self.from_pretrained:
                smi = torch.tensor(self.unique_smiles_dict[smi])
            batch = {
                'label' : torch.tensor([label]),
                'img' : torch.tensor(img).permute(2, 0, 1).float(),
                'img_for_generation' : torch.tensor(img_for_generation).permute(2, 0, 1).float(),
                'm_treated' : smi,
                'f_name' : self.total_images[idx].split('/')[-1].split('.')[:-1]
            }
        elif self.sample_strategy == 'retreival':
            smi = self.unique_smiles[idx]
            treated_1, treated_2 = random.sample(self.mol2img[smi], 2)
            treated_1 = self.transforms(image=load_hdf5(treated_1))['image']
            treated_2 = self.transforms(image=load_hdf5(treated_2))['image']
            batch = {
                'i_treated_1' : torch.tensor(treated_1).permute(2, 0, 1).float(),
                'i_treated_2' : torch.tensor(treated_2).permute(2, 0, 1).float(),
                'm_treated': smi
            }
        return batch
