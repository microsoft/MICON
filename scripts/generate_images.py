import pandas as pd
import numpy as np
import pickle as pkl
import random
import h5py
import sys
import os
import pickle as pkl
from PIL import Image
from subprocess import call
from collections import Counter
import tqdm
import time 

def to_hdf5(data, name):
    with h5py.File(name, "w") as f:
        f.create_dataset("raw_image", data=data)
    return

def load_hdf5(name):
    with h5py.File(name, "r") as f:
        data = f["raw_image"][:]
    return data

def to_numpy(data, name):
    with open(name, "wb") as f:
        np.save(f, data)
    return

def load_numpy(name):
    with open(name, "rb") as f:
        data = np.load(f)
    return data

def pil_load(file):
    with Image.open(file) as im:
            arr = np.array(im)
    return arr

if __name__ == "__main__":
    np.random.seed(42)
    random.seed(42)
    pos_control = ["JCP2022_085227", "JCP2022_037716", "JCP2022_025848", 
                   "JCP2022_046054", "JCP2022_035095", "JCP2022_064022", 
                   "JCP2022_050797", "JCP2022_012818"]
    
    metadata = pd.read_csv("/data/yemin/MICON-main/datasets/metadata/local_image_paths.csv")
    compound = pd.read_csv("/data/yemin/MICON-main/datasets/metadata/compound.csv")
    plates = pd.read_csv("/data/yemin/MICON-main/datasets/metadata/plate.csv.gz")
    
    target2_plates = plates.loc[plates["Metadata_PlateType"] == "TARGET1"]
    target2_compound = metadata.merge(target2_plates, on=['Metadata_Source', 'Metadata_Batch', 'Metadata_Plate'])
    target2_compound_smi = target2_compound.merge(compound, on=['Metadata_JCP2022'])
    target2_compound_smi = target2_compound_smi.drop_duplicates()
    
    jp_target = pd.read_csv("/data/yemin/MICON-main/datasets/JUMP-Target-2_compound_metadata_with_cp0016_inchikey.csv")
    inchikey2moa = dict(zip(jp_target['Metadata_InChIKey'], jp_target['target']))
    
    non_control_target2 = target2_compound_smi.loc[(target2_compound_smi["Metadata_InChIKey"].isin(inchikey2moa.keys()))] 
                                                #    & (target2_compound_smi["Metadata_InChIKey"] != "IAZDPXIOMUYVGZ-UHFFFAOYSA-N")]
    
    non_control_target2["Metadata_Target"] = [inchikey2moa[x] for x in non_control_target2['Metadata_InChIKey'].tolist()]
    
    # compound_plates = plates.loc[plates["Metadata_PlateType"] == "COMPOUND"]
    # metadata_compound = metadata.merge(compound_plates, on=['Metadata_Source', 'Metadata_Batch', 'Metadata_Plate'])
    # metadata_compound_smi = metadata_compound.merge(compound, on=['Metadata_JCP2022']).drop_duplicates()
    #with open("/home/t-yeminyu/target2.pkl","rb") as f:
        #moa_mols = pkl.load(f)
    #metadata_compound_smi = metadata_compound_smi.loc[[Met"Metadata_SMILES"]
    # metadata_compound_smi = metadata_comound_smi.loc[metadata_comound_smi['FOV'] == 5]
    
    # metadata_compound_smi = pd.read_csv("/home/t-yeminyu/target2_test_meta.csv")
    metadata_compound_smi = non_control_target2
    metadata_compound_smi.rename(columns={"FOV": "Metadata_Fov"}, inplace=True)
    smi_counter = Counter(metadata_compound_smi['Metadata_SMILES'].tolist())
    smi_list = sorted([(k,v) for k,v in smi_counter.items()], key=lambda x: x[1], reverse=True)
    # skip pos and neg control
    root = "/data/yemin/cell_painting_hot/"
    meta_df = pd.DataFrame()
    
    # for i, (smi, _) in tqdm.tqdm(enumerate(smi_list)):
    # selected_plates = metadata_compound_smi['Metadata_Plate'].sample(20)
    processed_image = os.listdir("/data/yemin/MICON-main/datasets/target1/treated/") + os.listdir("/data/yemin/MICON-main/datasets/target1/control/")
    print(f"Processing image {len(processed_image)}.")
    selected_plates = []
    for i in metadata_compound_smi['Metadata_Plate']:
        if i not in selected_plates:
            selected_plates.append(i)
    print(len(selected_plates))
    num_processed_image = 0
    total_image = len(metadata_compound_smi)
    start_time = time.time()
    for plate in selected_plates:
        _df = metadata_compound_smi.loc[metadata_compound_smi['Metadata_Plate'] == plate]
        bad_index = []
        for i in tqdm.tqdm(range(len(_df))):
            temp_df = _df.iloc[i]
            num_processed_image += 1
            treated_fname =  temp_df["Metadata_Source"] + "$" +temp_df["Metadata_Batch"] + "$" + temp_df["Metadata_Plate"] + "$" + temp_df["Metadata_Well"] + "$" + str(temp_df["Metadata_Fov"]) + ".h5"
            print(f"Processing well {num_processed_image}/{total_image}, total time {time.time() - start_time}s, Avg time per well {(time.time() - start_time)/num_processed_image}s.")
            if treated_fname in processed_image:
                print(f"Already processed image")
                continue
            if temp_df["Metadata_InChIKey"] != "IAZDPXIOMUYVGZ-UHFFFAOYSA-N":
                is_treated = True
            else:
                is_treated = False
            _treated = []
            for index in ["agp_path", "dna_path", "er_path", "mito_path", "rna_path"]:
                 _treated.append(pil_load(root + temp_df[index]))
            if len(_treated) == 5:
                _treated = np.stack(_treated, axis=2)
                if is_treated:
                    to_hdf5(_treated, "/data/yemin/MICON-main/datasets/target1/treated/"+ treated_fname)
                else:
                    to_hdf5(_treated, "/data/yemin/MICON-main/datasets/target1/control/"+ treated_fname)
            else:
                bad_index.append(i)
                continue
        _df.drop(_df.index[bad_index], inplace=True)
        meta_df = pd.concat((meta_df, _df), axis=0, ignore_index=True)
        meta_df.to_csv("/data/yemin/MICON-main/datasets/target1/metadata.csv", index=False)
        #with open("/home/t-yeminyu/code/MICON/datasets/treated_xlarge/treated_file_path.pkl", "wb") as f:
            #pkl.dump(treated_file_path, f)
            
#     neg_control_imgs = metadata.loc[(metadata['Metadata_JCP2022'] == "JCP2022_033924") & (metadata["FOV"] == 5)]
#     pos_control_imgs = metadata.loc[metadata['Metadata_JCP2022'].isin(pos_control) & (metadata["FOV"] == 5)].sample(frac=1).reset_index(drop=True)

#     neg_df = pd.DataFrame()
#     for i in range(len(pos_control_imgs)):
#         data = pos_control_imgs.iloc[i]
#         plate = data["Metadata_Plate"]
#     for i in range(len(pos_control_imgs)):
#         data = pos_control_imgs.iloc[i]
#         plate = data["Metadata_Plate"]
#         batch = data["Metadata_Batch"]
#         source = data["Metadata_Source"]
#         neg = neg_control_imgs.loc[neg_control_imgs["Metadata_Plate"] == plate]
#         if len(neg) == 0:
#             neg = neg_control_imgs.loc[neg_control_imgs["Metadata_Batch"] == batch]
#         if len(neg) == 0:
#             neg = neg_control_imgs.loc[neg_control_imgs["Metadata_Source"] == source]
#         _neg_df = neg.sample(1)
#         neg_df = pd.concat([neg_df, _neg_df], ignore_index=True)
    
#     root = "cell-painting-hot/"
#     ctl_file_path = []
#     treated_file_path = []

#     for i in tqdm.tqdm(range(len(pos_control_imgs))):
#         _control = []
#         _treated = []
#         for index in ["agp_path", "dna_path", "er_path", "mito_path", "rna_path"]:
#             # call(["cp", root + control.iloc[i][index], "resnet_test/control/"])
#             # call(["cp", root + treated.iloc[i][index], "resnet_test/treated/"])
#             _control.append(pil_load(root + neg_df.iloc[i][index]))
#             _treated.append(pil_load(root + pos_control_imgs.iloc[i][index]))
#         _control = np.stack(_control, axis=2)
#         _treated = np.stack(_treated, axis=2)
#         control_fname = neg_df.iloc[i]["Metadata_Plate"] + "+" + \
#                 neg_df.iloc[i]["Metadata_Well"] + "_" + str(neg_df.iloc[i]["FOV"]) + ".h5"
#         treated_fname = pos_control_imgs.iloc[i]["Metadata_Plate"] + "+" + pos_control_imgs.iloc[i]["Metadata_Well"] + "_" \
#                 + str(pos_control_imgs.iloc[i]["FOV"]) + ".h5"
                
#         ctl_file_path.append(control_fname)
#         treated_file_path.append(treated_fname)
#         to_hdf5(_control, "/home/t-yeminyu/MICON/datasets/pos_control/control/" + control_fname)
#         to_hdf5(_treated, "/home/t-yeminyu/MICON/datasets/pos_control/treated/"+ treated_fname)

#         if i % 100 == 0:
#             with open("/home/t-yeminyu/MICON/datasets/pos_control/control_files", "wb") as f:
#                 pkl.dump(ctl_file_path, f)
#             with open("/home/t-yeminyu/MICON/datasets/pos_control/treated_files", "wb") as f:
#                 pkl.dump(treated_file_path, f)
            
#         with open("/home/t-yeminyu/MICON/datasets/pos_control/control_files", "wb") as f:
#             pkl.dump(ctl_file_path, f)
#         with open("/home/t-yeminyu/MICON/datasets/pos_control/treated_files", "wb") as f:
#             pkl.dump(treated_file_path, f)
            
    # neg_df["data_file_path"] = ctl_file_path
    # pos_control_imgs["data_file_path"] = treated_file_path
    
    # neg_df.to_csv("/home/t-yeminyu/MICON/datasets/pos_control/control_metadata.csv", index=False)
