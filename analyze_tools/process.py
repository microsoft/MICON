import os
import io
import random
import pandas as pd
import numpy as np
import pickle as pkl
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import colorcet as cc
import seaborn as sns
import numba
import umap
import time
import pdb

import scib
import anndata as ad

from pycytominer import normalize
from pycytominer.operations.transform import Spherize
from pycytominer.operations.transform import RobustMAD

import warnings
# warnings.filterwarnings("ignore", category=DeprecationWarning) 

SMI2LABEL = {'c1ccc(-c2nn3c(c2-c2ccnc4cc(OCCN5CCOCC5)ccc24)CCC3)nc1': 1,
 'COc1ncc2cc(C(=O)Nc3cc(C(O)=NCc4cccc(Cl)c4)ccc3Cl)c(O)nc2n1': 2,
 'CC1CC2C3CC=C4CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO': 3,
 'C=CC1CN2CCC1CC2C(O)c1ccnc2ccc(OC)cc12': 4,
 'CCOC(=O)C1OC1C(O)=NC(CC(C)C)C(O)=NCCC(C)C': 5,
 'Cc1csc(-c2nnc(Nc3ccc(Oc4ncccc4-c4cc[nH]c(=N)n4)cc3)c3ccccc23)c1': 6,
 'O=C(c1ccccc1)N1CCC(CCCCN=C(O)C=Cc2cccnc2)CC1': 7,
 'CC(C)N=C(O)N1CCC(N=C2Nc3cc(F)ccc3N(CC(F)F)c3ccc(Cl)cc32)C1': 8,
 'CS(C)=O': 'control'}
SOURCE_LIST = ['source_2',  'source_3', 'source_5', 'source_6', 'source_7', 'source_8', 'source_11']

# Get MOA dict (inchikey to moa target)

JP_TARGET = pd.read_csv("datasets/JUMP-Target-2_compound_metadata_with_cp0016_inchikey.csv")
INCHIKEY2MOA = dict(zip(JP_TARGET['Metadata_InChIKey'], JP_TARGET['target']))

# For dropping features with abnormal standard deviation
def drop_bad_columns(df):
    cols = [c for c in df.columns if "Metadata_" not in c]
    stdev = [df[c].std() for c in cols]

    cols_to_drop = []
    cols_to_drop.extend([cols[i] for i, s in enumerate(stdev) if s < 0.1 or s > 5])
    cols_to_drop.extend([c for c in cols if "Nuclei_Correlation_RWC" in c])
    cols_to_drop.extend([c for c in cols if "Nuclei_Correlation_Manders" in c])
    cols_to_drop.extend([c for c in cols if "Nuclei_Granularity_14" in c])
    cols_to_drop.extend([c for c in cols if "Nuclei_Granularity_15" in c])
    cols_to_drop.extend([c for c in cols if "Nuclei_Granularity_16" in c])

    df = df[[c for c in df.columns if c not in cols_to_drop]]
    return df, cols_to_drop

# Postprocessing

def robustMAD(df, feature_cols, control_only=False):
    if not control_only:
        feature_df = df.loc[:, feature_cols]
    else:
        feature_df = df.query('`Metadata_SMILES` == "CS(C)=O"').loc[:, feature_cols]
        if len(feature_df) == 0:
            print("No control samples found. Fall back to full normailization")
            feature_df = df.loc[:, feature_cols]         
    rmad = RobustMAD()
    rmad.fit(feature_df)
    return rmad.transform(df.loc[:, feature_cols])
    
def spherize(df, feature_cols, control_only=False):
    if not control_only:
        feature_df = df.loc[:, feature_cols]
    else:
        feature_df = df.query('`Metadata_SMILES` == "CS(C)=O"').loc[:, feature_cols]
        if len(feature_df) == 0:
            print("No control samples found. Fall back to full normailization")
            feature_df = df.loc[:, feature_cols]         
    spherize = Spherize()
    spherize.fit(feature_df)
    return spherize.transform(df.loc[:, feature_cols])

def normalizer(df):
    normalized_df = normalize(
    profiles=df,
    meta_features="infer",
    method="mad_robustize"
)
    return normalized_df

# plate-wise process

def plate_wise_spherize_and_normailize(df, plate_col="Metadata_Plate", feature_cols=None, is_spherize=True, is_normalize=True, control_only=False):
    _df = df.copy(deep=True)
    if not feature_cols:
        feature_cols = [c for c in df.columns if not c.startswith('Metadata_')]
    for plate in tqdm(_df[plate_col].unique()):
        plate_df = _df.loc[_df[plate_col] == plate].copy(deep=True)
        if is_normalize:
            plate_df.loc[:, feature_cols] = robustMAD(plate_df, feature_cols=feature_cols, control_only=control_only)
        _df.loc[_df[plate_col] == plate] = plate_df.copy(deep=True)
    if is_spherize:
        _df.loc[:, feature_cols] = spherize(_df, feature_cols=feature_cols, control_only=control_only)   
    return _df

# Combine-fov to well profiles: ONLY used in micon features

def average_wells(df, feature_cols=None):
    if not feature_cols:
        feature_cols = [c for c in df.columns if not c.startswith('Metadata_')]
    else:
        feature_cols = [c for c in df.columns if c.startswith(feature_cols)]
    index_col = ['Metadata_Plate', 'Metadata_Well']
    index_keys = df[index_col].drop_duplicates()
    print(len(index_keys))
    non_feature_cols = [c for c in df.columns if not c in feature_cols and c != 'Metadata_Fov']
    full_cols = non_feature_cols + feature_cols
    res_values = []
    for _, _df in tqdm(index_keys.iterrows()):
        plate = _df['Metadata_Plate']
        well = _df['Metadata_Well']
        well_level_df = df.loc[(df['Metadata_Plate'] == plate) & (df['Metadata_Well'] == well)]
        avg_feature_df = well_level_df[feature_cols].mean(axis=0).tolist()
        meta_df = well_level_df[non_feature_cols].iloc[0].tolist()
        values = meta_df + avg_feature_df
        res_values.append(values)
    averaged_df = pd.DataFrame(res_values, columns = full_cols)
    return averaged_df


def get_moa(inchikeys):
    if inchikeys in INCHIKEY2MOA:
        return INCHIKEY2MOA[inchikeys]
    else:
        return np.nan

def read_file_embeddings(cpfname, fname, f_dim=512, feature_cols="micon_feat_", is_moa=False):
    if cpfname.split('.')[-1] == 'parquet':
        df_check = pd.read_parquet(cpfname)
    elif cpfname.split('.')[-1] == 'csv':
        df_check = pd.read_csv(cpfname)
    df_check["Metadata_Fov"] = df_check["Metadata_Fov"].astype(int)
    if is_moa:
        df_check["Metadata_Moa"] = df_check["Metadata_InChIKey"].apply(lambda x: get_moa(x)).tolist()
    with open(fname, "rb") as f:
        emb, fname = pkl.load(f)

    f_name = []
    for x in fname:
        [f_name.extend([t.split("$")]) for t in x]

    df_emb = pd.DataFrame({"Metadata_Plate": [x[2] for x in f_name], "Metadata_Well": [x[3] for x in f_name], "Metadata_Fov": [int(x[4]) for x in f_name]})
    for i in range(f_dim):
        df_emb = pd.concat([df_emb, pd.DataFrame({f"{feature_cols}{i}": emb[:, i]})], axis=1)

    df_check = df_check.merge(df_emb, on=["Metadata_Plate", "Metadata_Well", "Metadata_Fov"])
    return df_check

def generate_cp(selected_plates):
    # Generate cp features 
    profile_formatter = (
        "s3://cellpainting-gallery/cpg0016-jump/"
        "{Metadata_Source}/workspace/profiles/"
        "{Metadata_Batch}/{Metadata_Plate}/{Metadata_Plate}.parquet"
    )
    _dframes = pd.DataFrame()

    for _, row in tqdm(selected_plates.iterrows(), total=len(selected_plates)):
        s3_path = profile_formatter.format(**row.to_dict())
        df = pd.read_parquet(s3_path, storage_options={"anon": True})
        plate_name = row["Metadata_Plate"]
        df.to_parquet(f"jump_cp_target2/{plate_name}.parquet")
        _dframes = pd.concat([_dframes, df], axis=0, ignore_index=True)
        
    # dframes = []
    # for p in tqdm(metadata_target2['Metadata_Plate'].unique()):
    #     try:
    #         dframes.append(pd.read_parquet(f"jump_cp_compound/{p}.parquet"))
    #     except:
    #         dframes.append(pd.read_parquet(f"jump_cp_target2/{p}.parquet"))
    # dframes = pd.concat(dframes)

def transform_pd_to_ad(df, feature_col):
    meta_cols = [c for c in df.columns if c.startswith("Metadata_")]
    features = df[feature_col].to_numpy()
    adata = ad.AnnData(features)
    adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
    adata.var_names = feature_col
    for col in meta_cols:
        adata.obs[col] = pd.Categorical(df[col].to_numpy())
    return adata

def calculate_metric(annData_orig, annData_untreated, annData_untreated_2, exp_key, batch_key='Metadata_Source'):
    annData_scanorama = scib.ig.scanorama(annData_untreated, batch=batch_key)
    annData_harmony = scib.ig.harmony(annData_untreated_2, batch=batch_key)
    scib.preprocessing.reduce_data(
    annData_orig, n_top_genes=min(2000, len(annData_orig.var)), batch_key=batch_key, pca=True, neighbors=False
    )

    scanorama_res = scib.metrics.metrics(annData_scanorama, annData_scanorama, embed='X_scanorama', batch_key=batch_key, label_key="Metadata_SMILES",
                    ari_=True, nmi_=True, pcr_=False, silhouette_=True, isolated_labels_=True, graph_conn_=True,
                    kBET_=False, lisi_graph_=True).dropna().rename(columns={0:f'{exp_key}_{batch_key.split("_")[1]}_scanorama'})
    
    
    harmony_res = scib.metrics.metrics(annData_harmony, annData_harmony, embed='X_emb', batch_key=batch_key, label_key="Metadata_SMILES",
                    ari_=True, nmi_=True, pcr_=False, silhouette_=True, isolated_labels_=True, graph_conn_=True,
                    kBET_=False, lisi_graph_=True).dropna().rename(columns={0:f'{exp_key}_{batch_key.split("_")[1]}_harmony'})
    
    orig_res = scib.metrics.metrics(annData_orig, annData_orig, embed='X_pca', batch_key=batch_key, label_key="Metadata_SMILES",
                    ari_=True, nmi_=True, pcr_=False, silhouette_=True, isolated_labels_=True, graph_conn_=True,
                    kBET_=False, lisi_graph_=True).dropna().rename(columns={0:f'{exp_key}_{batch_key.split("_")[1]}_baseline'})
    
    res = pd.concat([orig_res, scanorama_res, harmony_res], axis=1)
    # res = orig_res
    print("##################  Finished one metric calculation complete ##################\n")
    return res

if __name__ == "__main__":
    cpfname = "embeddings/multi-source_treated_moa_target2_test.centered.parquet"

    test_fname = ["embeddings/multi-source_treated_moa_target2_new_raw_embeddings_supcon_freeze_img_test_47000.pkl", 
                "embeddings/multi-source_treated_moa_target2_new_raw_embeddings_supcon_freeze_img_test_42000.pkl",]
                #  "embeddings/single-source_treated_moa_target2_new_raw_embeddings_supcon_freeze_img_test_47000_generated.pkl"]

    # test_fname += [("embeddings/multi-source_treated_moa_target2_new_raw_embeddings_supcon_freeze_img_test_47000.pkl", 
    #              "embeddings/multi-source_treated_moa_target2_new_raw_embeddings_supcon_freeze_img_test_47000_generated.pkl")]
    
    tasks = ["double_original", "single_original"]

    # tasks = ['coembed']
    
    for fname_total, task in list(zip(test_fname, tasks)):
        if task == "coembed":
            fname, fname2 = fname_total
        else:
            fname = fname_total
        treated_moa_raw = read_file_embeddings(cpfname, fname, f_dim=1000, feature_cols="micon_train_", is_moa=True)
        treated_moa_raw_avg = average_wells(treated_moa_raw, feature_cols="micon_train_")
        cp_cols = [c for c in treated_moa_raw_avg.columns if not c.startswith("Metadata_") and not c.startswith("micon_") and not c.endswith("_path")]
        micon_cols = [c for c in treated_moa_raw_avg.columns if c.startswith("micon_")]

        if task == "coembed":
            treated_moa_raw_2 = read_file_embeddings(cpfname, fname2, f_dim=1000, feature_cols="micon_train_", is_moa=True)
            treated_moa_raw_avg_2 = average_wells(treated_moa_raw_2, feature_cols="micon_train_")
            treated_moa_raw_avg = pd.concat([treated_moa_raw_avg, treated_moa_raw_avg_2])

        print("##################  Reading complete ##################")



        treated_moa_raw_avg_processed_plate = plate_wise_spherize_and_normailize(treated_moa_raw_avg, plate_col="Metadata_Plate", feature_cols=cp_cols, control_only=True)
        treated_moa_raw_avg_processed_plate = plate_wise_spherize_and_normailize(treated_moa_raw_avg_processed_plate, plate_col="Metadata_Plate", feature_cols=micon_cols, control_only=True)
        treated_moa_raw_avg_processed_plate = treated_moa_raw_avg_processed_plate.loc[treated_moa_raw_avg_processed_plate["Metadata_SMILES"] != "CS(C)=O"]
        print("##################  Plate wise sphere complete ##################")

        treated_moa_raw_avg_processed_source = plate_wise_spherize_and_normailize(treated_moa_raw_avg, plate_col="Metadata_Source", feature_cols=cp_cols, control_only=True)
        treated_moa_raw_avg_processed_source = plate_wise_spherize_and_normailize(treated_moa_raw_avg_processed_source, plate_col="Metadata_Source", feature_cols=micon_cols, control_only=True)
        treated_moa_raw_avg_processed_source = treated_moa_raw_avg_processed_source.loc[treated_moa_raw_avg_processed_source["Metadata_SMILES"] != "CS(C)=O"]
        print("##################  Source wise sphere complete ##################")

        treated_moa_raw_avg_no_processed = treated_moa_raw_avg.loc[treated_moa_raw_avg["Metadata_SMILES"] != "CS(C)=O"].copy(deep=True)


        res = pd.concat([
            calculate_metric(transform_pd_to_ad(treated_moa_raw_avg_processed_plate, cp_cols),  transform_pd_to_ad(treated_moa_raw_avg_processed_plate, cp_cols), transform_pd_to_ad(treated_moa_raw_avg_processed_plate, cp_cols), exp_key="CP", batch_key="Metadata_Plate"),
            calculate_metric(transform_pd_to_ad(treated_moa_raw_avg_processed_plate, micon_cols),  transform_pd_to_ad(treated_moa_raw_avg_processed_plate, micon_cols), transform_pd_to_ad(treated_moa_raw_avg_processed_plate, micon_cols), exp_key="MICON", batch_key="Metadata_Plate"),
            calculate_metric(transform_pd_to_ad(treated_moa_raw_avg_processed_source, cp_cols),  transform_pd_to_ad(treated_moa_raw_avg_processed_source, cp_cols), transform_pd_to_ad(treated_moa_raw_avg_processed_source, cp_cols), exp_key="CP", batch_key="Metadata_Source"),
            calculate_metric(transform_pd_to_ad(treated_moa_raw_avg_processed_source, micon_cols),  transform_pd_to_ad(treated_moa_raw_avg_processed_source, micon_cols), transform_pd_to_ad(treated_moa_raw_avg_processed_source, micon_cols), exp_key="MICON", batch_key="Metadata_Source"),
            calculate_metric(transform_pd_to_ad(treated_moa_raw_avg_no_processed, cp_cols),  transform_pd_to_ad(treated_moa_raw_avg_no_processed, cp_cols), transform_pd_to_ad(treated_moa_raw_avg_no_processed, cp_cols), exp_key="CP-No-MAD-Sphering", batch_key="Metadata_Source"), 
            calculate_metric(transform_pd_to_ad(treated_moa_raw_avg_no_processed, micon_cols),  transform_pd_to_ad(treated_moa_raw_avg_no_processed, micon_cols), 
            transform_pd_to_ad(treated_moa_raw_avg_no_processed, micon_cols), exp_key="MICON-No-MAD-Sphering", batch_key="Metadata_Source"),                
            ], axis=1)


        # res = pd.concat([
        #     calculate_metric(transform_pd_to_ad(treated_moa_raw_avg_processed_plate, cp_cols),  transform_pd_to_ad(treated_moa_raw_avg_processed_plate, cp_cols), transform_pd_to_ad(treated_moa_raw_avg_processed_plate, cp_cols), exp_key="CP", batch_key="Metadata_Plate"),
        #     calculate_metric(transform_pd_to_ad(treated_moa_raw_avg_processed_plate, micon_cols),  transform_pd_to_ad(treated_moa_raw_avg_processed_plate, micon_cols), transform_pd_to_ad(treated_moa_raw_avg_processed_plate, micon_cols), exp_key="MICON", batch_key="Metadata_Plate"),
        #     calculate_metric(transform_pd_to_ad(treated_moa_raw_avg_no_processed, cp_cols),  transform_pd_to_ad(treated_moa_raw_avg_no_processed, cp_cols), transform_pd_to_ad(treated_moa_raw_avg_no_processed, cp_cols), exp_key="CP-No-MAD-Sphering", batch_key="Metadata_Plate"), 
        #     calculate_metric(transform_pd_to_ad(treated_moa_raw_avg_no_processed, micon_cols),  transform_pd_to_ad(treated_moa_raw_avg_no_processed, micon_cols), 
        #     transform_pd_to_ad(treated_moa_raw_avg_no_processed, micon_cols), exp_key="MICON-No-MAD-Sphering", batch_key="Metadata_Plate"),    
        #     ], axis=1)
        
        res.to_csv(f"scib_result_multi_source_{task}.csv")
        print(f"###### Finished task multi-source {task} ##########")