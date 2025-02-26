## Code Repository for MICON
This is the repository for paper ["Causal integration of chemical structures in self-supervised learning improves representations of microscopy images for morphological profiling"](Link). 

#### Requirements
- PIP environment requirements can be found in `./requirements.txt` for installation.

#### Dataset Creation

- Please use the official JUMP Cell Painting dataset [download link](https://github.com/gwatkinson/jump_download/blob/main/jump_download/metadata/download_jump_metadata.py) to obtain the metadata for dataset creation and save under folder `./data/dataset/metadata/`

- Follow the notebook `./data/Dataset_Creation.ipynb` to create two ID and OOD dataset splits for pos-control and target-2 dataset. 
  - For generating OOD knn splits, use flag `knn=True` for `sample_train_test_OOD_posctl()` and `sample_train_test_OOD_tgt2()`.


#### Model Training/Inference

- Modify the  configs `/config/default.yaml` and `/config/setup/encoders.yaml` to adjust different model archetecture and training/inference strategy.

- Run `python main.py` with config `mode=train` for training and `mode=generate_embedding` for generating embeddings.

- We also provided checkpoints for pre-trained models which are available in [link](https://www.dropbox.com/scl/fo/93yi5kr2878xznihjcf9w/APySE0JdKPbcM25e2UGF3cw?rlkey=casz2x8z3dwxgg99yom8ip1jb&st=mqtyc7mm&dl=0)

#### KNN-retrieval Experiments
- After generating embeddings pickle file with `mode=generate_embedding`, you can combine the result with cell-profiler embeddings parquet to calculate Not-Same(NS) metrics.

- Use the notebook `./analiyze_tools/analyze.ipynb` to calculate knn accuracies for different methods.
    - An example for analyzing Cell-Profiler/ micon embedding accuracies:
    ```
    cp_fname = "embeddings/target2.centered.parquet"
    model_fname = "embeddings/micon_embeddings.pkl"

    pos_control = read_file_embeddings(cp_fname, model_fname, f_dim=1000, feature_cols="micon_") 
    cp_cols = [c for c in pos_control.columns if not c.startswith("Metadata_") and not c.startswith("micon_") and not c.endswith("_path")]
    micon_cols = [c for c in pos_control.columns if c.startswith("micon_")]

    # Averaging fov features for single well statistics
    pos_control = average_wells(pos_control, feature_cols="micon_") 

    # You could change plate_col = (Metadata_Batch/Metadata_Plate/Metadata_Source) to adjust the scope of Control image for standardization
    # pos_control_processed = plate_wise_spherize_and_normailize(pos_control, plate_col="Metadata_Batch", feature_cols=cp_cols, control_only=True) 
    ```
    
- You could also download the processed and averaged embeddings from [link](https://www.dropbox.com/scl/fo/93yi5kr2878xznihjcf9w/APySE0JdKPbcM25e2UGF3cw?rlkey=casz2x8z3dwxgg99yom8ip1jb&st=mqtyc7mm&dl=0)

- Use ```NS_metric_across`` to calculate metrics NSB(`on="Metadata_Batch"`)/NSS(`on="Metadata_Source"`) for topk retrieval statistics between query set and retrieval set. AA
  ```
   # calculate metrics NSB(`on="Metadata_Batch"`)/NSS(`on="Metadata_Source"`) for topk retrieval statistics
    NS_metric_across(RETRIEVAL_SET, QUERY_SET, feature_col=cp_cols, on="Metadata_Batch", topk=10, all_negative=False, return_smiles=False)
    ```

### Citation

- WIP for citation bib.