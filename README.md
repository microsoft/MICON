## Code Repository for MICON
This is the repository for paper ["Causal integration of chemical structures in self-supervised learning improves representations of microscopy images for morphological profiling"](Link). 

#### Requirements
- PIP environment requirements can be found in `./requirements.txt` for installation.

#### Dataset Creation

- Please use the official JUMP Cell Painting dataset [download link](https://github.com/gwatkinson/jump_download/blob/main/jump_download/metadata/download_jump_metadata.py) to obtain the metadata for dataset creation and save under folder `./data/dataset/metadata/`

- Follow the notebook `./data/Dataset_Creation.ipynb` to create two ID and OOD dataset splits for pos-control and target-2 dataset. 
  - For generating OOD knn splits, use flag `knn=True` for `sample_train_test_OOD_posctl()` and `sample_train_test_OOD_tgt2()`.


#### Model Training/Generate

- Modify the  configs `/config/default.yaml` and `/config/setup/encoders.yaml` to adjust different model archetecture and training/inference strategy.

- Run `python main.py` with config `mode=train` for training and `mode=generate_embedding` for generating embeddings.

- We also provided checkpoints for pre-trained models which are available in [Zenodo]()

#### KNN-retrieval Experiments

- You can download the curated embeddings for Cell-Profiler and other methods from [Zenodo]() and save it under folder `./embeddings/`.

- Use the notebook `./analiyze_tools/analyze.ipynb` to calculate knn accuracies for different methods.
    - An example for analyzing Cell-Profiler/ micon embedding accuracies:
    ```
    cp_fname = "embeddings/pos_control.centered.parquet"
    model_fname = "embeddings/micon_embeddings.pkl"

    pos_control = read_file_embeddings(cp_fname, model_fname, f_dim=1000, feature_cols="micon_") 
    cp_cols = [c for c in pos_control.columns if not c.startswith("Metadata_") and not c.startswith("micon_") and not c.endswith("_path")]
    micon_cols = [c for c in pos_control.columns if c.startswith("micon_")]

    # Averaging fov features for single well statistics
    pos_control = average_wells(pos_control, feature_cols="micon_") 

    # You could change plate_col = (Metadata_Batch/Metadata_Plate/Metadata_Source) to adjust the scope of Control image for standardization
    # pos_control_processed = plate_wise_spherize_and_normailize(pos_control, plate_col="Metadata_Batch", feature_cols=cp_cols, control_only=True) 

    # Use NS_metric_across to calculate NSB(`on="Metadata_Batch"`)/NSS(`on="Metadata_Source"`) for topk retrieval statistics
    NS_metric_across(RETRIEVAL_SET, QUERY_SET, feature_col=cp_cols, on="Metadata_Batch", topk=10, all_negative=False, return_smiles=False)
    ```

### Citation

- WIP for link.