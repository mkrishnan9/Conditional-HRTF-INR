The data that we use in this project consists of HUTUBS, CIPIC, ARI, SCUT, and CHEDAR. Datasets may be downloaded at https://www.sofaconventions.org/mediawiki/index.php/Files

Required libraries are in requirements.txt.

After obtaining data, please run anthro_preprocess.py to save the .sofa files as preprocessed .pkl files.

Since the amount of data is very minimal, we are doing a 5-fold cross validation to have confidence in results.

Run ```python train_cv.py --include_datasets "hutubs,cipic" --data_path DATA_PATH```

CHEDAR data is numerically simulated and hence I am planning on not including it in validation data.

