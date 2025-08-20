# Conditional HRTF INR

## Updates (Aug 20 Wed) - MK

I have changed the model to **only take in ear parameters** (instead of ear and head) by taking an intersection of common measured ear features across datasets. No need to make any changes to currently preprocessed hutubs, cipic files. The code takes care of the change.

This allowed me to add a new dataset AXD SONICOM which has only ear measurements with anthropometry for 92 individuals. Training the model on just AXD data gives a validation LSD of ~4.6dB which is better than any other dataset. Combining this with HUTUBS, and CIPIC reduces the LSD to ~4.9dB which is better than the best performing method from earlier (which gave 4.98dB). **I have uploaded the preprocessed AXD data to the Google Drive.**

Currently, I am looking at removing a percentage of locations from the training subjects so that at validation we can evaluate seperately:
     (a) unseen subjects + seen locations
     (b) unseen subjects + unseen locations
     (c) seen subjects + unseen locations



## Previous Information

The data that we use in this project consists of HUTUBS, CIPIC, ARI, SCUT, and CHEDAR. Datasets may be downloaded at https://www.sofaconventions.org/mediawiki/index.php/Files

Required libraries are in requirements.txt.

After obtaining data, please run anthro_preprocess.py to save the .sofa files as preprocessed .pkl files.

Since the amount of data is very minimal, we are doing a 5-fold cross validation to have confidence in results.

Run ```python train_cv.py --include_datasets "hutubs,cipic" --data_path DATA_PATH```

CHEDAR data is numerically simulated and hence I am planning on not including it in validation data.

