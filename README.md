# nuclei-segmentation-pytorch
Repository containing code to train a U-net segmentation model to segment nuclei.

Data downloaded from: https://www.kaggle.com/c/data-science-bowl-2018
The code can be implemented in two ways, through the notebook - NucleiSegmentation.ipynb or through the individual .py files.

Notebook Instructions:
1. Download the data. Extract Stage 1 images.
stage_1:
  train
  test
2. Change default values in the notebook if needed.
3. Run the notebook.

Manual Implementation instructions:
1. Run utils.py with arguments:
  --train_dir: Directory containing train images.
  --test_dir: Directory containing test images.
  --text_dir: Directory to store train and test files.
  --val_split_size: Validation split size.
  --random_seed: Random seed for the split to ensure reproducibility
  
  Example: python utils.py --train_dir stage_1/train/ --test_dir stage_1/test/ --text_dir stage_1/ --val_split_size: 0.10 --random_seed: 7
 
 This combines all the masks for each image into a single mask.
2. Make changes to the config file if needed.
3. Run train.py with arguments:
  --config: Path to config file
  
  Example: python train.py --config config.yml
 
4. Run test.py with arguments:
  --output_dir: Output directory of the experiment
  --threshold: Threshold for model prediction
