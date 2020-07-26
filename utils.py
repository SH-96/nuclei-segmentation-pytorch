# Name: Sanjith Hebbar
# Date: 25-07-2020
# Description: Creating data files for training and testing u-net model.

# Import Libraries
import cv2
import os
import random
import argparse
import numpy as np

def combine_masks(data_dir):
    """
    Function to combine multiple masks into a single mask.
    Args:
    data_dir: input data directory containing images and masks.

    Example:
    data_dir = "stage_1/train/"
    combine_masks(data_dir)
    """
    for image_dir in os.listdir(data_dir):
        image_dir_path = os.path.join(data_dir, image_dir)
        mask_path = os.path.join(image_dir_path, 'masks')
        image_path = os.path.join(image_dir_path, 'images', image_dir+".png")
        og_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
        
        # Combined Mask
        new_mask_path = os.path.join(image_dir_path, 'mask')
        new_mask = np.zeros_like(og_img)
        
        if not os.path.exists(new_mask_path):
            os.makedirs(new_mask_path)
            
        for mask in os.listdir(mask_path):
            current_mask_path = os.path.join(mask_path, mask)
            img = cv2.cvtColor(cv2.imread(current_mask_path), cv2.COLOR_BGR2GRAY)
            idx = np.where(img == 255)
            new_mask[idx] = 255
        full_mask_path = os.path.join(new_mask_path, image_dir+".png")
        
        # Save combined mask
        cv2.imwrite(full_mask_path, new_mask)
        
    return

def get_image_paths(data_dir, data_type):
    """
    Function to create data dictionary containing image paths and mask paths.
    Args:
    data_dir: input data directory containing images and masks.
    data_type: "train" or "test".

    Example:
    data_dir = "stage_1/train/"
    train_dict = get_image_paths(data_dir, "train")
    """
    dataset = {}
    for image_dir in os.listdir(data_dir):
        image_dir_path = os.path.join(data_dir, image_dir)
        image_path = os.path.join(image_dir_path, 'images', image_dir+".png")
        mask_path = os.path.join(image_dir_path, 'mask', image_dir+".png")
        if(data_type == "train"):
            dataset[image_dir] = [image_path, mask_path]
        else:
            dataset[image_dir] = image_path

    return dataset

def create_split(train_dict, split_size = .10, seed = 7):
    """
    Function to create validation data from existing train data.
    Args:
    train_dict: Dictionary containing training images.
    split_size: Validation size.
    seed: Random seed for reproducing splits.

    Example:
    create_split(train_dict = train_dict, split_size = 0.10, seed = 7)
    """
    random.seed(seed)

    val_dict = dict(random.sample(list(train_dict.items()), int(len(train_dict)*split_size)))

    train_dict = {key:value for key, value in train_dict.items() if key not in val_dict.keys()}

    return train_dict, val_dict

def create_text_file(data_dict, data_type, img_txt_path, mask_txt_path = None):
    """
    Function to create text files containing image paths and mask paths
    Args:
    data_dict: Dictionary containing image paths and mask paths
    data_type: "train" or "test" data
    img_text_path: Path of text file containing image paths.
    img_mask_path: Path of text file containing mask paths.

    Example:
    create_text_file(train_dict, "train", "train_images.txt", "train_masks.txt")
    """
    if (data_type != "test"):
        with open(img_txt_path, 'w') as f:
            with open(mask_txt_path, 'w') as i:
                for key, value in data_dict.items():
                    f.write(value[0])
                    i.write(value[1])

                    f.write("\n")
                    i.write("\n")

    else:
        with open(img_txt_path, 'w') as f:
                for key, value in data_dict.items():
                    f.write(value)

                    f.write("\n")

    print("\n\n{0} files created.\nImages: {1}\nMasks: {2}".format(data_type.capitalize(), img_txt_path, mask_txt_path))
    return

def get_image_sample(data_loader):
    # Get Sample
    inputs, masks, idx = next(iter(data_loader))
    
    # Display Masks
    fig, axes = plt.subplots(1, 2)
    titles = ['Input', 'Mask']
    image_sets = [inputs[0], masks[0]]
    for i, axis in enumerate(axes):
        if(i == 0):
            axis.imshow(image_sets[i].squeeze(0).permute(1, 2, 0))
        else:
            axis.imshow(image_sets[i].squeeze(0), cmap = 'gray')
        axis.set_title(titles[i])

    print("Model Input Shape: ", inputs.shape)
    print("Masks Shape: ", masks.shape)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', required = True, help = "Parent directory containing training image directories.")
    parser.add_argument('--test_dir', required = True, help = "Parent directory containing testing image directories.")
    parser.add_argument('--text_dir', required = True, help = "Directory to store image and mask text files.")
    parser.add_argument('--val_split_size', required = True, help = "Validation split size. Must be in decimals. Example: 0.10")
    parser.add_argument('--random_seed', required = True, help = "Random Seed for splitting validation set.")
    args = parser.parse_args()

    # # Combine Masks into single masks
    # data_dir = args.train_dir
    # combine_masks(data_dir)
    print("\nMasks combined.")

    # Train Images
    train_dir = args.train_dir
    train_dict = get_image_paths(train_dir, "train")

    # Test Images
    test_dir = args.test_dir
    test_dict = get_image_paths(test_dir, "test")

    # Create validation split
    split_size = float(args.val_split_size)

    train_dict, val_dict = create_split(train_dict, split_size, seed = args.random_seed)

    train_img_txt = os.path.join(args.text_dir, "train_images.txt")
    train_mask_txt = os.path.join(args.text_dir, "train_masks.txt")
    val_img_txt = os.path.join(args.text_dir, "val_images.txt")
    val_mask_txt = os.path.join(args.text_dir, "val_masks.txt")
    test_img_txt = os.path.join(args.text_dir, "test_images.txt")

    create_text_file(train_dict, "train", train_img_txt, train_mask_txt)
    create_text_file(val_dict, "val", val_img_txt, val_mask_txt)
    create_text_file(test_dict, "test", test_img_txt)