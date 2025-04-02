import os
import shutil
import random

# Paths
dataset_path = "datasets/dataset"
image_dir = os.path.join(dataset_path, "images")
label_dir = os.path.join(dataset_path, "labels")

# Define train/val/test split
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Output directories
splits = ["train", "val", "test"]
for split in splits:
    os.makedirs(os.path.join(dataset_path, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, split, "labels"), exist_ok=True)

# Get all image files
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith((".jpg"))])
random.shuffle(image_files)

# Split dataset
total = len(image_files)
train_split = int(total * train_ratio)
val_split = int(total * (train_ratio + val_ratio))

train_files = image_files[:train_split]
val_files = image_files[train_split:val_split]
test_files = image_files[val_split:]

# Function to move files
def move_files(file_list, split):
    for img_file in file_list:
        img_src = os.path.join(image_dir, img_file)
        img_dst = os.path.join(dataset_path, split, "images", img_file)

        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_src = os.path.join(label_dir, label_file)
        label_dst = os.path.join(dataset_path, split, "labels", label_file)

        shutil.move(img_src, img_dst)
        if os.path.exists(label_src):
            shutil.move(label_src, label_dst)

# Move files
move_files(train_files, "train")
move_files(val_files, "val")
move_files(test_files, "test")

print("Dataset split completed!")
