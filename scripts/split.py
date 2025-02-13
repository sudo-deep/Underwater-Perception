import os
import random
import shutil

def split_images(source_dir, train_dir, test_dir, test_ratio=0.2, seed=42):
    random.seed(seed)  # Set the seed for reproducibility
    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    random.shuffle(image_files)
    test_count = int(len(image_files) * test_ratio)
    test_files = image_files[:test_count]
    train_files = image_files[test_count:]

    for f in train_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(train_dir, f))

    for f in test_files:
        shutil.copy(os.path.join(source_dir, f), os.path.join(test_dir, f))

    print("Train files:", len(train_files))
    print("Test files:", len(test_files))


if __name__ == "__main__":
    source_folder = "data/oceandark/images"  # Set the path to your folder here.
    train_folder = "data/oceandark/train"
    test_folder = "data/oceandark/test"
    split_images(source_folder, train_folder, test_folder, test_ratio=0.2, seed=42)