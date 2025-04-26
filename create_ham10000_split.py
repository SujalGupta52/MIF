import os
import json
import random
import argparse
from collections import defaultdict


def create_split_file(
    data_dir, output_file, train_ratio=0.7, val_ratio=0.15, kaggle_mode=False
):
    """
    Create a split file for the HAM10000 dataset.

    Args:
        data_dir: Path to the HAM10000 directory containing train and test subdirectories
        output_file: Path to save the split file
        train_ratio: Ratio of images to use for training (from train directory)
        val_ratio: Ratio of images to use for validation (from train directory)
        kaggle_mode: If True, adapts to Kaggle's directory structure
    """
    # Handle Kaggle directory structure
    if kaggle_mode:
        # In Kaggle, data is often in /kaggle/input/dataset-name/
        # Check if we're in a flat structure (common in Kaggle)
        if os.path.isdir(os.path.join(data_dir, "HAM10000_images_part_1")):
            train_dir = data_dir  # Treat main dir as containing all images
            test_dir = None
            # Will process differently below
        else:
            train_dir = os.path.join(data_dir, "train")
            test_dir = os.path.join(data_dir, "test")
    else:
        train_dir = os.path.join(data_dir, "train")
        test_dir = os.path.join(data_dir, "test")

    # Verify that data directory exists
    if not os.path.exists(data_dir):
        print(f"Error: {data_dir} does not exist!")
        return

    # Collect train images
    train_images = []
    class_counts = defaultdict(int)

    # Handle Kaggle HAM10000 specific structure
    if kaggle_mode and os.path.isdir(os.path.join(data_dir, "HAM10000_images_part_1")):
        # Get label information from metadata file
        metadata_file = os.path.join(data_dir, "HAM10000_metadata.csv")
        if not os.path.exists(metadata_file):
            print(
                f"Error: {metadata_file} not found. Kaggle HAM10000 dataset should include metadata CSV."
            )
            return

        import pandas as pd

        metadata = pd.read_csv(metadata_file)

        # Process HAM10000 images (may be in two folders)
        image_dirs = []
        if os.path.isdir(os.path.join(data_dir, "HAM10000_images_part_1")):
            image_dirs.append(os.path.join(data_dir, "HAM10000_images_part_1"))
        if os.path.isdir(os.path.join(data_dir, "HAM10000_images_part_2")):
            image_dirs.append(os.path.join(data_dir, "HAM10000_images_part_2"))

        all_images = []
        for img_dir in image_dirs:
            for img_name in os.listdir(img_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_id = os.path.splitext(img_name)[0]
                    all_images.append((img_id, os.path.join(img_dir, img_name)))

        # Create dictionary for faster lookup
        image_dict = {img_id: img_path for img_id, img_path in all_images}

        # Map each image to its class using metadata
        for _, row in metadata.iterrows():
            img_id = row["image_id"]
            class_name = row["dx"]  # Diagnosis is the class name

            if img_id in image_dict:
                img_path = image_dict[img_id]
                rel_path = os.path.relpath(img_path, data_dir)
                train_images.append((rel_path, class_counts[class_name], class_name))
                class_counts[class_name] += 1
    else:
        # Process regular train directory structure
        if not os.path.exists(train_dir):
            print(f"Error: {train_dir} does not exist!")
            return

        for class_name in os.listdir(train_dir):
            class_dir = os.path.join(train_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    rel_path = os.path.join("train", class_name, img_name)
                    train_images.append(
                        (rel_path, class_counts[class_name], class_name)
                    )
                    class_counts[class_name] += 1

    if not train_images:
        print("Error: No training images found!")
        return

    # Shuffle and split train images
    random.shuffle(train_images)
    total_train = len(train_images)
    train_end = int(total_train * train_ratio)
    val_end = train_end + int(total_train * val_ratio)

    train_data = train_images[:train_end]
    val_data = train_images[train_end:val_end]

    # Collect test images
    test_data = []

    if test_dir and os.path.exists(test_dir):
        for class_name in os.listdir(test_dir):
            class_dir = os.path.join(test_dir, class_name)
            if not os.path.isdir(class_dir):
                continue

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    rel_path = os.path.join("test", class_name, img_name)
                    test_data.append((rel_path, class_counts[class_name], class_name))
    else:
        # If no test directory, use remaining training data as test
        test_data = train_images[val_end:]

    # Create split dictionary
    split_data = {"train": train_data, "val": val_data, "test": test_data}

    # Save to file
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(split_data, f, indent=4)

    # Print statistics
    print(f"Split created at {output_file}")
    print(f"Train: {len(train_data)} images")
    print(f"Val: {len(val_data)} images")
    print(f"Test: {len(test_data)} images")
    print(f"Classes: {len(class_counts)}")
    for cls, count in class_counts.items():
        print(f"  {cls}: {count} images")

    return split_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a split file for HAM10000 dataset"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to HAM10000 directory"
    )
    parser.add_argument(
        "--output", type=str, default="split_HAM10000.json", help="Output file path"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.7, help="Ratio of images for training"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.15, help="Ratio of images for validation"
    )
    parser.add_argument(
        "--kaggle", action="store_true", help="Use Kaggle dataset structure"
    )

    args = parser.parse_args()
    create_split_file(
        args.data_dir, args.output, args.train_ratio, args.val_ratio, args.kaggle
    )


# Example usage in Kaggle notebook:
"""
# Uncomment and modify to run in Kaggle Jupyter notebook
import os

# Typical Kaggle paths
KAGGLE_INPUT_DIR = '/kaggle/input/ham10000-skin-cancer'  # Adjust to your dataset name
OUTPUT_DIR = '/kaggle/working'
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'split_HAM10000.json')

# Run the function directly
# split_data = create_split_file(KAGGLE_INPUT_DIR, OUTPUT_FILE, kaggle_mode=True)
"""
