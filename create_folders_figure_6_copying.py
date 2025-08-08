import os
import shutil
import argparse
import numpy as np

def prepare_folders(model_name, source_root, dest_root):
    print(f"Starting preparation for model: {model_name}")
    source_path = os.path.join(source_root, f"CIFAR10-{model_name}")
    dest_path = os.path.join(dest_root, f"CIFAR10-{model_name}")
    os.makedirs(dest_path, exist_ok=True)
    print(f"Destination path created: {dest_path}")
    
    # Determine number of classes (subfolders 0, 1, 2, ..., 9)
    class_folders = sorted([str(i) for i in range(10)])
    print(f"Detected {len(class_folders)} class folders.")
    print(f"{class_folders=}")
    n_samples_per_class = 1000
    # Select 100 samples per class
    selected_samples = {}
    for cls in class_folders:
        cls_source_path = os.path.join(source_path, cls)
        if not os.path.exists(cls_source_path):
            raise ValueError(f"Class folder {cls_source_path} does not exist.")
        
        all_files = os.listdir(cls_source_path)
        if len(all_files) < n_samples_per_class:
            raise ValueError(f"Not enough images in class {cls} of {model_name}. Found {len(all_files)}, required {n_samples_per_class}.")
        selected_samples[cls] = np.random.choice(all_files, n_samples_per_class, replace=False).tolist()

    print(f"{selected_samples.keys()=}")
    
    # Define subsets
    subsets = {}
    total_num_samples = 10000
    n_samples_per = [100, 200, 500, 1000]
    for n_samples in n_samples_per:
        multiplier = total_num_samples // (n_samples * len(class_folders))
        samples = []
        for cls in class_folders:
            imgs = selected_samples[cls][:n_samples] * multiplier
            samples.extend((img, cls) for img in imgs)
        print(f"{n_samples=}, {multiplier=}, {len(samples)=}, {class_folders=}, {len(selected_samples['0'][:n_samples])=}")
        subsets[n_samples] = samples
    # Create subsets
    for subset_name, images in subsets.items():
        subset_path = os.path.join(dest_path, str(subset_name))
        print(f"{subset_path=}")
        os.makedirs(subset_path, exist_ok=True)
        print(f"Creating subset: {subset_name} at {subset_path}")
        
        # Copy images with unique filenames
        for idx, img_tuple in enumerate(images):
            # Determine which class folder the image belongs to
            img, cls = img_tuple 
            cls = [cls for cls in selected_samples if img in selected_samples[cls]][0]
            cls_source_path = os.path.join(source_path, cls, img)
            if os.path.exists(cls_source_path):
                # Create a unique filename by appending an index
                unique_filename = f"{cls}_{idx}_{img}"
                shutil.copy(cls_source_path, os.path.join(subset_path, unique_filename))
            else:
                print(f"Warning: File {cls_source_path} not found, skipping.")
        
        print(f"Finished copying {len(images)} images to {subset_path}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare CIFAR10 subsets with specific class distributions.")
    args = parser.parse_args()
    
    SOURCE_ROOT = "/shared/sets/datasets/CIFAR10-dgm_eval"
    DEST_ROOT = "/shared/sets/datasets/PALATE_FIG_6_DUPLICATION"
    
    models = ["StyleGAN-XL", "PFGMPP", "MHGAN", "StyleGAN2-ada", "BigGAN-Deep"]
    for model_name in models:
        prepare_folders(model_name, SOURCE_ROOT, DEST_ROOT)
