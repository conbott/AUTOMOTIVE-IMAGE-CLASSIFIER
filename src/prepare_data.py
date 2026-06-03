import os
import shutil
import scipy.io
from sklearn.model_selection import train_test_split

def prepare_data():
    base_dir = r"c:\Users\con1k\Documents\PROJECTS\automotive-classification\data"
    devkit_dir = os.path.join(base_dir, "car_devkit", "devkit")
    train_dir_raw = os.path.join(base_dir, "cars_train")
    
    # Check if nested cars_train exists due to extraction
    if not os.path.exists(train_dir_raw) or len(os.listdir(train_dir_raw)) == 1 and "cars_train" in os.listdir(train_dir_raw):
        # some extractors create a nested folder
        train_dir_raw = os.path.join(train_dir_raw, "cars_train") if os.path.exists(os.path.join(train_dir_raw, "cars_train")) else train_dir_raw

    processed_dir = os.path.join(base_dir, "processed")
    train_processed = os.path.join(processed_dir, "train")
    val_processed = os.path.join(processed_dir, "val")

    print("Loading MATLAB annotations...")
    meta = scipy.io.loadmat(os.path.join(devkit_dir, "cars_meta.mat"))
    class_names = [c[0] for c in meta['class_names'][0]]
    
    # Sanitize class names for folder names
    class_names = [name.replace("/", "-") for name in class_names]

    annos = scipy.io.loadmat(os.path.join(devkit_dir, "cars_train_annos.mat"))
    annotations = annos['annotations'][0]

    images_and_labels = []
    for anno in annotations:
        fname = anno['fname'][0]
        # label in matlab is 1-indexed, convert to 0-indexed
        label_idx = int(anno['class'][0][0]) - 1 
        class_name = class_names[label_idx]
        images_and_labels.append((fname, class_name))

    print(f"Found {len(images_and_labels)} training images with labels.")

    # Split into train and validation (80% / 20%)
    train_data, val_data = train_test_split(images_and_labels, test_size=0.2, random_state=42, stratify=[l for _, l in images_and_labels])

    def copy_files(data_list, target_dir):
        os.makedirs(target_dir, exist_ok=True)
        count = 0
        for fname, class_name in data_list:
            src_path = os.path.join(train_dir_raw, fname)
            
            if not os.path.exists(src_path):
                continue
                
            class_dir = os.path.join(target_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            dst_path = os.path.join(class_dir, fname)
            shutil.copy(src_path, dst_path)
            count += 1
        return count

    print("Copying training files...")
    train_count = copy_files(train_data, train_processed)
    print(f"Copied {train_count} files to train.")

    print("Copying validation files...")
    val_count = copy_files(val_data, val_processed)
    print(f"Copied {val_count} files to validation.")

    print("Done preparing dataset!")

if __name__ == "__main__":
    prepare_data()
