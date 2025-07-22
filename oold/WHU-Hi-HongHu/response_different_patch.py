import os
import numpy as np
from tifffile import imread, imsave
from PIL import Image
from helpers.augmentations import RandomHorizontallyFlip, RandomVerticallyFlip, \
    RandomTranspose, Compose

# -------------------- Configuration --------------------
# Base directory of WHU-Hi-HongHu project
BASE_DIR = r""  # eg：.\oold\WHU-Hi-HongHu

# Original patch size (e.g., 32 for Image32_step_patch)
ORIG_PATCH_SIZE = 32
# Desired new patch size (e.g., 4, 8, or 16)
NEW_PATCH_SIZE = 4  # <-- Change this to 8 or 16 as needed

# Filenames for 32-sized lists
TXT_TRAIN = f"NEW_seed10_25_train.txt"
TXT_TEST  = f"NEW_seed10_25_test.txt"

# Input directories based on ORIG_PATCH_SIZE
INPUT_DIR = os.path.join(BASE_DIR, f"Image{ORIG_PATCH_SIZE}_step_patch")
DATA_IN  = os.path.join(INPUT_DIR, "HSI-rad")
LABEL_IN = os.path.join(INPUT_DIR, "Labels")
TXT_TRAIN_PATH = os.path.join(INPUT_DIR, TXT_TRAIN)
TXT_TEST_PATH  = os.path.join(INPUT_DIR, TXT_TEST)

# Output directories based on NEW_PATCH_SIZE
OUTPUT_DIR = os.path.join(BASE_DIR, f"Image{NEW_PATCH_SIZE}_step_patch")
DATA_OUT   = os.path.join(OUTPUT_DIR, "HSI-rad")
LABEL_OUT  = os.path.join(OUTPUT_DIR, "Labels")
os.makedirs(DATA_OUT, exist_ok=True)
os.makedirs(LABEL_OUT, exist_ok=True)

# Large full-image dimensions (fixed)
H, W, C = 928, 448, 270

# -------------------- Functions --------------------
def load_id_list(txt_file):
    with open(txt_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]


augs = []
augs.append(RandomHorizontallyFlip(p=1.0))  #水平翻转
augs_Horizon = Compose(augs)

augs2 = []
augs2.append(RandomVerticallyFlip(p=1.0))  #随机垂直翻转
augs_Vertically = Compose(augs2)

augs3 = []
augs3.append(RandomTranspose(p=1.0))  #转置
augs_Transpose = Compose(augs3)


def process_set(id_list, augment, out_txt_name):
    # Prepare large arrays and coverage mask
    data_large  = np.zeros((H, W, C), dtype=np.float32)
    label_large = np.zeros((H, W, 3), dtype=np.uint8)
    mask        = np.zeros((H, W), dtype=bool)
    collected_names = []

    # Place original patches
    for img_id in id_list:
        _, y_str, x_str = img_id.split('_')
        y0, x0 = int(y_str), int(x_str)
        data_patch = np.load(os.path.join(DATA_IN,  img_id + ".npy"))
        lbl_patch  = imread(os.path.join(LABEL_IN, img_id + ".tif"))
        data_large[y0:y0+ORIG_PATCH_SIZE, x0:x0+ORIG_PATCH_SIZE, :] = data_patch
        label_large[y0:y0+ORIG_PATCH_SIZE, x0:x0+ORIG_PATCH_SIZE] = lbl_patch
        mask[y0:y0+ORIG_PATCH_SIZE, x0:x0+ORIG_PATCH_SIZE] = True

    # Slide new patch window
    for y in range(0, H - NEW_PATCH_SIZE + 1, NEW_PATCH_SIZE):
        for x in range(0, W - NEW_PATCH_SIZE + 1, NEW_PATCH_SIZE):
            if not mask[y:y+NEW_PATCH_SIZE, x:x+NEW_PATCH_SIZE].all():
                continue
            base_name = f"image_{y}_{x}"
            patch_data = data_large[y:y+NEW_PATCH_SIZE, x:x+NEW_PATCH_SIZE, :]
            patch_lbl  = label_large[y:y+NEW_PATCH_SIZE, x:x+NEW_PATCH_SIZE]
            # Save
            np.save(os.path.join(DATA_OUT,   base_name + ".npy"), patch_data)
            imsave(os.path.join(LABEL_OUT,  base_name + ".tif"), patch_lbl)
            collected_names.append(base_name)

            print(patch_lbl.shape)  # (4, 4, 3)
            # Augment if requested
            if augment:
                # Horizontal flip
                h_data, h_lbl = augs_Horizon(patch_data, patch_lbl)  # HorizontallyFlip
                h_lbl = Image.fromarray(h_lbl)
                # h_data = np.flip(patch_data, axis=1)
                # h_lbl  = np.flip(patch_lbl, axis=1)
                h_name = base_name + "_Horizon"
                np.save(os.path.join(DATA_OUT,  h_name + ".npy"), h_data)
                h_lbl.save(os.path.join(LABEL_OUT, h_name + ".tif"))
                collected_names.append(h_name)
                # Vertical flip
                v_data, v_lbl = augs_Vertically(patch_data, patch_lbl)  # HorizontallyFlip
                v_lbl = Image.fromarray(v_lbl)
                # v_data = np.flip(patch_data, axis=0)
                # v_lbl  = np.flip(patch_lbl, axis=0)
                v_name = base_name + "_Vertically"
                np.save(os.path.join(DATA_OUT,  v_name + ".npy"), v_data)
                v_lbl.save(os.path.join(LABEL_OUT, v_name + ".tif"))
                collected_names.append(v_name)
                # Transpose
                t_data, t_lbl = augs_Transpose(patch_data, patch_lbl)
                t_lbl = Image.fromarray(t_lbl)
                # t_data = patch_data.transpose(1, 0, 2)
                # t_lbl  = patch_lbl.T
                t_name = base_name + "_Transpose"
                np.save(os.path.join(DATA_OUT,  t_name + ".npy"), t_data)
                t_lbl.save(os.path.join(LABEL_OUT, t_name + ".tif"))
                collected_names.append(t_name)

    # Write the new list
    with open(os.path.join(OUTPUT_DIR, out_txt_name + ".txt"), 'w') as f:
        for name in collected_names:
            f.write(name + "\n")


# -------------------- Main Execution --------------------
if __name__ == "__main__":
    train_ids = load_id_list(TXT_TRAIN_PATH)
    test_ids  = load_id_list(TXT_TEST_PATH)
    # Training: augment
    process_set(train_ids, augment=True,  out_txt_name="NEW_seed10_25_train_aug")
    # Testing: no augment
    process_set(test_ids,  augment=False, out_txt_name="NEW_seed10_25_test")
    print("Done: extracted", NEW_PATCH_SIZE, "×", NEW_PATCH_SIZE, "patches to", OUTPUT_DIR)
