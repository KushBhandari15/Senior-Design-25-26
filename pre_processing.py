import os
import SimpleITK as sitk
import numpy as np
import pandas as pd

def resample_image(itk_image, out_spacing=[1.0, 1.0, 1.0]):
    # 1. Get current metadata
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    # 2. Calculate New Size
    # Formula: New Size = (Old Size * Old Spacing) / New Spacing
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    # 3. Set up the "Redrawer" (Resampler)
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(new_size)

    # Keep the image in the same spot in space
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetOutputDirection(itk_image.GetDirection())

    # 4. Fill in the gaps
    # Linear is great for the image itself
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetTransform(sitk.Transform())

    return resample.Execute(itk_image)


def normalize_and_clip(numpy_image, min_hu = -1000, max_hu = 400):

    clipped_image = np.clip(numpy_image, min_hu, max_hu)
    normalized_image = (clipped_image - min_hu) / (max_hu - min_hu)
    return normalized_image.astype(np.float32)

def extract_patch_and_mask(resampled_numpy_image, world_coord, origin, diameter = None, patch_size=32):

    v_z = int(round(world_coord[0] - origin[0]))
    v_y = int(round(world_coord[1] - origin[1]))
    v_x = int(round(world_coord[2] - origin[2]))
    half = patch_size // 2
    patch = resampled_numpy_image[
            v_z -half: v_z + half,
            v_y -half: v_y + half,
            v_x -half: v_x + half
    ]

    if patch.shape != (patch_size, patch_size, patch_size):
        return None, None

    if diameter:
        mask_patch = np.zeros((patch_size, patch_size, patch_size), dtype=np.float32)
        radius = diameter / 2.0
        z, y, x = np.ogrid[-half:half, -half:half, -half:half]
        mask_filter = x**2 + y**2 + z**2 <= radius**2
        mask_patch[mask_filter] = 1.0

        return patch, mask_patch

    return patch, None

def save_patches(subset = 'subset0'):

    # Make new directory to save .npy files
    os.makedirs('data/pos', exist_ok=True)
    os.makedirs('data/neg', exist_ok=True)
    os.makedirs('data/mask', exist_ok=True)

    df_ann = pd.read_csv('annotations.csv')
    df_cand = pd.read_csv('candidates.csv')
    subset_path = f"{subset}/{subset}"

    uids = [p[:-4] for p in os.listdir(subset_path) if p.endswith('.mhd')]
    for uid in uids:
        print(f"Processing patient: {uid}")
        img_path = os.path.join(subset_path, f"{uid}.mhd")

        # Preprocessing
        itk_image = sitk.ReadImage(img_path)
        resampled_itk = resample_image(itk_image)

        # Convert to numpy and normalize
        numpy_image = sitk.GetArrayFromImage(resampled_itk)
        clean_image = normalize_and_clip(numpy_image)
        new_origin = np.array(list(reversed(resampled_itk.GetOrigin())))

        # Extract positives
        p_list = df_ann[df_ann['seriesuid'] == uid]
        for i, row in p_list.iterrows():
            world_coord = np.array([row['coordZ'], row['coordY'], row['coordX']])
            patch, mask_patch = extract_patch_and_mask(clean_image, world_coord, new_origin, row['diameter_mm'])
            if patch is not None:
                np.save(f'data/pos/{uid}_p{i}.npy', patch)
                np.save(f'data/mask/{uid}_p{i}.npy', mask_patch)

        # Extract negatives
        n_list = df_cand[(df_cand['seriesuid'] == uid) & (df_cand['class'] == 0)]
        n_list = n_list.sample(n=min(len(p_list), len(n_list))) # Equal number of positives and negatives patches
        for i, row in n_list.iterrows():
            world_coord = np.array([row['coordZ'], row['coordY'], row['coordX']])
            patch, _ = extract_patch_and_mask(clean_image, world_coord, new_origin)
            if patch is not None:
                np.save(f'data/neg/{uid}_n{i}.npy', patch)

if __name__ == "__main__":
    save_patches('subset0')
    save_patches('subset1')