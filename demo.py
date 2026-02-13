import os

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def load_itk_image(filename):

    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing

def world_to_voxel(world_coord, origin, spacing):
    stretched_voxel_coord = np.absolute(world_coord - origin)
    voxel_coord = stretched_voxel_coord / spacing
    return voxel_coord.astype(int)

df = pd.read_csv('annotations.csv')
subset_path = "subset0/subset0/"
available_paths = [f[:-4] for f in os.listdir(subset_path) if f.endswith('.mhd')]
my_annotations = df[df['seriesuid'].isin(available_paths)]
if my_annotations.empty:
    print(f"No matches found!")
else:
    # Let's look at the first nodule found in our folder
    sample_row = my_annotations.iloc[0]
    uid = sample_row['seriesuid']

    # World Coordinates from CSV are (X, Y, Z)
    world_coord = np.array([sample_row['coordZ'], sample_row['coordY'], sample_row['coordX']])
    diameter = sample_row['diameter_mm']

    # 3. Load the image
    img_path = os.path.join(subset_path, f"{uid}.mhd")
    image, origin, spacing = load_itk_image(img_path)

    # 4. Convert World -> Voxel
    v_z, v_y, v_x = world_to_voxel(world_coord, origin, spacing)

    print(f"Nodule found for patient: {uid}")
    print(f"World Coords (Z,Y,X): {world_coord}")
    print(f"Voxel Coords (Z,Y,X): {v_z, v_y, v_x}")

    # 5. Visualize the exact slice the nodule is on
    level, width = -600, 1500
    plt.imshow(image[v_z], cmap='gray', vmin=level - width / 2, vmax=level + width / 2)

    # Draw a circle/marker on the nodule
    plt.scatter(v_x, v_y, s=100, edgecolors='r', facecolors='none', label='Nodule')

    plt.title(f"Nodule at Slice {v_z} (Diameter: {diameter:.2f}mm)")
    plt.legend()
    plt.show()
