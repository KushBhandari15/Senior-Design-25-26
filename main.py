import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_model import NoduleSegmentation
from detection_model import NoduleDetection
import pre_processing as pre_processing
import SimpleITK as sitk
import matplotlib.pyplot as plt

class Lung_Cancer:
    def __init__(self, input_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.detection_model = NoduleDetection().to(self.device)
        self.detection_model.load_state_dict(torch.load('nodule_detection_model.pth'))
        self.detection_model.eval()

        self.segmentation_model = NoduleSegmentation().to(self.device)
        self.segmentation_model.load_state_dict(torch.load('nodule_segmentation_model.pth'))
        self.segmentation_model.eval()

        self.image_path = input_path

    def run_inference(self, patch_size=32, stride=16):

        itk_image = sitk.ReadImage(str(self.image_path))
        resampled_image = pre_processing.resample_image(itk_image)
        numpy_image = sitk.GetArrayFromImage(resampled_image)
        clean_image = pre_processing.normalize_and_clip(numpy_image)

        full_mask = np.zeros_like(clean_image)
        print(f"Analyzing scan volume: {clean_image.shape}...")
        for z in range(0, clean_image.shape[0] - patch_size, stride):
            for y in range(0, clean_image.shape[1] - patch_size, stride):
                for x in range(0, clean_image.shape[2] - patch_size, stride):

                    patch = clean_image[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                    patch_tensor = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        det_output = self.detection_model(patch_tensor)
                        _, predicted = torch.max(det_output, 1)
                        if predicted.item() == 1:
                            # print(f"Nodule detected at coordinates: Z{z}, Y{y}, X{x}")

                            seg_output = self.segmentation_model(patch_tensor)
                            seg_prob = torch.sigmoid(seg_output)
                            seg_mask = (seg_prob > 0.5).float().cpu().numpy().squeeze()

                            full_mask[z:z+patch_size, y:y+patch_size, x:x+patch_size] = np.maximum(
                                full_mask[z:z+patch_size, y:y+patch_size, x:x+patch_size],
                                seg_mask
                            )

        return clean_image, full_mask

    def save_results(self, clean_image, full_mask, output_name="prediction_result.mhd"):

        result_img = sitk.GetImageFromArray(full_mask)
        sitk.WriteImage(result_img, output_name)
        print(f"Result saved to {output_name}")

    def visualize_prediction(self, clean_image, full_mask):

        mask_sums = np.sum(full_mask, axis=(1, 2))
        best_slice = np.argmax(mask_sums)

        if mask_sums[best_slice] == 0:
            print(f"No nodule were found in this entire scan to visualize")
            best_slice = clean_image.shape[0] // 2

        print(f"Visualizing Slice {best_slice} (detected {mask_sums[best_slice]} nodule pixels)")

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(clean_image[best_slice], cmap='gray')
        plt.title(f"Original CT - Slice {best_slice}")
        plt.axis('off')
        # Subplot 2: AI Predicted Mask
        plt.subplot(1, 2, 2)
        # We overlay the mask on the image for better context
        plt.imshow(clean_image[best_slice], cmap='gray')
        plt.imshow(full_mask[best_slice], cmap='Reds', alpha=0.5)  # Red heat-map overlay
        plt.title("AI Nodule Detection (Red)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("prediction_viz.png")
        plt.show()

if __name__ == "__main__":
    test_file = "subset0/subset0/1.3.6.1.4.1.14519.5.2.1.6279.6001.105756658031515062000744821260.mhd"

    app = Lung_Cancer(test_file)
    original_volume, predicted_mask = app.run_inference()
    app.save_results(original_volume, predicted_mask)
    app.visualize_prediction(original_volume, predicted_mask)