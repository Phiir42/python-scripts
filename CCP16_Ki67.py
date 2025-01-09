# Step 2: Import necessary libraries
from readlif.reader import LifFile
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from skimage import measure, morphology
import pandas as pd

# Step 3: Define the folder containing the .lif files
folder_path = 'C:/Users/clchr/Downloads/CCP16a'

# Step 4: Prepare an empty list to store results for exporting to Excel
results = []

# Step 5: Iterate over all .lif files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.lif'):
        lif_file_path = os.path.join(folder_path, file_name)
        lif_file = LifFile(lif_file_path)

        # Step 6: Get the number of images in the lif file
        num_images = lif_file.num_images
        print(f"\nProcessing file: {file_name}")
        print(f"Number of images in the .lif file: {num_images}")

        # Step 7: Group the images into gels based on metadata from image names
        gels = {}
        for i in range(num_images):
            image = lif_file.get_image(i)
            image_name = image.name  # Get the image name to determine the gel and stack
            gel_id, stack_id = image_name.split('-')
            gel_key = f'Gel {gel_id}'
            if gel_key not in gels:
                gels[gel_key] = []
            gels[gel_key].append(i)

        # Step 8: Print the categorization results
        for gel, z_stacks in gels.items():
            print(f"{gel}: {', '.join([f'Z-stack {i+1}' for i in z_stacks])}")

        # Step 9: Iterate over each gel and z-stack
        for gel, z_stack_indices in gels.items():
            for idx in z_stack_indices:
                # Extract the specific z-stack image
                image = lif_file.get_image(idx)

                # Step 10: Verify the number of z-slices and channels in the z-stack
                num_z_slices = image.dims[2]  # The third element represents the number of z-slices
                num_channels = image.channels
                print(f"\n{gel} - Z-stack {idx + 1}: Number of z-slices: {num_z_slices}, Number of channels: {num_channels}")

                # Step 11: Create maximum intensity projection for each channel
                max_projections = []
                for c in range(num_channels):
                    max_projection = np.max([np.array(image.get_frame(z=z, t=0, c=c)) for z in range(num_z_slices)], axis=0)
                    max_projections.append(max_projection)

                # Step 12: Apply Gaussian blur to the DAPI and Ki-67 channels
                dapi_blurred = cv2.GaussianBlur(max_projections[0], (3, 3), 1)
                ki67_blurred = cv2.GaussianBlur(max_projections[2], (3, 3), 1)

                # Step 13: Threshold the DAPI and Ki-67 channels to create binary masks
                _, dapi_binary = cv2.threshold(dapi_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _, ki67_binary = cv2.threshold(ki67_blurred, 100, 255, cv2.THRESH_BINARY)
                
                # Step 14: Apply despeckling (remove small objects) to reduce non-specific signal
                dapi_binary = morphology.remove_small_objects(dapi_binary.astype(bool), min_size=20).astype(np.uint8) * 255
                ki67_binary = morphology.remove_small_objects(ki67_binary.astype(bool), min_size=10).astype(np.uint8) * 255

                # Step 15: Label DAPI objects
                dapi_labels = measure.label(dapi_binary)
                dapi_props = measure.regionprops(dapi_labels)

                # Step 16: Count Ki-67 positive and negative cells
                ki67_positive = 0
                ki67_negative = 0

                for prop in dapi_props:
                    # Create a mask for the current DAPI object
                    dapi_mask = dapi_labels == prop.label

                    # Check if there is any overlap with the Ki-67 binary mask
                    if np.any(np.logical_and(dapi_mask, ki67_binary)):
                        ki67_positive += 1
                    else:
                        ki67_negative += 1

                # Step 17: Calculate the ratio of Ki-67 positive cells to total cells
                total_cells = ki67_positive + ki67_negative
                ki67_ratio = ki67_positive / total_cells if total_cells > 0 else 0

                # Step 18: Store the results
                results.append({
                    'File Name': file_name,
                    'Gel': gel,
                    'Z-stack': f'Z-stack {idx + 1}',
                    'Total Cells': total_cells,
                    'Ki-67 Positive Cells': ki67_positive,
                    'Ki-67 Negative Cells': ki67_negative,
                    'Ki-67 Positive Ratio': ki67_ratio
                })

                # Step 19: Print the results for the current z-stack
                print(f"{gel} - Z-stack {idx + 1}: Ki-67 positive cells: {ki67_positive}, Ki-67 negative cells: {ki67_negative}, Ki-67 positive ratio: {ki67_ratio:.2f}")

                # Step 20: Define color maps for each channel
                colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # Blue for DAPI, Red for Phalloidin, Green for Ki-67

                # Step 21: Create an RGB image by combining channels
                rgb_image = np.zeros((max_projections[0].shape[0], max_projections[0].shape[1], 3), dtype=np.uint8)
                for i, max_projection in enumerate(max_projections):
                    color = colors[i]
                    for j in range(3):
                        rgb_image[:, :, j] += (max_projection * (color[j] / 255)).astype(np.uint8)

                # Step 22: Visualize the maximum intensity projection
                plt.imshow(rgb_image)
                plt.title(f'{gel} - Z-stack {idx + 1}: Maximum Intensity Projection from {file_name}')
                plt.axis('off')
                plt.show()

# Step 23: Save the results to an Excel spreadsheet
results_df = pd.DataFrame(results)
results_df.to_excel('Ki-67_analysis_results.xlsx', index=False)
print("Results have been saved to 'Ki-67_analysis_results.xlsx'")
