"""
Script: plot_hyperspectral_data.py
Author: Chris Long

Description:
    This script searches a specified directory (and all its subfolders) for Excel 
    files that start with "Hyperspectral_Results". For each found file, it reads 
    the second sheet ("Normalized Data") where:

    - The first row (B1:AG1) contains the 32 wavenumbers (x-axis).
    - Each subsequent row is a normalized hyperspectral dataset for a 
      particular lipid feature.

    It then:
    1) Plots each lipid feature (each row) in one figure, applying spline smoothing.
       Any zero-valued data points are removed prior to fitting the spline to 
       avoid bad acquisition points.
    2) Computes the average (pointwise) of all valid spectra, again applies 
       spline smoothing, and plots this as a single line in a separate figure.

    Both the multi-line figure and the average-spectrum figure are saved as 
    .jpg images in a subfolder named "Plots", alongside the data files.

    A few key notes:
    - The legend is omitted as requested.
    - Each .xlsx file yields two .jpg files:
        * <filename>_individual_spectra.jpg
        * <filename>_average_spectrum.jpg
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline  # for spline smoothing

def plot_hyperspectral_data(directory_path):
    """
    Searches a specified directory (and its subfolders) for .xlsx files starting 
    with "Hyperspectral_Results". For each valid file, reads the "Normalized Data" 
    sheet and creates two figures:

    1) A multi-line plot of each lipid feature (rows), after removing zero-valued 
       points and spline-smoothing the remaining data.
    2) A single-line plot of the average of all valid spectra, again spline-smoothed.

    Both plots are also saved as .jpg images in a local subfolder named "Plots".
    
    Parameters:
    -----------
    directory_path : str
        The absolute or relative path to the directory that should be searched.
    """

    # Walk through the directory, including all subfolders
    for root, dirs, files in os.walk(directory_path):
        for file_name in files:
            # Only consider .xlsx files that start with 'Hyperspectral_Results'
            if file_name.startswith("Hyperspectral_Results") and file_name.endswith(".xlsx"):
                file_path = os.path.join(root, file_name)
                
                # 1) Create a subfolder named 'Plots' in the same directory to save images
                plots_folder = os.path.join(root, "Plots")
                os.makedirs(plots_folder, exist_ok=True)

                # 2) Read the "Normalized Data" sheet
                df = pd.read_excel(file_path, sheet_name="Normalized Data", header=None)

                # ----------------------------------------------------
                # Data Extraction
                # ----------------------------------------------------
                # The first row (index 0) is your x-axis (wavenumbers)
                # from the second column onward, assuming the first column 
                # is a label or empty cell.
                x = df.iloc[0, 1:].values

                # The subsequent rows are the normalized hyperspectral data.
                # Also skip the first column in each row.
                data = df.iloc[1:, 1:].values  # shape: (#features, #wavenumbers)

                # Prepare an array to accumulate each spline-evaluated spectrum 
                # for computing the average
                # We'll define a common dense grid (300 points) for smoothing 
                # across all lipid features:
                x_smooth_global = np.linspace(x.min(), x.max(), 300)
                all_spectra_smoothed = []  # will store each row's y_smooth

                # ----------------------------------------------------
                # 1) Multi-line plot (individual spectra)
                # ----------------------------------------------------
                fig1 = plt.figure()  # new figure for individual spectra
                plt.title(file_name)
                plt.xlabel('Wavenumber (cm⁻¹)')
                plt.ylabel('Normalized Intensity')

                for i in range(data.shape[0]):
                    # Extract the current row (lipid feature)
                    y = data[i, :].astype(float)

                    # Remove any points where y == 0
                    mask = (y != 0)
                    x_valid = x[mask]
                    y_valid = y[mask]

                    # If fewer than 2 points remain, skip this row
                    if len(x_valid) < 2:
                        continue

                    # Create a spline for the valid points
                    spline = make_interp_spline(x_valid, y_valid, k=3)  # cubic spline
                    y_smooth = spline(x_smooth_global)

                    # Plot the smoothed curve for the individual spectrum
                    plt.plot(x_smooth_global, y_smooth)

                    # Store this row's smoothed curve for the average calculation
                    all_spectra_smoothed.append(y_smooth)

                # Remove the legend as requested (do nothing here since we never called plt.legend())
                # Save this figure as a .jpg in the "Plots" subfolder
                individual_filename = os.path.join(
                    plots_folder, 
                    f"{os.path.splitext(file_name)[0]}_individual_spectra.jpg"
                )
                fig1.savefig(individual_filename, dpi=300, bbox_inches='tight')
                plt.show()

                # ----------------------------------------------------
                # 2) Average Spectrum Plot
                # ----------------------------------------------------
                fig2 = plt.figure()  # new figure for the average spectrum
                plt.title(f"{file_name} (Average Spectrum)")
                plt.xlabel('Wavenumber (cm⁻¹)')
                plt.ylabel('Normalized Intensity')

                # Only compute and plot the average if we have at least one valid spectrum
                if len(all_spectra_smoothed) > 0:
                    # Convert to NumPy array for easy averaging: shape (n_spectra, 300)
                    all_spectra_smoothed = np.array(all_spectra_smoothed)

                    # Compute the pointwise average across rows
                    y_avg = np.mean(all_spectra_smoothed, axis=0)

                    # Plot the single average line
                    plt.plot(x_smooth_global, y_avg)

                # Save the average spectrum figure
                average_filename = os.path.join(
                    plots_folder, 
                    f"{os.path.splitext(file_name)[0]}_average_spectrum.jpg"
                )
                fig2.savefig(average_filename, dpi=300, bbox_inches='tight')
                plt.show()


# -------------------------------------------------------------------------
# Example usage (uncomment and set the correct directory path to run):
# -------------------------------------------------------------------------
if __name__ == "__main__":
    directory_to_search = r"C:\Users\clchr\OneDrive - Stanford\Research Documents\AD Project\2025"
    plot_hyperspectral_data(directory_to_search)

# Execution notes:
#  - For each "Hyperspectral_Results_*.xlsx" file, this script will:
#      * Show a figure with multiple individual spectra lines,
#      * Show a figure with the single average spectrum,
#      * Save both figures as ".jpg" images in a subfolder "Plots".
