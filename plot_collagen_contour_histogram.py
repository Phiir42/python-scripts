import os
import glob
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

def load_contour_lengths(mat_folder):
    """
    Search for all .mat files in 'mat_folder', load 'L_contour' from each,
    and concatenate them into a single 1D numpy array.

    Parameters
    ----------
    mat_folder : str
        Path to the folder containing the .mat files.

    Returns
    -------
    all_lengths : np.ndarray
        1D array of all concatenated contour lengths. Empty if no data found.
    """
    all_lengths_list = []
    mat_pattern = os.path.join(mat_folder, '*.mat')
    mat_files = glob.glob(mat_pattern)

    if not mat_files:
        print(f"No .mat files found in folder: {mat_folder}")
        return np.array([])

    for mat_path in mat_files:
        try:
            mat_data = scipy.io.loadmat(mat_path)
        except Exception as e:
            print(f"Error loading {mat_path}: {e}")
            continue

        if 'L_contour' in mat_data:
            # Squeeze in case it's shape (1, N) or (N, 1)
            lengths = np.squeeze(mat_data['L_contour'])
            lengths = np.atleast_1d(lengths).astype(float)
            all_lengths_list.append(lengths)
        else:
            print(f"Warning: 'L_contour' not found in {os.path.basename(mat_path)}")

    if all_lengths_list:
        all_lengths = np.concatenate(all_lengths_list)
    else:
        all_lengths = np.array([])

    return all_lengths

def plot_publication_histogram(lengths, bins=50):
    """
    Plot a Prism-like, publication-quality histogram of the input contour lengths.

    Parameters
    ----------
    lengths : np.ndarray
        1D array of contour-length values.
    bins : int, optional
        Number of histogram bins (default is 50).
    """
    # ----------------------------
    # 1) Global rcParams tweaks
    # ----------------------------
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],       # Prism typically uses a clean sans-serif
        'font.size': 14,                     # Base font size (e.g. 14 pt)
        'axes.titlesize': 16,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'axes.linewidth': 1.5,              # Thicker axis lines
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 6,
        'ytick.major.size': 6,
        'xtick.minor.visible': False,
        'ytick.minor.visible': False,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'grid.color': '#CCCCCC',            # Light gray grid
        'grid.linestyle': '--',
        'grid.linewidth': 0.8,
    })

    # ----------------------------
    # 2) Create the figure
    # ----------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # ----------------------------
    # 3) Compute & draw the histogram
    # ----------------------------
    # Use a light gray fill (or white) with a solid black edge on each bar
    n, bins_edges, patches = ax.hist(
        lengths,
        bins=bins,
        color='#d0cc93',            # bar face color (white or very light gray)
        edgecolor='black',
        linewidth=1.2
    )

    # ----------------------------
    # 4) Tweak the spines (only left & bottom)
    # ----------------------------
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)

    # ----------------------------
    # 5) Axis labels, title, and grid
    # ----------------------------
    ax.set_xlabel('Contour Length (µm)')
    ax.set_ylabel('Frequency')

    # Optional: turn on horizontal grid lines only
    ax.xaxis.grid(False)
    ax.xaxis.grid(False)

    # ----------------------------
    # 6) Tight layout for publication
    # ----------------------------
    plt.tight_layout()

    # ----------------------------
    # 7) Show
    # ----------------------------
    plt.show()
    
    
def compute_average_length(lengths):
    """
    Return the arithmetic mean of all contour-length values.

    Parameters
    ----------
    lengths : np.ndarray
        1D array of contour lengths.

    Returns
    -------
    float
        Mean contour length.
    """
    return float(np.mean(lengths))


if __name__ == '__main__':
    # --------------------------------------------------------------------------
    # NOTE: Change this path to the folder where your 27 .mat files reside.
    # If they're in your current working directory in Spyder, you can use '.'.
    # --------------------------------------------------------------------------
    mat_folder = r'C:\Users\clchr\Downloads\collagen_fibril_histogram_data'  # e.g. r'C:\Users\YourName\Documents\collagen_data'

    # Load & concatenate
    all_contour_lengths = load_contour_lengths(mat_folder)

    if all_contour_lengths.size == 0:
        print("No contour-length data to plot. Exiting.")
    else:
        # Compute & report average
        avg_len = compute_average_length(all_contour_lengths)
        print(f"Average contour length = {avg_len:.2f} µm")
        
        # You can adjust bins if needed
        plot_publication_histogram(all_contour_lengths, bins=100)