# CARS & Confocal Analysis Pipeline

## lipid_analysis.py  
- Entry point: `main()`  
- Key functions: `load_images()`, `segment_cells()`, `quantify_lipids()`, `export_to_excel()`  

## myelin_analysis.py  
- Helper functions: `detect_myelin()`, `compute_myelin_mask_area()`  

## config_files/config_ADxx.py  
- Contains per‐experiment parameters (thresholds, file paths, ROI settings)
