config = {
    "file_keywords": {
        "magnification_keyword": "100X",
        "fluorescence_markers": ["DAPI", "Ki67", "Phalloidin"],
        "cars_keyword": "CARS2850",
        "stacks_keywords": ["OC8"],
        "hyperspectral_keyword": "Spectrum"
    },
    "reference_mosaic": {
        "nrows": 3,
        "ncols": 3,
        "overlap_percent": 5.0
    },
    "channel_map": {
        "DAPI": 0,
        "Ki67": 1,
        "Phalloidin": 2
    },
    "cell_markers": ["Phalloidin"],
    "marker_thresholds": {
        "Phalloidin": {
            "threshold_method": "triangle",
            "offset": 0.725
        }
    },
    "morphology_params": {
        "fluorescence_params": {
            "cell_size": 10000,
            "min_size": 2000,
            "closing_radius": 14,
            "gaussian_sigma": 2,
            "fill_holes": True,
            "threshold_method": "otsu",  # fallback
            "offset": 1.0               # fallback
        },
        "nuclei_params": {
            "cell_size": 5000,
            "min_size": 1000,
            "closing_radius": 3,
            "gaussian_sigma": 2,
            "fill_holes": True,
            "threshold_method": "triangle",
            "offset": 1.00
        },
        "foci_params": {
            "sigma": 0,
            "min_distance": 15,
            "min_size": 35,
            "std_dev_multiplier": 3,
            "remove_saturated": True,
            "saturation_threshold": 2500,
            "saturated_min_size": 5000
        },
        "foci_params_hyperspectral": {
            "sigma": 2,
            "min_distance": 15,
            "min_size": 35,
            "std_dev_multiplier": 3,
            "remove_saturated": False,
            "saturation_threshold": 2500,
            "saturated_min_size": 5000
        }
    },
    "colormaps": {
        "DAPI": (0, 0, 255),  # blue
        "Ki67": (0, 255, 0), # green
        "Phalloidin": (255, 0, 0),  # red
        "DEFAULT": (255, 255, 255)  # fallback color (white)
    },
    "paths": {
        "data_directory": r""
    },
    "stack_offset": {
        "DAPI": 0,
        "Ki67": 0,
        "Phalloidin": 0
    },
    "cell_marker_map": {
        # The key should match the *label* after "Stacks"
        # e.g. if you have "StacksMicroglia" => label is "Microglia".
        # So "Microglia" => ["IBA1"], "Astrocytes" => ["GFAP"]
        "OC8" : ["Phalloidin"]
    }
}