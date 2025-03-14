config = {
    "file_keywords": {
        "magnification_keyword": "100X",
        "fluorescence_markers": ["DAPI", "IBA1", "GFAP", "Autofluorescence"],
        "cars_keyword": "CARS2850",
        "stacks_keywords": ["Microglia", "Astrocytes"],
        "hyperspectral_keyword": "Spectrum"
    },
    "channel_map": {
        "DAPI": 0,
        "IBA1": 1,
        "Autofluorescence": 2,
        "GFAP": 3
    },
    "cell_markers": ["IBA1", "GFAP"],
    "marker_thresholds": {
        "IBA1": {
            "threshold_method": "triangle",
            "offset": 0.9
        },
        "GFAP": {
            "threshold_method": "triangle",
            "offset": 0.95
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
        },
        "autofluorescence_params": {
            "min_size": 35,
            "closing_radius": 1,
            "gaussian_sigma": 2,
            "fill_holes": True,
            "threshold_method": "triangle",
            "offset": 1.0
        }
    },
    "colormaps": {
        "DAPI": (0, 0, 255),  # blue
        "IBA1": (0, 255, 0), # green
        "Autofluorescence": (255, 0, 0),  # red
        "GFAP": (255, 0, 255),  # magenta
        "DEFAULT": (255, 255, 255)  # fallback color (white)
    },
    "paths": {
        "data_directory": r"C:\Users\clchr\OneDrive - Stanford\Research Documents\AD Project\2025\AD3a"
    },
    "stack_offset": {
        "DAPI": 0,
        "IBA1": 0,
        "GFAP": 0
    },
    "cell_marker_map": {
        # The key should match the *label* after "Stacks"
        # e.g. if you have "StacksMicroglia" => label is "Microglia".
        # So "Microglia" => ["IBA1"], "Astrocytes" => ["GFAP"]
        "Microglia": ["IBA1"],
        "Astrocytes": ["GFAP"]
    }
}
