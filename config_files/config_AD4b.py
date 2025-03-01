config = {
    "file_keywords": {
        "magnification_keyword": "100X",
        "fluorescence_markers": ["DAPI", "IBA1", "MAP2_Sigma", "TUJ_Ck", "TUJ_Ms", "LAMP2"],
        "cars_keyword": "CARS2850",
        "stacks_keywords": ["Neurons"],
        "hyperspectral_keyword": "Spectrum"
    },
    "channel_map": {
        "DAPI": 0,
        "IBA1": 1,
        "MAP2_Sigma": 1,
        "Autofluorescence": 2,
        "TUJ_Ck": 3,
        "TUJ_Ms": 3,
        "LAMP2": 3
    },
    "cell_markers": ["IBA1", "MAP2_Sigma", "TUJ_Ck", "TUJ_Ms"],
    "marker_thresholds": {
        "IBA1": {
            "threshold_method": "triangle",
            "offset": 0.95
        },
        "MAP2_Sigma": {
            "threshold_method": "triangle",
            "offset": 0.85
        },
        "TUJ_Ck": {
            "threshold_method": "triangle",
            "offset": 0.85
        },
        "TUJ_Ms": {
            "threshold_method": "triangle",
            "offset": 0.85
        }
    },
    "morphology_params": {
        "fluorescence_params": {
            "min_size": 10000,
            "closing_radius": 8,
            "gaussian_sigma": 2,
            "fill_holes": True,
            "threshold_method": "otsu",  # fallback
            "offset": 1.0               # fallback
        },
        "nuclei_params": {
            "min_size": 5000,
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
        "IBA1": (0, 255, 0), # green
        "MAP2_Sigma": (0, 255, 0),
        "Autofluorescence": (255, 0, 0),  # red
        "TUJ_Ck": (255, 0, 255),  # magenta
        "TUJ_Ms": (255, 0, 255),
        "LAMP2": (255, 0, 255),
        "DEFAULT": (255, 255, 255)  # fallback color (white)
    },
    "paths": {
        "data_directory": r"C:\Users\clchr\Downloads\AD4b",
    },
    "stack_offset": {
        "DAPI": 0,
        "IBA1": 0,
        "MAP2_Sigma": 0,
        "TUJ_Ck": 0,
        "TUJ_Ms": 0,
        "LAMP2": 0
    },
    "cell_marker_map": {
        # The key should match the *label* after "Stacks"
        # e.g. if you have "StacksMicroglia" => label is "Microglia".
        # So "Microglia" => ["IBA1"], "Astrocytes" => ["GFAP"]
        "Neurons": ["IBA1", "MAP2_Sigma", "TUJ_Ck", "TUJ_Ms"]
    }
}

