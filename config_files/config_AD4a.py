config = {
    "file_keywords": {
        "magnification_keyword": "100X",
        "fluorescence_markers": ["DAPI", "IBA1", "TUJ", "Autofluorescence"],
        "cars_keyword": "CARS2850",
        "stacks_keywords": ["Microglia"],
        "hyperspectral_keyword": "Spectrum"
    },
    "channel_map": {
        "DAPI": 0,
        "IBA1": 1,
        "Autofluorescence": 2,
        "TUJ": 3
    },
    "cell_markers": ["IBA1"],
    "marker_thresholds": {
        "IBA1": {
            "threshold_method": "triangle",
            "offset": 0.9
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
            "offset": 1.0,
            "bad_slice_frac_threshold": 0.55,
            "bad_slice_max_components": None,
            "bad_slice_use_mip_if_fraction_over": 0.34,
            "clip_to_mip_mask": True
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
            "sigma": 2,
            "min_distance": 15,
            "min_size": 35,
            "std_dev_multiplier": 3,
            "remove_saturated": True,
            "saturation_threshold": 2500,
            "saturated_min_size": 5000
        }
    },
    "colormaps": {
        "DAPI": (0, 0, 255),  # blue
        "IBA1": (0, 255, 0),
        "Autofluorescence": (255, 0, 0),  # red
        "TUJ": (255, 0, 255),  # magenta
        "DEFAULT": (255, 255, 255)  # fallback color (white)
    },
    "paths": {
        "data_directory": r"D:\OneDrive - Stanford\Research Documents\AD Project\2025\AD4a"
    },
    "stack_offset": {
        "DAPI": 0,
        "IBA1": 0,
        "TUJ": 0
    },
    "cell_marker_map": {
        # The key should match the *label* after "Stacks"
        # e.g. if you have "StacksMicroglia" => label is "Microglia".
        # So "Microglia" => ["IBA1"], "Astrocytes" => ["GFAP"]
        "Microglia": ["IBA1"]
    }
}
# Folder-name tokens → stacks label + allowed markers (priority left→right)
config["hyperspectral_folder_map"] = {
    "astrocyte": {
        "label": "Astrocytes",
        "markers": ["GFAP"],              # add GFAP to channel_map if you use it
    },
    "microglia": {
        "label": "Microglia",
        "markers": ["IBA1"],
    },
    "neuron": {
        "label": "Neurons",
        "markers": ["TUJ", "TUJ_Ck", "TUJ_Ms", "MAP2_Sigma"],  # <-- your 3 neuron markers
    },
}

