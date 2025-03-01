config = {
    "file_keywords": {
        "magnification_keyword": "100X",
        "fluorescence_markers": ["DAPI", "aSMA", "Calretinin", "Ki67"],
        "cars_keyword": "CARS2850",
        "stacks_keywords": ["Stacks1", "Stacks2", "Stacks3", "Stacks4"],
        "hyperspectral_keyword": "SpectraCH"
    },
    "channel_map": {
        "DAPI": 0,
        "aSMA": 1,
        "Calretinin": 3,
        "Ki67": 3
    },
    "cell_markers": [],
    "morphology_params": {
        "fluorescence_params": {
            "min_size": 1000,
            "closing_radius": 10,
            "gaussian_sigma": 3,
            "fill_holes": True
        },
        "foci_params": {
            "sigma": 2,
            "min_distance": 15,
            "min_size": 35,
            "std_dev_multiplier": 3
        }
    },
    "colormaps": {
        "DAPI": (0, 0, 255),  # blue
        "aSMA": (0, 255, 0),  # green
        "Calretinin": (255, 0, 255),  # magenta
        "Ki67": (255, 0, 255),  # magenta
        "DEFAULT": (255, 255, 255)  # fallback color (white)
    },
    "paths": {
        "data_directory": r"C:\Users\clchr\Downloads\OV5",
    },
    "stack_offset": {
        "Calretinin": 0,
        "Ki67": 2
    }
}
