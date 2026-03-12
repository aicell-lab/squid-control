"""
Simulation sample presets for the virtual microscope.

Each entry maps a canonical sample name to its microscope configuration and zarr dataset path.
Used by MicroscopeHyphaService.switch_sample() and list_simulation_samples().
"""

# HPA channel layout (Opera Phoenix 63x — no brightfield)
_HPA_CHANNELS = [
    "Fluorescence_405_nm_Ex",   # DAPI (nucleus)
    "Fluorescence_488_nm_Ex",   # Antibody target (protein of interest)
    "Fluorescence_561_nm_Ex",   # Microtubules / tubulin
    "Fluorescence_638_nm_Ex",   # ER / mitochondria
]

# HPA full-scan channel layout (20x Squid — no brightfield)
_HPA_FULL_SCAN_CHANNELS = [
    "Fluorescence_405_nm_Ex",   # DAPI (nucleus)
    "Fluorescence_488_nm_Ex",   # Antibody target (protein of interest)
    "Fluorescence_561_nm_Ex",   # Microtubules / tubulin
    "Fluorescence_638_nm_Ex",   # ER / mitochondria
    "Fluorescence_730_nm_Ex",   # Additional far-red marker
]

SIMULATION_SAMPLES = {
    "U2OS_FUCCI": {
        "config_name": "HCS_v2",
        "zarr_dataset_path": "/mnt/shared_documents/20251215-illumination-calibrated/data.zarr",
        "description": "U2OS osteosarcoma cells with FUCCI cell-cycle reporters (20x). Cdt1-mKO2=G1 (red), Geminin-mAG=S/G2/M (green).",
        "cell_line": "U2OS (Human osteosarcoma)",
        "staining": "FUCCI live reporter (Cdt1-mKO2 / Geminin-mAG)",
        "objective": "20x",
        "scan_type": "full scan",
        "channels": [
            "BF_LED_matrix_full",
            "Fluorescence_488_nm_Ex",
            "Fluorescence_561_nm_Ex",
        ],
    },
    "HPA_FULL_SCAN": {
        "config_name": "HCS_v2",
        "zarr_dataset_path": "/mnt/shared_documents/hpa-full-scan-2026-03-10-17-48/data.zarr",
        "description": "[Recommended] Human Protein Atlas full plate scan (2026-03-10) — 20x. Full-well coverage with DAPI, antibody target, microtubules, ER markers, and far-red channel. Other HPA plates only have images at the well centre.",
        "cell_line": "Various human cell lines (Human Protein Atlas)",
        "staining": "Immunofluorescence (subcellular protein localisation)",
        "objective": "20x",
        "scan_type": "full scan",
        "channels": _HPA_FULL_SCAN_CHANNELS,
    },
    "HPA_PLATE1": {
        "config_name": "HCS_v2_63x",
        "zarr_dataset_path": "/mnt/shared_documents/hpa_plate1/data.zarr",
        "description": "Human Protein Atlas plate 1 — Opera Phoenix 63x. DAPI, antibody target, microtubules, ER markers.",
        "cell_line": "Various human cell lines (Human Protein Atlas)",
        "staining": "Immunofluorescence (subcellular protein localisation)",
        "objective": "63x",
        "scan_type": "centre only",
        "channels": _HPA_CHANNELS,
    },
    "HPA_PLATE2": {
        "config_name": "HCS_v2_63x",
        "zarr_dataset_path": "/mnt/shared_documents/hpa_plate2/data.zarr",
        "description": "Human Protein Atlas plate 2 — Opera Phoenix 63x. DAPI, antibody target, microtubules, ER markers.",
        "cell_line": "Various human cell lines (Human Protein Atlas)",
        "staining": "Immunofluorescence (subcellular protein localisation)",
        "objective": "63x",
        "scan_type": "centre only",
        "channels": _HPA_CHANNELS,
    },
    "HPA_PLATE3": {
        "config_name": "HCS_v2_63x",
        "zarr_dataset_path": "/mnt/shared_documents/hpa_plate3/data.zarr",
        "description": "Human Protein Atlas plate 3 — Opera Phoenix 63x. DAPI, antibody target, microtubules, ER markers.",
        "cell_line": "Various human cell lines (Human Protein Atlas)",
        "staining": "Immunofluorescence (subcellular protein localisation)",
        "objective": "63x",
        "scan_type": "centre only",
        "channels": _HPA_CHANNELS,
    },
    "HPA_PLATE4": {
        "config_name": "HCS_v2_63x",
        "zarr_dataset_path": "/mnt/shared_documents/hpa_plate4/data.zarr",
        "description": "Human Protein Atlas plate 4 — Opera Phoenix 63x. DAPI, antibody target, microtubules, ER markers.",
        "cell_line": "Various human cell lines (Human Protein Atlas)",
        "staining": "Immunofluorescence (subcellular protein localisation)",
        "objective": "63x",
        "scan_type": "centre only",
        "channels": _HPA_CHANNELS,
    },
    "HPA_PLATE5": {
        "config_name": "HCS_v2_63x",
        "zarr_dataset_path": "/mnt/shared_documents/hpa_plate5/data.zarr",
        "description": "Human Protein Atlas plate 5 — Opera Phoenix 63x. DAPI, antibody target, microtubules, ER markers.",
        "cell_line": "Various human cell lines (Human Protein Atlas)",
        "staining": "Immunofluorescence (subcellular protein localisation)",
        "objective": "63x",
        "scan_type": "centre only",
        "channels": _HPA_CHANNELS,
    },
    "HPA_PLATE6": {
        "config_name": "HCS_v2_63x",
        "zarr_dataset_path": "/mnt/shared_documents/hpa_plate6/data.zarr",
        "description": "Human Protein Atlas plate 6 — Opera Phoenix 63x. DAPI, antibody target, microtubules, ER markers.",
        "cell_line": "Various human cell lines (Human Protein Atlas)",
        "staining": "Immunofluorescence (subcellular protein localisation)",
        "objective": "63x",
        "scan_type": "centre only",
        "channels": _HPA_CHANNELS,
    },
    "HPA_PLATE7": {
        "config_name": "HCS_v2_63x",
        "zarr_dataset_path": "/mnt/shared_documents/hpa_plate7/data.zarr",
        "description": "Human Protein Atlas plate 7 — Opera Phoenix 63x. DAPI, antibody target, microtubules, ER markers.",
        "cell_line": "Various human cell lines (Human Protein Atlas)",
        "staining": "Immunofluorescence (subcellular protein localisation)",
        "objective": "63x",
        "scan_type": "centre only",
        "channels": _HPA_CHANNELS,
    },
}

# Case-insensitive aliases accepted by switch_sample (keys must be upper-cased + underscored)
SAMPLE_ALIASES = {
    "HCS_V2": "U2OS_FUCCI",
    "DEFAULT": "U2OS_FUCCI",
    "20X": "U2OS_FUCCI",
    "U2OS": "U2OS_FUCCI",
    "FUCCI": "U2OS_FUCCI",
    "HCS_V2_63X": "HPA_FULL_SCAN",
    "OPERA": "HPA_FULL_SCAN",
    "63X": "HPA_FULL_SCAN",
    "HPA": "HPA_FULL_SCAN",
    "HUMAN_PROTEIN_ATLAS": "HPA_FULL_SCAN",
}
