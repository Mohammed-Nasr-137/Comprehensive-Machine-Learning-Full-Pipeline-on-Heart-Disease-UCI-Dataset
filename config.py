"""
Configuration file for the Heart Disease ML Project.
Contains all path configurations and project settings.
"""

import os
from pathlib import Path

# Get the project root directory (where this config.py file is located)
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
UI_DIR = PROJECT_ROOT / "ui"

# Specific file paths
CLEANED_DATA_PATH = DATA_DIR / "cleaned_data.csv"
COMBINED_UCI_DATA_PATH = DATA_DIR / "combined_uci_heart.data"
PCA_13_COMPONENTS_PATH = DATA_DIR / "pca_13_components.csv"
PCA_9_COMPONENTS_PATH = DATA_DIR / "pca_9_components.csv"
PROCESSED_CLEVELAND_PATH = DATA_DIR / "processed.cleveland.data"
PROCESSED_HUNGARIAN_PATH = DATA_DIR / "processed.hungarian.data"
PROCESSED_SWITZERLAND_PATH = DATA_DIR / "processed.switzerland.data"
REDUCED_CLEANED_DATA_PATH = DATA_DIR / "reduced_cleaned_data.csv"
RF_SELECTED_DATA_PATH = DATA_DIR / "rf_selected_data.csv"
UNION_CLEANED_DATA_PATH = DATA_DIR / "union_cleaned_data.csv"

# Model paths
FINAL_VOTING_MODEL_PATH = MODELS_DIR / "final_voting_model3.pkl"
FINAL_VOTING_MODEL2_PATH = MODELS_DIR / "final_voting_model2.pkl"
FINAL_VOTING_MODEL3_PATH = MODELS_DIR / "final_voting_model3.pkl"
FINAL_VOTING_MODEL6_PATH = MODELS_DIR / "final_voting_model6.pkl"
FINAL_VOTING_MODEL7_PATH = MODELS_DIR / "final_voting_model7.pkl"

# UI specific paths
EXPECTED_FEATURES_PATH = UI_DIR / "expected_features.txt"

# Create directories if they don't exist
def create_directories():
    """Create project directories if they don't exist."""
    directories = [DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR, UI_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

# Utility function to get relative path from project root
def get_relative_path(file_path):
    """Convert absolute path to relative path from project root."""
    return os.path.relpath(file_path, PROJECT_ROOT)

# Utility function to resolve path (handles both absolute and relative paths)
def resolve_path(path):
    """Resolve path relative to project root if it's not absolute."""
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
