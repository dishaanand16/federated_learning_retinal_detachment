"""Configuration for the independent SRM site node (srm_server.py)."""
import os
from config import DATASET_PATH, CLASS_NAMES, CENTRAL_SERVER_ADDRESS, srm_hospital


SRM_HOSPITAL       = srm_hospital
SRM_DATA_DIR       = os.path.join(DATASET_PATH, SRM_HOSPITAL)
SRM_LABEL          = CLASS_NAMES.index('RD')      # every SRM image is RD -> 1
SRM_SERVER_ADDRESS = os.environ.get("SRM_SERVER_ADDRESS", CENTRAL_SERVER_ADDRESS)
