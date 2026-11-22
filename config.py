import os


DATASET_PATH          = r"Datasets_various hospital"
IMG_SIZE              = (224, 224)
BATCH_SIZE            = 16
SEED                  = 42
LR                    = 1e-4
CLASS_NAMES           = ['Non-RD', 'RD']
NUM_ROUNDS            = 5
hospital_names        = ['LV_prasad', 'SMCH', 'Medanta', 'Aravid_eye_care']
centralized_hospitals = ['LV_prasad', 'SMCH', 'Medanta']
federated_hospital    = 'Aravid_eye_care'

# SRM — unlabelled folder in which every image is RD (positive-only site).
# It is an extra TRAINING client only; it never replaces federated_hospital.
srm_hospital            = 'srm'
INCLUDE_SRM             = True
positive_only_hospitals = [srm_hospital]
training_hospitals      = centralized_hospitals + ([srm_hospital] if INCLUDE_SRM else [])

# Central aggregation server (gRPC deployment mode: `python server.py --grpc`)
SERVER_PORT            = int(os.environ.get("FL_SERVER_PORT", 8080))
CENTRAL_SERVER_ADDRESS = os.environ.get("FL_SERVER_ADDRESS", f"127.0.0.1:{SERVER_PORT}")
