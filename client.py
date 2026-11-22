import os
import argparse
import flwr as fl
import tensorflow as tf
from model import build_model
from config import (
    DATASET_PATH, CLASS_NAMES, IMG_SIZE,
    BATCH_SIZE, SEED, training_hospitals,
    positive_only_hospitals, CENTRAL_SERVER_ADDRESS
)


def load_data(hospital_name):
    if hospital_name in positive_only_hospitals:
        return load_positive_only_data(hospital_name)

    path = os.path.join(DATASET_PATH, hospital_name)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        labels           = "inferred",
        label_mode       = "binary",
        class_names      = CLASS_NAMES,
        validation_split = 0.2,
        subset           = "training",
        seed             = SEED,
        image_size       = IMG_SIZE,
        batch_size       = BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        path,
        labels           = "inferred",
        label_mode       = "binary",
        class_names      = CLASS_NAMES,
        validation_split = 0.2,
        subset           = "validation",
        seed             = SEED,
        image_size       = IMG_SIZE,
        batch_size       = BATCH_SIZE
    )
    return train_ds, val_ds


def load_positive_only_data(hospital_name):
    """Flat, unlabelled folder in which every image is RD.

    Same split/resize/batching as load_data; the label (RD = 1) is assigned
    programmatically, in the same (batch, 1) float32 format as label_mode="binary".
    """
    path  = os.path.join(DATASET_PATH, hospital_name)
    label = float(CLASS_NAMES.index('RD'))

    def add_label(images):
        return images, tf.fill([tf.shape(images)[0], 1], label)

    splits = []
    for subset in ("training", "validation"):
        ds = tf.keras.utils.image_dataset_from_directory(
            path,
            labels           = None,
            validation_split = 0.2,
            subset           = subset,
            seed             = SEED,
            image_size       = IMG_SIZE,
            batch_size       = BATCH_SIZE
        )
        file_paths    = ds.file_paths
        ds            = ds.map(add_label)
        ds.file_paths = file_paths
        splits.append(ds)
    return tuple(splits)


class HospitalClient(fl.client.NumPyClient):

    def __init__(self, hospital_name):
        self.hospital_name         = hospital_name
        # Only RD labels: AUC is undefined and accuracy equals sensitivity
        self.positive_only         = hospital_name in positive_only_hospitals
        self.model                 = build_model()
        self.train_ds, self.val_ds = load_data(hospital_name)
        print(f"[CLIENT] {hospital_name} — initialized")

    def get_parameters(self, config):
        print(f"[CLIENT] {self.hospital_name} — sending weights to server")
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        print(f"[CLIENT] {self.hospital_name} — local training started")

        history = self.model.fit(
            self.train_ds,
            epochs          = 5,
            validation_data = self.val_ds,
            verbose         = 1
        )

        val_auc = history.history['val_auc'][-1]
        val_acc = history.history['val_accuracy'][-1]
        print(f"[CLIENT] {self.hospital_name} — done")
        metrics = {"hospital": self.hospital_name, "val_accuracy": float(val_acc)}
        if self.positive_only:
            print(f"         Val Sensitivity: {val_acc:.4f} (positive-only site, AUC undefined)")
            metrics["val_sensitivity"] = float(val_acc)
        else:
            print(f"         Val AUC: {val_auc:.4f} | Val Acc: {val_acc:.4f}")
            metrics["val_auc"] = float(val_auc)
        print(f"         Uploading weights to server...")

        return self.model.get_weights(), len(self.train_ds), metrics

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy, auc = self.model.evaluate(self.val_ds, verbose=0)
        if self.positive_only:
            print(f"[CLIENT] {self.hospital_name} — eval | "
                  f"Sensitivity: {accuracy:.4f} (positive-only site)")
            return loss, len(self.val_ds), {
                "accuracy"   : float(accuracy),
                "sensitivity": float(accuracy)
            }
        print(f"[CLIENT] {self.hospital_name} — eval | "
              f"AUC: {auc:.4f} | Acc: {accuracy:.4f}")
        return loss, len(self.val_ds), {
            "accuracy": float(accuracy),
            "auc"     : float(auc)
        }


def client_fn(cid: str):
    hospital = training_hospitals[int(cid)]
    return HospitalClient(hospital_name=hospital)


def start_hospital_node(hospital_name, server_address=CENTRAL_SERVER_ADDRESS):
    """Run one training hospital as an independent process (gRPC mode)."""
    if hospital_name not in training_hospitals:
        raise ValueError(f"'{hospital_name}' is not a training hospital: {training_hospitals}")
    print(f"[NODE] {hospital_name} — connecting to {server_address}")
    fl.client.start_numpy_client(
        server_address = server_address,
        client         = HospitalClient(hospital_name=hospital_name)
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a hospital FL node")
    parser.add_argument("--hospital", required=True, choices=training_hospitals)
    parser.add_argument("--server-address", default=CENTRAL_SERVER_ADDRESS)
    args = parser.parse_args()
    start_hospital_node(args.hospital, args.server_address)
