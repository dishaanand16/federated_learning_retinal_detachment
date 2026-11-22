"""SRM integration checks.  Run from the project root:

    flenv\\Scripts\\python.exe -m unittest discover -s tests -v
"""
import os
import sys
import hashlib
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np
import tensorflow as tf
import config
from client import load_data, client_fn
from srm_config import SRM_HOSPITAL, SRM_LABEL, SRM_DATA_DIR

IMAGE_EXTS = (".bmp", ".gif", ".jpeg", ".jpg", ".png")


def _labels(ds):
    return np.concatenate([y.numpy() for _, y in ds]).ravel()


def _md5(path):
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


class TestSRMDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_ds, cls.val_ds = load_data(SRM_HOSPITAL)
        cls.ref_train, _ = load_data(config.centralized_hospitals[0])

    def test_class_mapping(self):
        self.assertEqual(config.CLASS_NAMES, ['Non-RD', 'RD'])
        self.assertEqual(SRM_LABEL, 1)

    def test_all_files_loaded_once(self):
        on_disk = {f for f in os.listdir(SRM_DATA_DIR) if f.lower().endswith(IMAGE_EXTS)}
        train   = {os.path.basename(p) for p in self.train_ds.file_paths}
        val     = {os.path.basename(p) for p in self.val_ds.file_paths}
        self.assertFalse(train & val)
        self.assertEqual(train | val, on_disk)

    def test_every_sample_is_positive(self):
        for ds in (self.train_ds, self.val_ds):
            labels = _labels(ds)
            self.assertEqual(len(labels), len(ds.file_paths))
            self.assertTrue(np.all(labels == 1.0))

    def test_tensor_spec_matches_existing_hospitals(self):
        self.assertEqual(self.train_ds.element_spec, self.ref_train.element_spec)
        x, y = next(iter(self.train_ds))
        self.assertEqual(tuple(x.shape[1:]), (*config.IMG_SIZE, 3))
        self.assertEqual(x.dtype, tf.float32)
        self.assertEqual(y.shape[1:], (1,))
        self.assertGreaterEqual(float(tf.reduce_min(x)), 0.0)    # raw 0-255 pixels, as
        self.assertLessEqual(float(tf.reduce_max(x)), 255.0)     # EfficientNet expects


class TestFLRegistration(unittest.TestCase):

    def test_srm_is_training_client_only(self):
        self.assertIn(SRM_HOSPITAL, config.training_hospitals)
        self.assertIn(SRM_HOSPITAL, config.positive_only_hospitals)
        self.assertNotEqual(SRM_HOSPITAL, config.federated_hospital)
        self.assertEqual(config.training_hospitals[:3], config.centralized_hospitals)

    def test_client_fn_maps_cids(self):
        clients = [client_fn(str(i)) for i in range(len(config.training_hospitals))]
        self.assertEqual([c.hospital_name for c in clients], config.training_hospitals)
        self.assertTrue(clients[-1].positive_only)
        self.assertFalse(any(c.positive_only for c in clients[:-1]))


class TestTestingHospitalUnchanged(unittest.TestCase):

    def test_config_unchanged(self):
        self.assertEqual(config.federated_hospital, 'Aravid_eye_care')
        self.assertEqual(config.centralized_hospitals, ['LV_prasad', 'SMCH', 'Medanta'])
        self.assertNotIn(config.federated_hospital, config.training_hospitals)

    def test_split_identical_to_original_pipeline(self):
        # Same call the original client.load_data made for the test hospital
        original = tf.keras.utils.image_dataset_from_directory(
            os.path.join(config.DATASET_PATH, config.federated_hospital),
            labels="inferred", label_mode="binary", class_names=config.CLASS_NAMES,
            validation_split=0.2, subset="validation", seed=config.SEED,
            image_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE)
        current = load_data(config.federated_hospital)[1]
        self.assertEqual(sorted(current.file_paths), sorted(original.file_paths))
        self.assertEqual(len(current.file_paths), 80)
        labels = _labels(current)
        self.assertEqual(int((labels == 0).sum()), 19)
        self.assertEqual(int((labels == 1).sum()), 61)

    def test_no_srm_image_in_test_hospital(self):
        test_dir  = os.path.join(config.DATASET_PATH, config.federated_hospital)
        test_hash = {_md5(os.path.join(dp, f))
                     for dp, _, fs in os.walk(test_dir) for f in fs}
        srm_hash  = {_md5(os.path.join(SRM_DATA_DIR, f)) for f in os.listdir(SRM_DATA_DIR)}
        self.assertFalse(srm_hash & test_hash)


if __name__ == "__main__":
    unittest.main()
