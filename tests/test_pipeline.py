import os
import unittest
import numpy as np
from src.preprocessing.pipeline import run_preprocessing

class TestPreprocessingPipeline(unittest.TestCase):
    def test_pipeline_execution(self):
        data = run_preprocessing()
        self.assertIn("X_train", data)
        self.assertIn("X_test", data)
        self.assertEqual(data["X_train"].shape[1], 18)
        self.assertFalse(np.isnan(data["X_train"]).any())
        self.assertFalse(np.isnan(data["X_test"]).any())

if __name__ == "__main__":
    unittest.main()
