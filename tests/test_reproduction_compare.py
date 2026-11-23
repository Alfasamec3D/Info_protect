"""The final replication audit must detect altered scientific outputs."""
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

spec=importlib.util.spec_from_file_location('compare_runs',Path(__file__).resolve().parents[1]/'report/compare_runs.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReplicationAuditTests(unittest.TestCase):
    def test_unused_nan_metadata_does_not_hide_changed_weights(self):
        with TemporaryDirectory() as folder:
            a=Path(folder)/'a';b=Path(folder)/'b';a.mkdir();b.mkdir()
            np.savez_compressed(a/'model.npz',constant=np.nan,weights=np.array([1.,2.]))
            np.savez_compressed(b/'model.npz',constant=np.nan,weights=np.array([1.,2.]))
            self.assertEqual(len(module.compare(a,b)),1)
            np.savez_compressed(b/'model.npz',constant=np.nan,weights=np.array([1.,3.]))
            with self.assertRaises(AssertionError):module.compare(a,b)
            np.savez_compressed(b/'model.npz',constant=0.,weights=np.array([1.,2.]))
            with self.assertRaises(AssertionError):module.compare(a,b)

    def test_runtime_variation_allowed_but_accuracy_change_detected(self):
        with TemporaryDirectory() as folder:
            a=Path(folder)/'a';b=Path(folder)/'b';a.mkdir();b.mkdir()
            (a/'metrics.csv').write_text('accuracy,fit_seconds\n0.95,1\n')
            (b/'metrics.csv').write_text('accuracy,fit_seconds\n0.95,7\n')
            self.assertEqual(len(module.compare(a,b)),1)
            (b/'metrics.csv').write_text('accuracy,fit_seconds\n0.94,7\n')
            with self.assertRaises(AssertionError):module.compare(a,b)

    def test_prediction_flip_detected(self):
        with TemporaryDirectory() as folder:
            a=Path(folder)/'a';b=Path(folder)/'b';a.mkdir();b.mkdir()
            np.savez_compressed(a/'predictions.npz',response=np.array([False,True]))
            np.savez_compressed(b/'predictions.npz',response=np.array([False,False]))
            with self.assertRaises(AssertionError):module.compare(a,b)

    def test_missing_output_and_parameter_change_detected(self):
        with TemporaryDirectory() as folder:
            a=Path(folder)/'a';b=Path(folder)/'b';a.mkdir();b.mkdir()
            (a/'manifest.json').write_text('{"seed":1,"workers":1}')
            with self.assertRaises(AssertionError):module.compare(a,b)
            (b/'manifest.json').write_text('{"seed":1,"workers":5}')
            self.assertEqual(len(module.compare(a,b)),1)
            (b/'manifest.json').write_text('{"seed":2}')
            with self.assertRaises(AssertionError):module.compare(a,b)


if __name__=='__main__':unittest.main()
