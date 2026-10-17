"""SCEPTR generator on the fixture data (needs sceptr_env; skipped where `sceptr` is not installed)."""
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import embeddings as emb

SCRIPT = ROOT / "scripts" / "generate_sceptr_embeddings.py"
HAVE_SCEPTR = importlib.util.find_spec("sceptr") is not None


def run(*args, check=True):
    res = subprocess.run([sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True)
    if check and res.returncode != 0:
        raise AssertionError(f"generator failed ({res.returncode}):\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
    return res


@unittest.skipUnless(HAVE_SCEPTR, "sceptr is not installed in this environment")
class SceptrMicro(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        cls.out = cls.d / "embeddings" / "sceptr" / "train"
        run("--input_path", ROOT / "tests" / "data" / "train", "--output_path", cls.out, "--max_samples", "3")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_output_is_valid_and_loadable_by_the_pipeline(self):
        d = emb.embedding_dir(self.d, "sceptr", "train")
        meta = emb.load_dataset_meta(d)
        cfg = emb.load_embedding_config(d)
        keys, E = emb.concat_embeddings(emb.h5_path(d), meta["sample_id"].tolist())
        self.assertEqual((len(meta), E.shape[1]), (3, 64))
        self.assertEqual((cfg["method"], cfg["variant"], cfg["latent_dim"]), ("sceptr", "default", 64))
        np.testing.assert_allclose(np.linalg.norm(E, axis=1), 1.0, atol=1e-4)       # unit-norm vectors

    def test_rejected_v_genes_are_dropped_and_logged(self):
        cfg = json.loads((self.out / "embedding_config.json").read_text())
        self.assertGreater(cfg["n_rows_dropped_invalid_v"], 0)                      # fixture has TRBV21-1 etc.
        self.assertTrue(set(cfg["invalid_v_genes"]) & {"TRBV21-1", "TRBV23-1", "TRBV5-3", "TRBV1"})

    def test_error_policy_stops_and_cdr3_only_keeps_everything(self):
        bad = run("--input_path", ROOT / "tests" / "data" / "train", "--output_path", self.d / "x",
                  "--max_samples", "3", "--invalid_v_policy", "error", check=False)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("V genes SCEPTR rejects", bad.stderr + bad.stdout)
        run("--input_path", ROOT / "tests" / "data" / "train", "--output_path", self.d / "y",
            "--max_samples", "3", "--variant", "cdr3_only")
        cfg = json.loads((self.d / "y" / "embedding_config.json").read_text())
        self.assertEqual(cfg["n_rows_dropped_invalid_v"], 0)

    def test_cuda_fails_clearly_without_a_gpu(self):
        try:
            import torch
            if torch.cuda.is_available():
                self.skipTest("this machine has CUDA")
        except ImportError:
            pass
        res = run("--input_path", ROOT / "tests" / "data" / "train", "--output_path", self.d / "z",
                  "--max_samples", "1", "--device", "cuda", check=False)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("no CUDA device", res.stderr + res.stdout)


COMPACT = ROOT / "scripts" / "compact_embeddings.py"
CHECK = ROOT / "scripts" / "check_embeddings.py"
RAW = ROOT / "tests" / "data"


def script(path, *args):
    return subprocess.run([sys.executable, str(path), *map(str, args)], capture_output=True, text=True)


@unittest.skipUnless(HAVE_SCEPTR, "sceptr is not installed in this environment")
class SceptrCompact(unittest.TestCase):
    """float16 storage and deleting the CSVs after a successful check."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = Path(cls.tmp.name)
        cls.f32 = cls.d / "a" / "embeddings" / "sceptr" / "train"
        run("--input_path", RAW / "train", "--output_path", cls.f32, "--max_samples", "3")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def copy(self, name):
        dst = self.d / name / "embeddings" / "sceptr" / "train"
        shutil.copytree(self.f32, dst)
        return dst

    def test_generator_writes_float16_and_deletes_csvs_after_checks(self):
        out = self.d / "g" / "embeddings" / "sceptr" / "train"
        res = run("--input_path", RAW / "train", "--output_path", out, "--max_samples", "3",
                  "--h5_dtype", "float16", "--delete_csv")
        self.assertIn("CSVs deleted", res.stdout)
        self.assertEqual(list(out.glob("*_embed.csv")), [])
        cfg = json.loads((out / "embedding_config.json").read_text())
        self.assertEqual((cfg["h5_dtype"], cfg["csv_removed"]), ("float16", True))
        self.assertEqual(emb.h5_dtype_of(out / emb.H5_NAME), "float16")
        meta = emb.load_dataset_meta(out)
        keys, E = emb.concat_embeddings(emb.h5_path(out), meta["sample_id"].tolist())
        self.assertEqual(E.dtype, np.float32)                       # the pipeline always sees float32
        _, E32 = emb.concat_embeddings(emb.h5_path(self.f32), meta["sample_id"].tolist())
        order = np.argsort(keys), np.argsort(self._keys32())
        np.testing.assert_allclose(E[order[0]], E32[order[1]], atol=3e-4)
        chk = script(CHECK, "--data_dir", self.d / "g", "--raw_dir", RAW, "--check_sequences", "--check_values")
        self.assertEqual(chk.returncode, 0, chk.stdout[-1500:])     # checked from the HDF5 alone
        self.assertIn("deleted earlier", chk.stdout)

    def _keys32(self):
        meta = emb.load_dataset_meta(self.f32)
        return emb.concat_embeddings(emb.h5_path(self.f32), meta["sample_id"].tolist())[0]

    def test_compact_converts_an_existing_folder_and_deletes_csvs(self):
        dst = self.copy("c")
        before = (dst / emb.H5_NAME).stat().st_size
        res = script(COMPACT, "--data_dir", self.d / "c", "--raw_dir", RAW, "--h5_dtype", "float16", "--delete_csv")
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertEqual(emb.h5_dtype_of(dst / emb.H5_NAME), "float16")
        self.assertLess((dst / emb.H5_NAME).stat().st_size, before)
        self.assertEqual(list(dst.glob("*_embed.csv")), [])
        again = script(COMPACT, "--data_dir", self.d / "c", "--raw_dir", RAW, "--h5_dtype", "float16", "--delete_csv")
        self.assertEqual(again.returncode, 0)
        self.assertIn("already float16", again.stdout)

    def test_failed_check_keeps_the_csvs(self):
        dst = self.copy("bad")
        victim = sorted(dst.glob("*_embed.csv"))[0]
        victim.write_text(victim.read_text().splitlines()[0] + "\n")      # header only: all sequences lost
        n = len(list(dst.glob("*_embed.csv")))
        res = script(COMPACT, "--data_dir", self.d / "bad", "--raw_dir", RAW, "--delete_csv")
        self.assertEqual(res.returncode, 1)
        self.assertIn("KEPT", res.stdout)
        self.assertEqual(len(list(dst.glob("*_embed.csv"))), n)
        self.assertNotIn("csv_removed", json.loads((dst / "embedding_config.json").read_text()))

    def test_unfinished_dataset_is_left_untouched(self):
        dst = self.copy("unf")
        (dst / "embedding_config.json").unlink()
        res = script(COMPACT, "--data_dir", self.d / "unf", "--raw_dir", RAW, "--h5_dtype", "float16", "--delete_csv")
        self.assertEqual(res.returncode, 0)
        self.assertIn("unfinished", res.stdout)
        self.assertEqual(emb.h5_dtype_of(dst / emb.H5_NAME), "float32")
        self.assertGreater(len(list(dst.glob("*_embed.csv"))), 0)

    def test_float16_overflow_is_refused(self):
        with self.assertRaises(ValueError):
            emb._as_h5_dtype(np.array([7e4], dtype=np.float32), "float16")


if __name__ == "__main__":
    unittest.main()
