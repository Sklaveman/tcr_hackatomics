"""Unit tests for the cluster-occupancy features in src/clustering.py and the embedding ingestion in
src/embeddings.py (small synthetic embeddings; needs faiss: sceptr_env)."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HAVE_FAISS = importlib.util.find_spec("faiss") is not None
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic
from src import clustering as cl
from src import embeddings as emb


class Ingestion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.info = synthetic.make_synthetic_data(cls.tmp.name, n_per_disease=3, n_seq=60, n_bulk=3,
                                                 n_sc=3, n_public=4)
        cls.dir = emb.embedding_dir(cls.tmp.name, "sceptr", "train")
        cls.meta = emb.load_dataset_meta(cls.dir)
        cls.h5 = emb.h5_path(cls.dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_metadata_and_labels(self):
        self.assertEqual(len(self.meta), 18)
        self.assertEqual(self.meta.groupby("disease_group")["y"].first().to_dict(),
                         {"autoimmunity": 0, "cancer": 1, "infection": 2})
        self.assertNotIn("MHCI", self.meta.columns)

    def test_dim_inferred_from_data(self):
        self.assertEqual(emb.infer_embedding_dim(self.dir, self.meta["sample_id"][0]), 32)

    def test_concat_pools_unique_sequences(self):
        ids = self.meta["sample_id"].tolist()
        keys, E = emb.concat_embeddings(self.h5, ids)
        per = emb.read_sample_keys(self.h5, ids)
        self.assertEqual(len(keys), len(set(np.concatenate(list(per.values())))))
        self.assertEqual(len(set(keys)), len(keys))                 # no duplicates
        self.assertEqual(E.shape, (len(keys), 32))
        # a public clonotype is shared by several samples but pooled once
        pub = self.info["public"][0][0].encode()
        self.assertGreater(sum(pub in k for k in per.values()), 1)
        self.assertEqual(int((keys == pub).sum()), 1)

    def test_concat_keep_keys_filters(self):
        ids = self.meta["sample_id"].tolist()
        keep = np.array([self.info["public"][c][0].encode() for c in range(3)], dtype="S120")
        keys, E = emb.concat_embeddings(self.h5, ids, keep_keys=keep)
        self.assertEqual(sorted(keys.tolist()), sorted(keep.tolist()))
        self.assertEqual(E.shape, (3, 32))

    def test_table_roundtrip_csv_fallback(self):
        import pandas as pd
        df = pd.DataFrame({"a": [1.0, 2.0]}, index=["s1", "s2"])
        with tempfile.TemporaryDirectory() as d:
            path = emb.save_table(df, str(Path(d) / "t"))
            self.assertTrue(emb.table_exists(str(Path(d) / "t")))
            pd.testing.assert_frame_equal(emb.load_table(str(Path(d) / "t")), df)
            self.assertTrue(path.endswith((".parquet", ".csv.gz")))


@unittest.skipUnless(HAVE_FAISS, "faiss is not installed in this environment")
class Clusters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = cls.tmp.name
        synthetic.make_synthetic_data(cls.tmp.name, n_per_disease=4, n_seq=120, n_bulk=3, n_sc=3)
        cls.dir = emb.embedding_dir(cls.tmp.name, "sceptr", "train")
        cls.meta = emb.load_dataset_meta(cls.dir)
        cls.h5 = emb.h5_path(cls.dir)
        ids = cls.meta["sample_id"].tolist()
        cls.keys, cls.E = emb.concat_embeddings(cls.h5, ids)
        cls.model = cl.fit_model(cls.keys, cls.E, 20, 0.95, 10, spherical=True, seed=42, log=lambda *_: None)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_clusters_named_and_labels_valid(self):
        m = self.model
        self.assertTrue(1 <= m["n_clusters"] <= 20)
        self.assertEqual(m["cluster_names"], [f"c{i}" for i in range(m["n_clusters"])])
        self.assertTrue((m["labels_sorted"] >= -1).all())

    def test_features_shape_and_sums(self):
        ids = self.meta["sample_id"].tolist()[:4]
        f = cl.extract_features(self.h5, ids, self.model, verbose=False)
        K = self.model["n_clusters"]
        self.assertEqual(f.shape, (4, K + 3))
        self.assertEqual(list(f.columns[-3:]), ["unassigned", "evenness", "log_depth"])
        np.testing.assert_allclose(f.iloc[:, :K].sum(axis=1) + f["unassigned"], 1.0, atol=1e-9)
        self.assertTrue(((f["evenness"] >= 0) & (f["evenness"] <= 1)).all())

    def test_downsampling_caps_depth_and_is_reproducible(self):
        ids = self.meta["sample_id"].tolist()[:3]
        a = cl.extract_features(self.h5, ids, self.model, downsample_to=40, random_state=1, verbose=False)
        b = cl.extract_features(self.h5, ids, self.model, downsample_to=40, random_state=1, verbose=False)
        np.testing.assert_allclose(a["log_depth"], np.log(40))
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy())

    def test_nearest_centroid_cache_is_used_and_gives_the_same_features(self):
        ids = self.meta["sample_id"].tolist()[:3]
        cache = {}
        a = cl.extract_features(self.h5, ids, self.model, verbose=False, neighbor_cache=cache)
        self.assertEqual(len(cache), 3)
        b = cl.extract_features(self.h5, ids, self.model, verbose=False, neighbor_cache=cache)
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy())

    def test_align_features_raises_on_missing(self):
        f = cl.extract_features(self.h5, self.meta["sample_id"].tolist()[:2], self.model, verbose=False)
        with self.assertRaises(ValueError):
            cl.align_features(f, ["c0", "nope"], "x")

    def test_model_pickles_and_reproduces_assignment(self):
        import pickle
        m2 = pickle.loads(pickle.dumps(self.model))
        e = emb.embedding_dir(self.root, "sceptr", "test_bulk")
        h5 = emb.h5_path(e)
        tb = emb.load_dataset_meta(e)["sample_id"].tolist()[:2]
        a = cl.extract_features(h5, tb, self.model, verbose=False)
        b = cl.extract_features(h5, tb, m2, verbose=False)
        np.testing.assert_allclose(a.to_numpy(), b.to_numpy())


if __name__ == "__main__":
    unittest.main()
