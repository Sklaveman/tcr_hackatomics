"""FAISS k-means clustering (src/clustering.py); skipped where faiss is not installed."""
import importlib.util
import pickle
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
HAVE_FAISS = importlib.util.find_spec("faiss") is not None
if HAVE_FAISS:
    from src import clustering as km


def blobs(n_per=300, k=6, d=8, spread=0.3, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * 5
    X = np.vstack([c + rng.normal(scale=spread, size=(n_per, d)) for c in centers]).astype(np.float32)
    keys = np.array([b"CASS%dF+TRBV1_TRBJ1-1" % i for i in range(len(X))], dtype="S120")
    return keys, X, centers


@unittest.skipUnless(HAVE_FAISS, "faiss is not installed in this environment")
class KMeans(unittest.TestCase):
    def test_distances_are_euclidean_or_cosine(self):
        rng = np.random.default_rng(1)
        C = rng.normal(size=(5, 6)).astype(np.float32)
        X = rng.normal(size=(40, 6)).astype(np.float32)
        idx, dist = km.nearest_centroid(C, X, spherical=False)
        full = np.linalg.norm(X[:, None] - C[None], axis=2)
        np.testing.assert_array_equal(idx, full.argmin(1))
        np.testing.assert_allclose(dist, full.min(1), atol=1e-4)
        Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
        idx, dist = km.nearest_centroid(Cn, X, spherical=True)
        cos = (X / np.linalg.norm(X, axis=1, keepdims=True)) @ Cn.T
        np.testing.assert_array_equal(idx, cos.argmax(1))
        np.testing.assert_allclose(dist, 1 - cos.max(1), atol=1e-5)

    def test_radius_is_the_quantile_of_member_distances(self):
        idx = np.array([0] * 10 + [1] * 4 + [3] * 1)
        dist = np.concatenate([np.arange(1, 11), [5, 1, 3, 2], [7]]).astype(np.float32)
        r = km.radii(idx, dist, 4, 0.5)
        np.testing.assert_array_equal(r, [5, 2, -1, 7])                # cluster 2 is empty
        np.testing.assert_array_equal(km.radii(idx, dist, 4, 1.0), [10, 5, -1, 7])
        np.testing.assert_array_equal(km.radii(idx, dist, 4, 0.1), [1, 1, -1, 7])
        with self.assertRaises(ValueError):
            km.radii(idx, dist, 4, 0.0)

    def test_fit_assign_threshold_and_min_cluster_size(self):
        keys, X, centers = blobs()
        base = km.fit_base(keys, X, 6, spherical=False, seed=1, log=lambda *_: None)
        full = km.make_model(base, 1.0, 10)
        self.assertEqual(full["n_clusters"], 6)
        self.assertEqual(full["share_assigned_train"], 1.0)             # q = 1: every training member is inside
        tight = km.make_model(base, 0.8, 10)
        share = tight["share_assigned_train"]
        self.assertTrue(0.78 <= share <= 0.84, share)                   # about 80 % of every cluster
        # a far-away point is unassigned; a point on a centroid gets that cluster
        far = np.full((1, 8), 100.0, dtype=np.float32)
        self.assertEqual(km.assign(full, far)[0], -1)
        on = base["centroids"][2:3]
        self.assertEqual(km.assign(full, on)[0], full["remap"][2])
        # clusters smaller than min_cluster_size are dropped and the rest renumbered 0..K'-1
        big = km.make_model(base, 1.0, 301)
        self.assertEqual(big["n_clusters"], 0)
        some = km.make_model(base, 0.5, 120)
        kept = sorted(set(some["labels_unsorted"][some["labels_unsorted"] >= 0]))
        self.assertEqual(kept, list(range(some["n_clusters"])))
        self.assertEqual(some["cluster_names"], [f"c{i}" for i in range(some["n_clusters"])])

    def test_search_is_shared_across_radii_and_model_pickles(self):
        keys, X, _ = blobs()
        base = km.fit_base(keys, X, 6, spherical=False, seed=1, log=lambda *_: None)
        calls, real = [], km.nearest_centroid
        km.nearest_centroid = lambda *a, **kw: (calls.append(1), real(*a, **kw))[1]
        try:
            cache = {}
            a = km.assign(km.make_model(base, 0.8, 10), X[:500], cache=cache, cache_key="p")
            b = km.assign(km.make_model(base, 1.0, 10), X[:500], cache=cache, cache_key="p")
            self.assertEqual(len(calls), 1)                              # one search for two radii
        finally:
            km.nearest_centroid = real
        self.assertTrue((a == -1).sum() >= (b == -1).sum())
        m = km.make_model(base, 0.9, 10)
        again = pickle.loads(pickle.dumps(m))
        np.testing.assert_array_equal(km.assign(m, X[:200]), km.assign(again, X[:200]))

    def test_spherical_model_on_unit_vectors(self):
        keys, X, _ = blobs(spread=0.2)
        base = km.fit_base(keys, X, 6, spherical=True, seed=1, log=lambda *_: None)
        m = km.make_model(base, 0.9, 10)
        self.assertTrue(m["spherical"])
        self.assertTrue((m["radius"][m["radius"] >= 0] <= 2.0).all())    # cosine distance is in [0, 2]
        self.assertEqual(km.assign(m, X[:50]).shape, (50,))

    def test_too_few_sequences_is_a_clear_error(self):
        keys, X, _ = blobs(n_per=2, k=3)
        with self.assertRaises(ValueError):
            km.fit_base(keys, X, 10, spherical=False, log=lambda *_: None)

    def test_cuda_path_runs_where_faiss_offers_it(self):
        try:
            km.check_available("cuda")
        except SystemExit:
            self.skipTest("this FAISS cannot use a GPU")
        keys, X, _ = blobs()
        a = km.fit_base(keys, X, 6, spherical=False, device="cuda", seed=1, log=lambda *_: None)
        m = km.make_model(a, 1.0, 10)
        self.assertEqual(km.assign(m, X[:100], device="cuda").shape, (100,))
        self.assertEqual(m["n_clusters"], 6)

    def test_unusable_gpu_gives_a_clear_message_instead_of_a_crash(self):
        real = km.probe_gpu
        km.probe_gpu = lambda *a, **k: (False, "exit code -6: CUDA error 209 no kernel image is available")
        try:
            with self.assertRaises(SystemExit) as cm:
                km.check_available("cuda")
        finally:
            km.probe_gpu = real
        msg = str(cm.exception)
        self.assertIn("cannot use the GPU", msg)
        self.assertIn("no kernel image", msg)
        self.assertIn("--kmeans_device cpu", msg)

    def test_probe_survives_a_process_abort(self):
        import subprocess
        real, calls = subprocess.run, []
        def fake_run(cmd, **kw):                       # a child that aborts like FAISS does on an unsupported GPU
            calls.append(1)
            return subprocess.CompletedProcess(cmd, 134, "", "Faiss assertion 'err__ == cudaSuccess' failed")
        subprocess.run = fake_run
        try:
            ok, msg = km.probe_gpu()
        finally:
            subprocess.run = real
        self.assertFalse(ok)
        self.assertIn("exit code 134", msg)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
