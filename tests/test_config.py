import unittest
from pathlib import Path

from mr import config


class ResolveSifTest(unittest.TestCase):
    def test_empty_means_global_image(self):
        self.assertEqual(config.resolve_sif(""), config.MR_SIF)
        self.assertEqual(config.resolve_sif(Path()), config.MR_SIF)

    def test_bare_name_resolves_next_to_global_image(self):
        self.assertEqual(config.resolve_sif("vllm-0.32.sif"), config.MR_SIF.parent / "vllm-0.32.sif")

    def test_absolute_path_kept(self):
        self.assertEqual(config.resolve_sif("/x/y.sif"), Path("/x/y.sif"))


class LoadSifTest(unittest.TestCase):
    def test_every_shipped_toml_resolves_a_sif(self):
        # Models without `sif` keep the global image; any that set one must
        # still land in an absolute path the job can open.
        for name in config.available():
            m = config.load(name)
            self.assertTrue(m.sif.is_absolute(), name)


class SiteTest(unittest.TestCase):
    def test_every_site_profile_loads_with_required_slurm_keys(self):
        for p in sorted(config.SITES_DIR.glob("*.toml")):
            site = config.load_site(p.stem)
            for key in ("account", "partition", "qos", "time"):
                self.assertIn(key, site.slurm, f"{p.stem} lacks slurm.{key}")
            for key in ("tag", "dest"):
                self.assertIn(key, site.image, f"{p.stem} lacks image.{key}")
            self.assertTrue(site.weights, p.stem)

    def test_unknown_site_is_a_clear_error(self):
        with self.assertRaises(FileNotFoundError):
            config.load_site("no-such-cluster")

    def test_site_slurm_fills_what_the_model_leaves_out(self):
        # Shipped model TOMLs only set `nodes`; the rest is the site's.
        for name in config.available():
            m = config.load(name)
            self.assertEqual(m.slurm.account, config.SITE.slurm["account"], name)
            self.assertEqual(m.slurm.time, config.SITE.slurm["time"], name)


class PerSiteOverrideTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())
        self._models_dir = config.MODELS_DIR
        config.MODELS_DIR = self.tmp
        (self.tmp / "m.toml").write_text(
            'hf_repo = "org/M"\nsif = "a.sif"\n'
            '[slurm]\nnodes = 2\n'
            '[engine]\npipeline_parallel_size = 2\n'
            f'[sites.{config.MR_SITE}]\nsif = ""\n'
            f'[sites.{config.MR_SITE}.slurm]\nnodes = 3\n'
            f'[sites.{config.MR_SITE}.engine]\npipeline_parallel_size = 3\n'
            '[sites.elsewhere.slurm]\nnodes = 99\n'
        )

    def tearDown(self):
        config.MODELS_DIR = self._models_dir

    def test_current_site_table_wins_and_others_are_ignored(self):
        m = config.load("m")
        self.assertEqual(m.slurm.nodes, 3)
        self.assertEqual(m.engine.pipeline_parallel_size, 3)
        self.assertEqual(m.sif, config.MR_SIF)          # "" -> site default image
        self.assertEqual(m.slurm.qos, config.SITE.slurm["qos"])


class WeightsDirTest(unittest.TestCase):
    def test_env_var_in_weights_dir_expands(self):
        import os
        m = config.ModelSpec(name="x", hf_repo="o/X", slurm=None, engine=None,
                             weights_dir="$SCRATCH/weights/X")
        self.assertEqual(m.weights_dir, Path(os.environ["SCRATCH"]) / "weights" / "X")

    def test_unset_env_var_is_a_clear_error(self):
        with self.assertRaises(RuntimeError):
            config.ModelSpec(name="x", hf_repo="o/X", slurm=None, engine=None,
                             weights_dir="$MR_DEFINITELY_UNSET_VAR/X")

    def test_empty_weights_dir_defaults_under_mr_weights(self):
        m = config.ModelSpec(name="x", hf_repo="o/X", slurm=None, engine=None)
        self.assertEqual(m.weights_dir, config.MR_WEIGHTS / "X")


if __name__ == "__main__":
    unittest.main()
