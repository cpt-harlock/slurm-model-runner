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


if __name__ == "__main__":
    unittest.main()
