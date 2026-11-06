"""Every skill must be well-formed so that agents can load it and users cannot hit a dead link."""
import json
import re
import shutil
import subprocess
import sys
import unittest

import _paths

ROOT = _paths.ROOT
SKILLS = ROOT / "skills"
EXPECTED = {"leakage-hunter", "honest-eval", "dataset-audit", "reproducible-runs", "llm-eval", "ml-roast"}


def frontmatter(text):
    text = text.replace("\r\n", "\n")  # Windows checkouts may convert line endings
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert m, "missing YAML front matter"
    out = {}
    for line in m.group(1).splitlines():
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


class SkillFileTests(unittest.TestCase):
    def test_exactly_the_six_skills_exist(self):
        self.assertEqual({p.name for p in SKILLS.iterdir() if p.is_dir()}, EXPECTED)

    def test_frontmatter_is_valid(self):
        for name in EXPECTED:
            text = (SKILLS / name / "SKILL.md").read_text(encoding="utf-8")
            fm = frontmatter(text)
            self.assertEqual(set(fm), {"name", "description"}, name)
            self.assertEqual(fm["name"], name)
            self.assertTrue(re.fullmatch(r"[a-z0-9-]{1,64}", fm["name"]))
            self.assertTrue(40 < len(fm["description"]) <= 1024, f"{name}: description length {len(fm['description'])}")

    def test_scripts_named_in_skills_exist_and_are_documented_in_readme(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for name in EXPECTED:
            text = (SKILLS / name / "SKILL.md").read_text(encoding="utf-8")
            for rel in set(re.findall(r"<skill-dir>/((?:scripts|templates)/[\w./-]+)", text)):
                self.assertTrue((SKILLS / name / rel).exists(), f"{name}: references missing {rel}")
            for script in (SKILLS / name / "scripts").glob("*.py") if (SKILLS / name / "scripts").exists() else []:
                self.assertIn(script.name, readme, f"README does not mention {script.name}")
                self.assertIn(script.name, text, f"{name}/SKILL.md does not document {script.name}")

    def test_every_skill_mentions_skill_dir_meaning_and_works_with(self):
        for name in EXPECTED:
            text = (SKILLS / name / "SKILL.md").read_text(encoding="utf-8")
            if "<skill-dir>" in text:
                self.assertIn("means the folder that contains this SKILL.md", text, name)
            self.assertIn("## Works with", text, name)

    def test_readme_links_resolve(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        readme = re.sub(r"<!--.*?-->", "", readme, flags=re.DOTALL)  # the GIF placeholder is commented out
        for target in re.findall(r"\]\(((?!http|#)[^)]+)\)", readme):
            self.assertTrue((ROOT / target).exists(), f"README link is broken: {target}")
        for target in re.findall(r'href="((?!http|#)[^"]+)"', readme):
            self.assertTrue((ROOT / target).exists(), f"README href is broken: {target}")

    def test_readme_names_all_skills(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for name in EXPECTED:
            self.assertIn(f"skills/{name}/SKILL.md", readme)


class ManifestTests(unittest.TestCase):
    def test_plugin_manifests_are_valid_json(self):
        plugin = json.loads((ROOT / ".claude-plugin" / "plugin.json").read_text(encoding="utf-8"))
        market = json.loads((ROOT / ".claude-plugin" / "marketplace.json").read_text(encoding="utf-8"))
        self.assertEqual(plugin["name"], "honest-ml")
        self.assertTrue(re.fullmatch(r"\d+\.\d+\.\d+", plugin["version"]))
        self.assertEqual(market["plugins"][0]["name"], plugin["name"])
        self.assertTrue((ROOT / "LICENSE").exists())


class ScriptHygieneTests(unittest.TestCase):
    def test_scripts_are_ascii_so_windows_consoles_cannot_crash(self):
        for path in list(SKILLS.glob("*/scripts/*.py")) + list((ROOT / "examples").glob("*.py")):
            data = path.read_bytes()
            try:
                data.decode("ascii")
            except UnicodeDecodeError as exc:
                self.fail(f"{path.relative_to(ROOT)} has non-ASCII bytes at {exc.start}")

    def test_every_script_has_working_help(self):
        for path in SKILLS.glob("*/scripts/*.py"):
            args = [sys.executable, str(path), "--help"]
            r = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, f"{path.name}: {r.stderr[-300:]}")
            self.assertIn("usage", r.stdout.lower(), path.name)


class DemoGifTests(unittest.TestCase):
    def test_gif_is_valid_embedded_and_reasonably_small(self):
        gif = ROOT / "assets" / "demo.gif"
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("assets/demo.gif", re.sub(r"<!--.*?-->", "", readme, flags=re.DOTALL))
        data = gif.read_bytes()
        self.assertTrue(data.startswith((b"GIF87a", b"GIF89a")))
        self.assertLess(len(data), 2_000_000, "keep the README GIF light")
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed: header check only")
        with Image.open(gif) as im:
            self.assertGreater(im.n_frames, 10)
            self.assertGreaterEqual(im.width, 600)
            self.assertEqual(im.info.get("loop"), 0)


class DocsMatchBehaviourTests(unittest.TestCase):
    def test_ml_roast_example_matches_the_real_scanner_output(self):
        """The worked example in ml-roast/SKILL.md must be what the scanner really prints."""
        import roast_scan
        text = (SKILLS / "ml-roast" / "SKILL.md").read_text(encoding="utf-8")
        documented = set(re.findall(r"bad_example\.py:(\d+)\]?\s+\[(R\d+)\]", text))
        actual = {(f["where"].split(":")[1], f["rule"])
                  for f in roast_scan.scan_lines(roast_scan.load_lines(ROOT / "examples" / "bad_example.py"))}
        self.assertEqual(documented, actual)

    def test_every_scanner_rule_is_documented_in_the_skill(self):
        import roast_scan
        text = (SKILLS / "ml-roast" / "SKILL.md").read_text(encoding="utf-8")
        for rule in roast_scan.FIXES:
            self.assertIn(rule, text, f"{rule} is not described in ml-roast/SKILL.md")


@unittest.skipUnless(shutil.which("claude"), "Claude Code CLI not installed")
class ClaudeCliTests(unittest.TestCase):
    def test_plugin_and_marketplace_validate(self):
        for target in (ROOT, ROOT / ".claude-plugin" / "plugin.json"):
            r = subprocess.run(["claude", "plugin", "validate", str(target)], capture_output=True, text=True,
                               timeout=120)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertNotIn("warning", r.stdout.lower(), r.stdout)


if __name__ == "__main__":
    unittest.main()
