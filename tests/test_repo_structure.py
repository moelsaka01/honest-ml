"""Every skill must be well-formed so that agents can load it and users cannot hit a dead link."""
import json
import re
import subprocess
import sys
import unittest

import _paths

ROOT = _paths.ROOT
SKILLS = ROOT / "skills"
EXPECTED = {"leakage-hunter", "honest-eval", "dataset-audit", "reproducible-runs", "llm-eval", "ml-roast"}


def frontmatter(text):
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


if __name__ == "__main__":
    unittest.main()
