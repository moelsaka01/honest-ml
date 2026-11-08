import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import _paths
import run_eval as ev

TEMPLATES = _paths.ROOT / "skills" / "llm-eval" / "templates"


def write_jsonl(path, rows):
    Path(path).write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def run_quiet(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = ev.run(argv)
    return out, buf.getvalue()


class CheckTests(unittest.TestCase):
    def test_each_check_type(self):
        cases = [
            ({"type": "contains", "value": "paris"}, "It is Paris.", True),
            ({"type": "contains", "value": "paris", "case": True}, "It is Paris.", False),
            ({"type": "not_contains", "value": "error"}, "all good", True),
            ({"type": "equals", "value": "42"}, " 42\n", True),
            ({"type": "regex", "pattern": r"^\d+$"}, "123", True),
            ({"type": "regex", "pattern": r"^\d+$"}, "12a", False),
            ({"type": "json_valid"}, '{"a": 1}', True),
            ({"type": "json_valid"}, "{a: 1}", False),
            ({"type": "json_keys", "keys": ["a", "b"]}, '{"a": 1, "b": 2}', True),
            ({"type": "json_keys", "keys": ["a", "b"]}, '{"a": 1}', False),
            ({"type": "json_keys", "keys": ["a"]}, "not json", False),
            ({"type": "max_chars", "n": 5}, "123456", False),
            ({"type": "numeric", "value": 3.14, "tol": 0.01}, "pi is about 3.141", True),
            ({"type": "numeric", "value": 1000}, "that is 1,000 units", True),
            ({"type": "numeric", "value": 5}, "no numbers here", False),
            ({"type": "bogus"}, "x", False),
        ]
        for check, output, expected in cases:
            self.assertEqual(ev.run_check(check, output)[0], expected, (check, output))


class RowFieldCheckTests(unittest.TestCase):
    def test_field_checks_read_the_output_row(self):
        row = {"output": "x", "judge_pass": True, "retrieved_ids": ["d1", "d7"]}
        self.assertTrue(ev.run_check({"type": "field_true", "field": "judge_pass"}, "x", row)[0])
        self.assertFalse(ev.run_check({"type": "field_true", "field": "missing"}, "x", row)[0])
        self.assertTrue(ev.run_check({"type": "field_contains", "field": "retrieved_ids", "value": "d7"}, "x", row)[0])
        self.assertFalse(ev.run_check({"type": "field_contains", "field": "retrieved_ids", "value": "d9"}, "x", row)[0])
        self.assertFalse(ev.run_check({"type": "field_contains", "field": "nope", "value": "d9"}, "x", row)[0])


class RunTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.cases = [{"id": f"c{i}", "input": "q", "expected": f"ans{i}", "tags": ["a" if i % 2 else "b"]}
                      for i in range(40)]
        write_jsonl(self.d / "cases.jsonl", self.cases)

    def tearDown(self):
        self.tmp.cleanup()

    def outputs(self, pass_ids, name):
        rows = [{"id": c["id"], "output": c["expected"] if c["id"] in pass_ids else "wrong",
                 "cost": 0.001, "latency_s": 1.0} for c in self.cases]
        write_jsonl(self.d / name, rows)
        return str(self.d / name)

    def test_pass_rate_and_flips(self):
        old = self.outputs({f"c{i}" for i in range(30)}, "old.jsonl")
        new = self.outputs({f"c{i}" for i in range(5, 40)}, "new.jsonl")
        out, text = run_quiet(["--cases", str(self.d / "cases.jsonl"), "--outputs", new, "--baseline", old])
        self.assertAlmostEqual(out["new"], 35 / 40)
        self.assertEqual(sorted(out["flipped_to_fail"]), sorted(f"c{i}" for i in range(5)))
        self.assertEqual(sorted(out["flipped_to_pass"]), sorted(f"c{i}" for i in range(30, 40)))
        self.assertIn("Flipped to pass", text)

    def test_flip_lists_are_exact(self):
        old = self.outputs({"c0", "c1", "c2"} | {f"c{i}" for i in range(10, 40)}, "old.jsonl")
        new = self.outputs({"c1", "c2", "c3"} | {f"c{i}" for i in range(10, 40)}, "new.jsonl")
        out, _ = run_quiet(["--cases", str(self.d / "cases.jsonl"), "--outputs", new, "--baseline", old])
        self.assertEqual(out["flipped_to_fail"], ["c0"])
        self.assertEqual(out["flipped_to_pass"], ["c3"])

    def test_regression_verdict(self):
        old = self.outputs({c["id"] for c in self.cases}, "old.jsonl")
        new = self.outputs({f"c{i}" for i in range(15)}, "new.jsonl")
        out, text = run_quiet(["--cases", str(self.d / "cases.jsonl"), "--outputs", new, "--baseline", old])
        self.assertIn("regressed", out["verdict"])

    def test_small_eval_set_says_needs_more_cases(self):
        small = self.cases[:5]
        write_jsonl(self.d / "small.jsonl", small)
        write_jsonl(self.d / "o.jsonl", [{"id": c["id"], "output": c["expected"]} for c in small])
        out, text = run_quiet(["--cases", str(self.d / "small.jsonl"), "--outputs", str(self.d / "o.jsonl"),
                               "--baseline", str(self.d / "o.jsonl")])
        self.assertEqual(out["verdict"], "needs more cases")
        self.assertIn("fewer than 30", text)

    def test_missing_ungraded_and_unknown_are_reported(self):
        cases = self.cases + [{"id": "nograde", "input": "q"}]
        write_jsonl(self.d / "cases2.jsonl", cases)
        rows = [{"id": c["id"], "output": c["expected"]} for c in self.cases[:30]]
        rows.append({"id": "ghost", "output": "x"})
        write_jsonl(self.d / "o2.jsonl", rows)
        out, text = run_quiet(["--cases", str(self.d / "cases2.jsonl"), "--outputs", str(self.d / "o2.jsonl")])
        self.assertIn("NOT graded", text)
        self.assertIn("no output", text)
        self.assertIn("unknown case ids", text)
        self.assertAlmostEqual(out["new"], 30 / 40)

    def test_multiple_runs_are_averaged(self):
        write_jsonl(self.d / "c.jsonl", [{"id": "a", "input": "q", "expected": "yes"}])
        write_jsonl(self.d / "o.jsonl", [{"id": "a", "output": "yes", "run": 0},
                                         {"id": "a", "output": "no", "run": 1}])
        s = ev.score(ev.read_jsonl(str(self.d / "c.jsonl")), ev.read_jsonl(str(self.d / "o.jsonl")))
        self.assertEqual(s["rates"]["a"], 0.5)

    def test_tags_breakdown_printed(self):
        o = self.outputs({c["id"] for c in self.cases}, "o.jsonl")
        _, text = run_quiet(["--cases", str(self.d / "cases.jsonl"), "--outputs", o])
        self.assertIn("By tag", text)

    def test_bad_input_exits_2(self):
        (self.d / "bad.jsonl").write_text("{not json}\n", encoding="utf-8")
        for argv in (["--cases", str(self.d / "bad.jsonl"), "--outputs", str(self.d / "bad.jsonl")],
                     ["--cases", "nope.jsonl", "--outputs", "nope.jsonl"]):
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
                run_quiet(argv)
            self.assertEqual(cm.exception.code, 2)
        write_jsonl(self.d / "dup.jsonl", [{"id": "a", "expected": "x"}, {"id": "a", "expected": "y"}])
        write_jsonl(self.d / "o.jsonl", [{"id": "a", "output": "x"}])
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
            run_quiet(["--cases", str(self.d / "dup.jsonl"), "--outputs", str(self.d / "o.jsonl")])
        self.assertEqual(cm.exception.code, 2)


class NegativeControlTests(unittest.TestCase):
    """An eval that cannot fail cannot meaningfully pass: prove the checks can fail."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.cases = [
            {"id": "strong", "input": "q", "expected": "Paris"},
            {"id": "vacuous", "input": "q", "checks": [{"type": "not_contains", "value": "Traceback"}]},
            {"id": "size", "input": "q", "checks": [{"type": "max_chars", "n": 50}]},
        ]
        write_jsonl(self.d / "cases.jsonl", self.cases)
        write_jsonl(self.d / "good.jsonl", [{"id": "strong", "output": "Paris"},
                                            {"id": "vacuous", "output": "fine"},
                                            {"id": "size", "output": "short"}])

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, *extra):
        return ["--cases", str(self.d / "cases.jsonl"), "--outputs", str(self.d / "good.jsonl"), *extra]

    def test_known_bad_that_fails_everywhere_is_ok(self):
        write_jsonl(self.d / "bad.jsonl", [{"id": "strong", "output": "Lyon"},
                                           {"id": "vacuous", "output": "Traceback (most recent call last)"},
                                           {"id": "size", "output": "x" * 80}])
        out, text = run_quiet(self.args("--known-bad", str(self.d / "bad.jsonl")))
        self.assertEqual(out["cannot_fail"], [])
        self.assertIn("OK: all 3 covered cases failed", text)

    def test_a_check_that_passes_a_wrong_output_is_reported(self):
        write_jsonl(self.d / "bad.jsonl", [{"id": "strong", "output": "Lyon"},
                                           {"id": "vacuous", "output": "completely wrong answer"},   # still passes
                                           {"id": "size", "output": "x" * 80}])
        out, text = run_quiet(self.args("--known-bad", str(self.d / "bad.jsonl")))
        self.assertEqual(out["cannot_fail"], ["vacuous"])
        self.assertIn("cannot fail", text)

    def test_cases_without_a_known_bad_output_are_listed_as_uncontrolled(self):
        write_jsonl(self.d / "bad.jsonl", [{"id": "strong", "output": "Lyon"}])
        out, text = run_quiet(self.args("--known-bad", str(self.d / "bad.jsonl")))
        self.assertIn("not controlled", text)
        self.assertIn("vacuous", text)
        self.assertEqual(out["cannot_fail"], [])

    def test_a_known_bad_file_covering_nothing_says_so(self):
        write_jsonl(self.d / "bad.jsonl", [{"id": "ghost", "output": "x"}])
        out, text = run_quiet(self.args("--known-bad", str(self.d / "bad.jsonl")))
        self.assertIn("covers none of the graded cases", text)
        self.assertIn("unknown case ids", text)
        self.assertNotIn("OK:", text)

    def test_junk_check_warns_about_cases_that_accept_junk(self):
        out, text = run_quiet(self.args("--junk-check"))
        self.assertEqual(out["junk_passes"], ["size", "vacuous"])      # empty / short junk passes these two
        self.assertIn("WARNING", text.split("Junk check")[1])
        self.assertEqual(out["cannot_fail"], [])                       # junk only warns, it never fails the run

    def test_junk_check_is_clean_when_every_check_demands_something(self):
        write_jsonl(self.d / "c2.jsonl", [{"id": "a", "input": "q", "expected": "Paris"},
                                          {"id": "b", "input": "q", "checks": [{"type": "numeric", "value": 7}]}])
        write_jsonl(self.d / "o2.jsonl", [{"id": "a", "output": "Paris"}, {"id": "b", "output": "7"}])
        out, text = run_quiet(["--cases", str(self.d / "c2.jsonl"), "--outputs", str(self.d / "o2.jsonl"),
                               "--junk-check"])
        self.assertEqual(out["junk_passes"], [])
        self.assertIn("no junk output passes", text)

    def test_field_checks_fail_on_junk(self):
        write_jsonl(self.d / "c3.jsonl", [{"id": "j", "input": "q", "checks": [{"type": "field_true", "field": "judge_pass"}]}])
        write_jsonl(self.d / "o3.jsonl", [{"id": "j", "output": "x", "judge_pass": True}])
        out, _ = run_quiet(["--cases", str(self.d / "c3.jsonl"), "--outputs", str(self.d / "o3.jsonl"), "--junk-check"])
        self.assertEqual(out["junk_passes"], [])

    def test_default_run_prints_no_negative_control_section(self):
        _, text = run_quiet(self.args())
        self.assertNotIn("Negative control", text)
        self.assertNotIn("Junk check", text)

    def test_exit_code_is_1_only_when_a_known_bad_output_passes(self):
        script = _paths.ROOT / "skills" / "llm-eval" / "scripts" / "run_eval.py"
        write_jsonl(self.d / "bad_ok.jsonl", [{"id": "strong", "output": "Lyon"}])
        write_jsonl(self.d / "bad_weak.jsonl", [{"id": "vacuous", "output": "anything"}])
        base = [sys.executable, str(script), *self.args()]
        for bad, code in (("bad_ok.jsonl", 0), ("bad_weak.jsonl", 1)):
            r = subprocess.run(base + ["--known-bad", str(self.d / bad)], capture_output=True, text=True)
            self.assertEqual(r.returncode, code, r.stdout + r.stderr)


class TemplateTests(unittest.TestCase):
    def test_shipped_templates_pass_end_to_end(self):
        out, text = run_quiet(["--cases", str(TEMPLATES / "cases.example.jsonl"),
                               "--outputs", str(TEMPLATES / "outputs.example.jsonl")])
        self.assertEqual(out["new"], 1.0)
        self.assertIn("fewer than 30", text)

    def test_shipped_known_bad_template_fails_every_case(self):
        out, text = run_quiet(["--cases", str(TEMPLATES / "cases.example.jsonl"),
                               "--outputs", str(TEMPLATES / "outputs.example.jsonl"),
                               "--known-bad", str(TEMPLATES / "known_bad.example.jsonl")])
        self.assertEqual(out["cannot_fail"], [], text)
        self.assertIn("OK: all 6 covered cases failed", text)

    def test_junk_check_exposes_the_weak_template_case(self):
        out, _ = run_quiet(["--cases", str(TEMPLATES / "cases.example.jsonl"),
                            "--outputs", str(TEMPLATES / "outputs.example.jsonl"), "--junk-check"])
        self.assertEqual(out["junk_passes"], ["regress-empty-input"])   # only forbids 'Traceback'; junk satisfies it


if __name__ == "__main__":
    unittest.main()
