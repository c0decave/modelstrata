"""Guardrails for the no-Node/no-browser-tooling host policy."""
import os
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class NoNodePolicyTests(unittest.TestCase):
    def test_no_node_package_manifests(self):
        forbidden = {
            "package.json",
            "package-lock.json",
            "pnpm-lock.yaml",
            "yarn.lock",
            "playwright.config.js",
            "playwright.config.ts",
        }
        found = []
        for base, dirs, files in os.walk(ROOT):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
            for name in files:
                if name in forbidden:
                    found.append(os.path.relpath(os.path.join(base, name), ROOT))
        self.assertEqual([], found)

    def test_core_host_paths_do_not_recommend_node_or_playwright(self):
        checked = [
            "README.md",
            "README.en.md",
            "tests/run.sh",
            "tools/analyze.py",
            "tools/build_dashboard.py",
        ]
        forbidden = (
            "node --check",
            "nodejs",
            "npm install",
            "npx ",
            "pnpm ",
            "yarn install",
            "playwright install",
        )
        hits = []
        for rel in checked:
            with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
                text = fh.read().lower()
            for needle in forbidden:
                if needle in text:
                    hits.append(f"{rel}: {needle.strip()}")
        self.assertEqual([], hits)

    def test_browser_asset_scripts_are_opt_in_before_import(self):
        for rel in ("scripts/build_logo.py", "scripts/shoot_demo_dashboard.py"):
            with open(os.path.join(ROOT, rel), encoding="utf-8") as fh:
                text = fh.read()
            guard = text.index("def require_workstation_opt_in")
            call = text.index("require_workstation_opt_in()")
            import_pos = text.index("from playwright.sync_api import sync_playwright")
            self.assertLess(guard, import_pos)
            self.assertLess(call, import_pos)


if __name__ == "__main__":
    unittest.main()
