import os
from subprocess import run
from glob import glob
from pylint.lint import Run


class TestStyle:

    def setup_class(self):
        project_path = os.path.split(os.path.dirname(__file__))[0]
        course_file = glob(os.path.join(project_path, "thermosnooker", "[a-zA-Z]*.py"))

        pylint_results = Run(course_file, exit=False)
        self.style_result = pylint_results.linter.stats.global_note

    def test_20pc(self):
        assert self.style_result >= 2.

    def test_40pc(self):
        assert self.style_result >= 4.

    def test_60pc(self):
        assert self.style_result >= 6.

    def test_80pc(self):
        assert self.style_result >= 8.

    def test_90pc(self):
        assert self.style_result >= 9.

class TestDocumentation:

    def test_documentation_present(self):
        project_path = os.path.split(os.path.dirname(__file__))[0]
        course_files = " ".join(repr(path) for path in glob(os.path.join(project_path, "thermosnooker", "[a-zA-Z]*.py")))
        cmd = rf"pydocstyle --select=D100,D102,D103,D419 {course_files}"
        res = run(cmd, shell=True, capture_output=True, check=False, text=True)
        if res.returncode:
            print("pydocstyle")
            print("---------")
            print(f"cmd:\n{cmd}")
            print(f"return code: {res.returncode}")
            print(f"stdout:\n{res.stdout}")
            print(f"stderr:\n{res.stderr}")
        assert res.returncode == 0, "pydocstyle failed"
        missing_docs = len(res.stdout.splitlines()) / 2
        if missing_docs:
            print(res.stdout)
        assert missing_docs == 0

    def test_documentation_style(self):
        project_path = os.path.split(os.path.dirname(__file__))[0]
        course_files = " ".join(repr(path) for path in glob(os.path.join(project_path, "thermosnooker", "[a-zA-Z]*.py")))
        cmd = rf"darglint {course_files}"
        res = run(cmd, shell=True, capture_output=True, check=False, text=True)
        if res.returncode:
            print("darglight")
            print("---------")
            print(f"cmd:\n{cmd}")
            print(f"return code: {res.returncode}")
            print(f"stdout:\n{res.stdout}")
            print(f"stderr:\n{res.stderr}")
        assert res.returncode == 0, "darglint failed"
        malformed_docs = len(res.stdout.splitlines())
        if malformed_docs:
            print(malformed_docs)
        assert malformed_docs == 0
