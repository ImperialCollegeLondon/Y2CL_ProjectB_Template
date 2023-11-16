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
        cmd = rf'pydocstyle --select=D100,D102,D103,D419 "{course_files}"'
        res = run(cmd, shell=True, capture_output=True, check=False, text=True)
        if res.returncode:
            print("pydocstyle")
            print("---------")
            print(f"cmd:\n{cmd}")
            print(f"return code: {res.returncode}")
            print(f"stdout:\n{res.stdout}")
            print(f"stderr:\n{res.stderr}")
            assert not res.stderr, "Problem running pydocstyle command."
        missing_docs = len(res.stdout.splitlines()) / 2
        if missing_docs:
            print(f"Num of problems: {missing_docs}")
            print(f"stdout:\n{res.stdout}")
        assert missing_docs == 0

    def test_documentation_style(self):
        project_path = os.path.split(os.path.dirname(__file__))[0]
        course_files = " ".join(repr(path) for path in glob(os.path.join(project_path, "thermosnooker", "[a-zA-Z]*.py")))
        cmd = rf'darglint "{course_files}"'
        res = run(cmd, shell=True, capture_output=True, check=False, text=True)
        if res.returncode:
            print("darglight")
            print("---------")
            print(f"cmd:\n{cmd}")
            print(f"return code: {res.returncode}")
            print(f"stdout:\n{res.stdout}")
            print(f"stderr:\n{res.stderr}")
            assert not res.stderr, "Problem running darglint command."
        malformed_docs = max(len(res.stdout.splitlines()) - 1, 0)
        if malformed_docs:
            print(f"Num of problems: {malformed_docs}")
            print(f"stdout:\n{res.stdout}")
        assert malformed_docs == 0
