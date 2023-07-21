import os
from subprocess import run
from glob import glob
from pylint.lint import Run


class TestStyle:

    def setup_class(self):
        project_path = os.path.split(os.path.dirname(__file__))[0]
        course_file = glob(os.path.join(project_path, "thermosnooker", "*.py"))

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
        course_files = " ".join(glob(os.path.join(project_path, "thermosnooker", "*.py")))
        cmd = rf"pydocstyle --select=D100,D102,D103,D419 {course_files}"
        res = run(cmd, shell=True, capture_output=True, check=False, text=True)
        missing_docs = len(res.stdout.splitlines()) / 2
        if missing_docs:
            print(res.stdout)
        assert missing_docs == 0

    def test_documentation_style(self):
        project_path = os.path.split(os.path.dirname(__file__))[0]
        course_files = " ".join(glob(os.path.join(project_path, "thermosnooker", "*.py")))
        cmd = rf"darglint {course_files}"
        res = run(cmd, shell=True, capture_output=True, check=False, text=True)
        print(res.stdout)
        malformed_docs = len(res.stdout.splitlines()) - 1
        if malformed_docs:
            print(malformed_docs)
        assert malformed_docs == 0
