import os
from subprocess import run
from glob import glob
from pylint.lint import Run


class TestStyle:

    def setup_class(self):
        project_path = os.path.split(os.path.dirname(__file__))[0]
        course_file = glob(os.path.join(project_path, "raytracer", "*.py"))

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

def test_documentation():
    project_path = os.path.split(os.path.dirname(__file__))[0]
    course_files = " ".join(glob(os.path.join(project_path, "raytracer", "*.py")))
    cmd = rf"pydocstyle --select=D100,D102,D103,D419 {course_files}"
    res = run(cmd, shell=True, capture_output=True, check=False)
    missing_docs = len(res.stdout.splitlines()) / 2
    assert missing_docs == 0
