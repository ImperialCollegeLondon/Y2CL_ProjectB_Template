"""Pytest fixtures."""
from importlib import import_module
from pathlib import Path
from unittest.mock import patch
import pytest
import numpy as np
import scipy.optimize as spo
import matplotlib
import matplotlib.axes as mplaxes
import matplotlib.pyplot as plt

matplotlib.use("agg")  # Non-interactive backend more stable for testing than interactive Tk


# pylint: disable=redefined-outer-name, unused-argument
@pytest.fixture(scope="function")
def balls():
    return import_module('thermosnooker.balls')


@pytest.fixture(scope="function")
def simulations():
    return import_module("thermosnooker.simulations")


@pytest.fixture(scope="function")
def physics():
    return import_module("thermosnooker.physics")


@pytest.fixture(scope="function")
def an():
    yield import_module("thermosnooker.analysis")
    plt.close('all')


@pytest.fixture
def default_ball(balls):
    return balls.Ball()


@pytest.fixture
def custom_ball(balls):
    return balls.Ball(pos=[1., 2.], vel=[3., 4.], radius=5., mass=6.)


@pytest.fixture
def container_class(balls):
    return balls.Container


@pytest.fixture
def default_container(balls):
    return balls.Container()


@pytest.fixture
def colliding_ball(balls):
    return balls.Ball(pos=[9., 0.])


@pytest.fixture(scope="module")
def source_files():
    excluded_files = ()
    src_files = [str(file_) for file_ in Path(__file__).parent.parent.glob("thermosnooker/[a-zA-Z]*.py")
                 if file_.name not in excluded_files]
    assert src_files, "No source files found to check!"
    return src_files


@pytest.fixture(scope="module")
def source_files_str(source_files):
    return ' '.join(source_files)

# @pytest.fixture(scope="function")
# def an_mock_run(simulations_mod, monkeypatch):
#     matplotlib.use("agg")  # Non-interactive backend more stable for testing that interactive Tk
#     with monkeypatch.context() as m:
#         run_mock = MagicMock(autospec=True)
#         m.setattr(simulations_mod.MultiBallSimulation, "run", run_mock)
#         yield import_module("thermosnooker.analysis"), run_mock
#     plt.close()

# @pytest.fixture(scope="function")
# def an_mock_run(simulations_mod):
#     matplotlib.use("agg")  # Non-interactive backend more stable for testing that interactive Tk

#     simulations_mod.MultiBallSimulation._run = simulations_mod.MultiBallSimulation.run
#     with patch.object(simulations_mod.MultiBallSimulation,
#                       "run",
#                       autospec=True,
#                       side_effect=lambda self, _: simulations_mod.MultiBallSimulation._run(self, 1)) as run_mock:
#         yield import_module("thermosnooker.analysis"), run_mock
#     delattr(simulations_mod.MultiBallSimulation, "_run")
#     plt.close()


@pytest.fixture(scope="function")
def curve_fit_mock():
    with patch("scipy.optimize.curve_fit", wraps=spo.curve_fit) as cf_mock:
        yield cf_mock


@pytest.fixture(scope="function")
def sbs_run_mock(simulations):
    original_run = simulations.SingleBallSimulation.run
    with (patch.object(simulations.SingleBallSimulation,
                       "__init__",
                       autospec=True,
                       wraps=simulations.SingleBallSimulation.__init__) as sbs_mock,
          patch.object(simulations.SingleBallSimulation,
                       "run",
                       autospec=True,
                       side_effect=lambda self, *args, **kwargs: original_run(self, 1)) as run_mock):
        yield sbs_mock, run_mock


@pytest.fixture(scope="function")
def sbs_run20_mock(simulations):
    """Specially for Task9 running to get numerical output."""
    original_run = simulations.SingleBallSimulation.run
    with (patch.object(simulations.SingleBallSimulation,
                       "__init__",
                       autospec=True,
                       wraps=simulations.SingleBallSimulation.__init__) as sbs_mock,
          patch.object(simulations.SingleBallSimulation,
                       "run",
                       autospec=True,
                       side_effect=lambda self, *args, **kwargs: original_run(self, 20)) as run_mock):
        yield sbs_mock, run_mock


@pytest.fixture(scope="function")
def mbs_run_mock(simulations):
    original_run = simulations.MultiBallSimulation.run
    with (patch.object(simulations.MultiBallSimulation,
                       "__init__",
                       autospec=True,
                       wraps=simulations.MultiBallSimulation.__init__) as mbs_mock,
          patch.object(simulations.MultiBallSimulation,
                       "run",
                       autospec=True,
                       side_effect=lambda self, *args, **kwargs: original_run(self, 1)) as run_mock):
        yield mbs_mock, run_mock


@pytest.fixture(scope="function")
def bms_run_mock(simulations):
    original_run = simulations.BrownianSimulation.run
    with (patch.object(simulations.BrownianSimulation,
                       "__init__",
                       autospec=True,
                       wraps=simulations.BrownianSimulation.__init__) as bms_mock,
          patch.object(simulations.BrownianSimulation,
                       "run",
                       autospec=True,
                       side_effect=lambda self, *args, **kwargs: original_run(self, 1)) as run_mock):
        yield bms_mock, run_mock


@pytest.fixture
def var_name_map(custom_ball):
    ret = {}
    for var_name, var_val in vars(custom_ball).items():
        if isinstance(var_val, (list, np.ndarray)) and len(var_val) and np.allclose(var_val, [1., 2.]):
            ret["pos"] = var_name
        elif isinstance(var_val, (list, np.ndarray)) and len(var_val) and np.allclose(var_val, [3., 4.]):
            ret["vel"] = var_name
        elif isinstance(var_val, (int, float)) and np.isclose(var_val, 5.):
            ret["radius"] = var_name
        elif isinstance(var_val, (int, float)) and np.isclose(var_val, 6.):
            ret["mass"] = var_name
    return ret


@pytest.fixture(scope="function")
def range_mock(an):
    with patch.object(an, "range", return_value=range(1, 10)) as range_mock:
        yield range_mock


@pytest.fixture(scope="function")
def hist_mock():
    with (patch("matplotlib.pyplot.hist") as hist_mock1,
          patch.object(mplaxes.Axes, "hist", autospec=True) as hist_mock2):
        yield hist_mock1, hist_mock2


@pytest.fixture(scope="function")
def task9_output(an):
    yield an.task9()


@pytest.fixture(scope="function")
def task11_output(range_mock, an):
    yield an.task11()


@pytest.fixture(scope="function")
def task12_output(range_mock, an):
    yield an.task12()


@pytest.fixture(scope="function")
def task13_output(an):
    yield an.task13()


@pytest.fixture(scope="function")
def task14_output(an):
    yield an.task14()


@pytest.fixture(scope="function")
def task15_output(an):
    yield an.task15()


@pytest.fixture(scope="function")
def task16_output(an):
    yield an.task16()


@pytest.fixture(scope="function")
def task17_output(an):
    yield an.task17()

# @pytest.fixture
# def sim_module():
#     return import_module("simulation")

# @pytest.fixture
# def my_simulation(simulation_module):
#     """
#     create simulation instance
#     """
#     my_simulation = simulation_module.Simulation()
#     return my_simulation
