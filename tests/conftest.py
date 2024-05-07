"""Pytest fixtures."""
from importlib import import_module
from pathlib import Path
import pytest
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


@pytest.fixture(scope="session")
def balls_mod():
    #return import_module('thermosnooker.newballs')
    return import_module('thermosnooker.balls')


@pytest.fixture
def default_ball(balls_mod):
    return balls_mod.Ball()

@pytest.fixture
def custom_ball(balls_mod):
    return balls_mod.Ball(pos=[1., 2.], vel=[3., 4.], radius=5., mass=6.)

@pytest.fixture
def container_class(balls_mod):
    return balls_mod.Container

@pytest.fixture
def default_container(balls_mod):
    return balls_mod.Container()

@pytest.fixture
def colliding_ball(balls_mod):
    return balls_mod.Ball(pos=[9., 0.])

@pytest.fixture(scope="session")
def simulations_mod():
    return import_module("thermosnooker.simulations")


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


@pytest.fixture(scope="session")
def an():
    matplotlib.use("agg")  # Non-interactive backend more stable for testing that interactive Tk
    yield import_module("thermosnooker.analysis")
    plt.close()

@pytest.fixture(scope="session")
def physics_mod():
    return import_module("thermosnooker.physics")

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
