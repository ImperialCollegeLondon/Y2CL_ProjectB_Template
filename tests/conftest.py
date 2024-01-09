"""Pytest fixtures."""
from importlib import import_module
from pathlib import Path
import pytest
import numpy as np
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
    return balls_mod.Ball(pos=[1., 2.], vel=[3., 4.], radius=5, mass=6.)

@pytest.fixture
def container_class(balls_mod):
    return balls_mod.Container

@pytest.fixture
def default_container(balls_mod):
    return balls_mod.Container()


@pytest.fixture(scope="session")
def simulation_mod():
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
