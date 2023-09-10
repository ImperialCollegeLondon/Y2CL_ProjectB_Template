"""Pytest fixtures."""
from importlib import import_module
import pytest
import numpy as np
import matplotlib.pyplot as plt


class CombinedNamespace:
    def __init__(self, *modules):
        self._modules = modules

    def __getattr__(self, name):
        for module in self._modules:
            if (var:=getattr(module, name, None)) is not None:
                return var

        raise AttributeError("No such attribute: " + name)

    def __iter__(self):
        for module in self._modules:
            yield from vars(module).keys()

@pytest.fixture
def ball_mod():
    return import_module('ball')


@pytest.fixture
def default_ball(ball_mod):
    return ball_mod.Ball()

@pytest.fixture
def con_class(ball_mod):
    return ball_mod.Container

@pytest.fixture
def default_con(ball_mod):
    return ball_mod.Container()


@pytest.fixture
def sim_module():
    return import_module("simulation")

@pytest.fixture
def my_simulation(simulation_module):
    """
    create simulation instance
    """
    my_simulation = simulation_module.Simulation()
    return my_simulation
