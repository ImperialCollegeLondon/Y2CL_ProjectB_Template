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
def rt():
    rays_module = import_module("raytracer.rays")
    elements_module = import_module("raytracer.elements")
    lenses_module = import_module("raytracer.lenses")
    return CombinedNamespace(rays_module, elements_module, lenses_module)

@pytest.fixture
def ph():
    return import_module("raytracer.physics")

@pytest.fixture
def rays():
    return import_module("raytracer.rays")

@pytest.fixture
def elements():
    return import_module("raytracer.elements")

@pytest.fixture
def lenses():
    return import_module("raytracer.lenses")

@pytest.fixture
def an():
    yield import_module("raytracer.analysis")
    plt.close()

@pytest.fixture
def var_name_map(rays):
    pos = [1.,2.,3.]
    direc = [4., 5., 6.]
    udirec = np.array(direc) / np.linalg.norm(direc)
    r = rays.Ray(pos=pos, direc=direc)

    ret={}
    for var_name, val in vars(r).items():
        if np.allclose(val, pos):
            ret['pos'] = var_name
        elif np.allclose(val, direc) or np.allclose(val, udirec):
            ret['direc'] = var_name
    return ret

@pytest.fixture
def default_ray(rays):
    return rays.Ray()

@pytest.fixture
def default_ray_pos(rays, var_name_map):
    return getattr(rays.Ray(), var_name_map["pos"])

@pytest.fixture
def default_ray_direc(rays, var_name_map):
    return getattr(rays.Ray(), var_name_map["direc"])

@pytest.fixture
def test_ray(rays):
    return rays.Ray(pos=[1.,2.,3.], direc=[4.,5.,6.])

@pytest.fixture
def test_ray_pos(rays, var_name_map):
    return getattr(rays.Ray(pos=[1.,2.,3.], direc=[4.,5.,6.]), var_name_map["pos"])

@pytest.fixture
def test_ray_direc(rays, var_name_map):
    return getattr(rays.Ray(pos=[1.,2.,3.], direc=[4.,5.,6.]), var_name_map["direc"])

@pytest.fixture
def ray_bundle(rays):
    return getattr(rays, "RayBundle", getattr(rays, "Bundle", None))
