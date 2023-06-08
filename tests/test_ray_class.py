from inspect import signature
from types import NoneType
import numpy as np
import pytest
from fixtures import *

class TestInternals:

    def test_default_construction(self, default_ray):
        pass

    def test_construction(self, test_ray):
        pass

    def test_partial_construction(self, rays):
        rays.Ray([1.,2.,3.])
        rays.Ray(direc=[4.,5.,6.])

    def test_pos_iterable_check(self, rays):
        with pytest.raises(Exception) as excinfo:
            rays.Ray(pos=12)

    def test_pos_too_long_check(self, rays):
        with pytest.raises(Exception) as excinfo:
            rays.Ray(pos=[1., 2., 3., 4.])

    def test_pos_too_short_check(self, rays):
        with pytest.raises(Exception) as excinfo:
            rays.Ray(pos=[1., 2.])

    def test_default_pos_list(self, default_ray_pos):
        assert isinstance(default_ray_pos, list)
 
    def test_pos_list(self, test_ray_pos):
        assert isinstance(test_ray_pos, list)

    def test_default_pos_single_element(self, default_ray_pos):
        assert len(default_ray_pos) == 1

    def test_pos_single_element(self, test_ray_pos):
        assert len(test_ray_pos) == 1

    def test_default_pos_ndarray(self, default_ray_pos):
        assert isinstance(default_ray_pos[0], np.ndarray)

    def test_pos_ndarray(self, test_ray_pos):
        assert isinstance(test_ray_pos[0], np.ndarray)

    def test_default_pos_dtype(self, default_ray_pos):
        init_pos = default_ray_pos[0]
        assert init_pos.dtype == np.dtype("float64") or \
               init_pos.dtype == np.dtype("float32")

    def test_pos_dtype(self, test_ray_pos):
        init_pos = test_ray_pos[0]
        assert init_pos.dtype == np.dtype("float64") or \
               init_pos.dtype == np.dtype("float32")

    def test_pos_set(self, test_ray_pos):
        assert np.allclose(test_ray_pos[0], [1., 2., 3.])

    def test_dir_iterable_check(self, rays):
        with pytest.raises(Exception) as excinfo:
            rays.Ray(direc=12)

    def test_dir_too_long_check(self, rays):
        with pytest.raises(Exception) as excinfo:
            rays.Ray(direc=[1., 2., 3., 4.])

    def test_dir_too_short_check(self, rays):
        with pytest.raises(Exception) as excinfo:
            rays.Ray(direc=[1., 2.])

    def test_default_dir_ndarray(self, default_ray_direc):
        assert isinstance(default_ray_direc, np.ndarray)

    def test_dir_ndarray(self, test_ray_direc):
        assert isinstance(test_ray_direc, np.ndarray)
        
    def test_default_dir_dtype(self, default_ray_direc):
        assert default_ray_direc.dtype == np.dtype("float64") or \
               default_ray_direc.dtype == np.dtype("float32")
        
    def test_dir_dtype(self, test_ray_direc):
        assert test_ray_direc.dtype == np.dtype("float64") or \
               test_ray_direc.dtype == np.dtype("float32")

    def test_default_dir_normalised(self, default_ray_direc):
        assert np.isclose(np.linalg.norm(default_ray_direc), 1.)

    def test_dir_normalised(self, test_ray_direc):
        assert np.isclose(np.linalg.norm(test_ray_direc), 1.)

    def test_dir_sensible_default(self, default_ray_direc):
        assert np.allclose(default_ray_direc, [0., 0., 1.])

    def test_k_set(self, test_ray_direc):
        test_k = np.array([4., 5. ,6.])
        test_k /= np.linalg.norm(test_k)
        assert np.allclose(test_ray_direc, test_k)

    def test_hidden_variables(self, rays):
        for var in vars(rays.Ray()):
            assert var.startswith('_')

    def test_constructor_default_types(self, rays):
        params = signature(rays.Ray.__init__).parameters
        pos_default = params["pos"].default
        direc_default = params["direc"].default
        assert isinstance(pos_default, (NoneType, list, np.ndarray))        
        assert isinstance(direc_default, (NoneType, list, np.ndarray))
