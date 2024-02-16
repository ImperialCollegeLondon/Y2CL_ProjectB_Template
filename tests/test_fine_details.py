import re
from pprint import pformat
from inspect import getmembers, isclass, getsource
from types import MethodType
from unittest.mock import MagicMock
import pytest
import numpy as np
from matplotlib.patches import Circle


class TestBallInternals:

    def test_pos_list_or_array_check(self, balls_mod):
        with pytest.raises(Exception) as excinfo:
            balls_mod.Ball(pos=12)

    def test_pos_too_long_check(self, balls_mod):
        with pytest.raises(Exception) as excinfo:
            balls_mod.Ball(pos=[1., 2., 3., 4.])

    def test_pos_too_short_check(self, balls_mod):
        with pytest.raises(Exception) as excinfo:
            balls_mod.Ball(pos=[1.])

    def test_hidden_vars(self, balls_mod):
        a = set(vars(balls_mod.Ball()).keys())
        b = set(vars(Circle([0., 0.], 5)).keys())

        public_vars = {i for i in a.difference(b) if not i.startswith("_")}
        assert not public_vars

    def test_vel_list_or_array_check(self, balls_mod):
        with pytest.raises(Exception) as excinfo:
            balls_mod.Ball(pos=12)

    def test_vel_too_long_check(self, balls_mod):
        with pytest.raises(Exception) as excinfo:
            balls_mod.Ball(pos=[1., 2., 3., 4.])

    def test_vel_too_short_check(self, balls_mod):
        with pytest.raises(Exception) as excinfo:
            balls_mod.Ball(pos=[1.])

    def test_default_pos_type(self, default_ball, var_name_map):
        pos = getattr(default_ball, var_name_map["pos"])
        assert isinstance(pos, np.ndarray)
        assert pos.dtype == np.float_

    def test_default_vel_type(self, default_ball, var_name_map):
        vel = getattr(default_ball, var_name_map["vel"])
        assert isinstance(vel, np.ndarray)
        assert vel.dtype == np.float_

    def test_custom_pos_type(self, custom_ball, var_name_map):
        pos = getattr(custom_ball, var_name_map["pos"])
        assert isinstance(pos, np.ndarray)
        assert pos.dtype == np.float_

    def test_custom_vel_type(self, custom_ball, var_name_map):
        vel = getattr(custom_ball, var_name_map["vel"])
        assert isinstance(vel, np.ndarray)
        assert vel.dtype == np.float_

    def test_set_vel_list_or_array_check(self, default_ball):
        with pytest.raises(Exception):
            if isinstance(default_ball.vel, MethodType):
                default_ball.set_vel(1.)
            else:
                default_ball.vel = 1.

    def test_set_vel_too_short(self, default_ball):
        with pytest.raises(Exception):
            if isinstance(default_ball.vel, MethodType):
                default_ball.set_vel([1.])
            else:
                default_ball.vel = [1.]

    def test_set_vel_too_long(self, default_ball):
        with pytest.raises(Exception):
            if isinstance(default_ball.vel, MethodType):
                default_ball.set_vel([1.,2.,3.,4.])
            else:
                default_ball.vel = [1.,2.,3.,4.]

    def test_set_vel_type(self, default_ball, var_name_map):
        if isinstance(default_ball.vel, MethodType):
            default_ball.set_vel([1., 2.])
        else:
            default_ball.vel = [1., 2.]
        vel = getattr(default_ball, var_name_map["vel"])
        assert isinstance(vel, np.ndarray)
        assert vel.dtype == np.float_


DATA_ATTRIBUTE_REGEX = re.compile(r"^\s*self\.([_a-zA-Z0-9]+)[^=]*=(?!=)", re.MULTILINE)


class TestAdvancedDesign:

    def test_container_inheritance(self, balls_mod):
        assert balls_mod.Container.__bases__ == (balls_mod.Ball,)

    def test_container_inherits_ttc(self, balls_mod):
        assert "time_to_collision" not in vars(balls_mod.Container)
        assert hasattr(balls_mod.Container, "time_to_collision")

    def test_container_doesnt_hide_vars(self, balls_mod):
        ball_vars = set(DATA_ATTRIBUTE_REGEX.findall(getsource(balls_mod.Ball.__init__)))
        cont_vars = set(DATA_ATTRIBUTE_REGEX.findall(getsource(balls_mod.Container.__init__)))
        hidden_vars = ball_vars.intersection(cont_vars)
        assert not hidden_vars, f"Container hides the following variables from Base class Ball:\n {pformat(hidden_vars)}"

    def test_hidden_variables(self, balls_mod, simulations_mod):
        non_hidden_vars = set()
        for module in (balls_mod, simulations_mod):
            for name, cls in getmembers(module, isclass):
                if module == balls_mod and cls == Circle:
                    continue
                if init_func := vars(cls).get("__init__", False):
                    non_hidden_vars.update(f"{name}.{var}" for var in DATA_ATTRIBUTE_REGEX.findall(getsource(init_func))
                                           if not var.startswith('_'))

        assert not non_hidden_vars, f"Non hidden data attributes:\n {pformat(non_hidden_vars)}"

    def test_collide_doesnt_call_ttc(self, balls_mod, monkeypatch):
        b1 = balls_mod.Ball(pos=[-5., 0.], vel=[1., 0.])
        b2 = balls_mod.Ball(pos=[5., 0.], vel=[-1., 0.])

        ttc_mock1 = MagicMock(wraps=b1.time_to_collision)
        ttc_mock2 = MagicMock(wraps=b2.time_to_collision)
        with monkeypatch.context() as m:
            m.setattr(b1, "time_to_collision", ttc_mock1)
            m.setattr(b2, "time_to_collision", ttc_mock2)
            b1.collide(b2)

        ttc_mock1.assert_not_called()
        ttc_mock2.assert_not_called()
