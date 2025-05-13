import re
from pprint import pformat
from inspect import getmembers, isclass, getsource
from types import MethodType
from unittest.mock import MagicMock
import pytest
import numpy as np
from matplotlib.patches import Circle


class TestBallInternals:

    def test_pos_list_or_array_check(self, balls):
        with pytest.raises(Exception) as excinfo:
            balls.Ball(pos=12)

    def test_pos_too_long_check(self, balls):
        with pytest.raises(Exception) as excinfo:
            balls.Ball(pos=[1., 2., 3., 4.])

    def test_pos_too_short_check(self, balls):
        with pytest.raises(Exception) as excinfo:
            balls.Ball(pos=[1.])

    def test_hidden_vars(self, balls):
        a = set(vars(balls.Ball()).keys())
        b = set(vars(Circle([0., 0.], 5)).keys())

        public_vars = {i for i in a.difference(b) if not i.startswith("_")}
        assert not public_vars

    def test_vel_list_or_array_check(self, balls):
        with pytest.raises(Exception) as excinfo:
            balls.Ball(pos=[0., 0.], vel=12)

    def test_vel_too_long_check(self, balls):
        with pytest.raises(Exception) as excinfo:
            balls.Ball(pos=[0., 0.], vel=[1., 2., 3., 4.])

    def test_vel_too_short_check(self, balls):
        with pytest.raises(Exception) as excinfo:
            balls.Ball(pos=[0., 0.], vel=[1.])

    def test_default_pos_type(self, default_ball, var_name_map):
        pos = getattr(default_ball, var_name_map["pos"])
        assert isinstance(pos, np.ndarray)
        assert pos.dtype == float

    def test_default_vel_type(self, default_ball, var_name_map):
        vel = getattr(default_ball, var_name_map["vel"])
        assert isinstance(vel, np.ndarray)
        assert vel.dtype == float

    def test_custom_pos_type(self, custom_ball, var_name_map):
        pos = getattr(custom_ball, var_name_map["pos"])
        assert isinstance(pos, np.ndarray)
        assert pos.dtype == float

    def test_custom_vel_type(self, custom_ball, var_name_map):
        vel = getattr(custom_ball, var_name_map["vel"])
        assert isinstance(vel, np.ndarray)
        assert vel.dtype == float

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
        assert vel.dtype == float


DATA_ATTRIBUTE_REGEX = re.compile(r"^\s*self\.([_a-zA-Z0-9]+)[^=]*=(?!=)", re.MULTILINE)


class TestAdvancedDesign:

    def test_brownian_inheritance(self, simulations):
        assert simulations.BrownianSimulation.__bases__ == (simulations.MultiBallSimulation,)

    def test_container_inheritance(self, balls):
        assert balls.Container.__bases__ == (balls.Ball,)

    def test_container_inherits_ttc(self, balls):
        assert "time_to_collision" not in vars(balls.Container)
        assert hasattr(balls.Container, "time_to_collision")

    def test_container_doesnt_hide_vars(self, balls):
        ball_vars = set(DATA_ATTRIBUTE_REGEX.findall(getsource(balls.Ball.__init__)))
        cont_vars = set(DATA_ATTRIBUTE_REGEX.findall(getsource(balls.Container.__init__)))
        hidden_vars = ball_vars.intersection(cont_vars)
        assert not hidden_vars, f"Container hides the following variables from Base class Ball:\n {pformat(hidden_vars)}"

    def test_hidden_variables(self, balls, simulations):
        non_hidden_vars = set()
        for module in (balls, simulations):
            for name, cls in getmembers(module, isclass):
                if module == balls and cls == Circle:
                    continue
                if init_func := vars(cls).get("__init__", False):
                    non_hidden_vars.update(f"{name}.{var}" for var in DATA_ATTRIBUTE_REGEX.findall(getsource(init_func))
                                           if not var.startswith('_'))

        assert not non_hidden_vars, f"Non hidden data attributes:\n {pformat(non_hidden_vars)}"

    def test_collide_doesnt_call_ttc(self, balls, monkeypatch):
        b1 = balls.Ball(pos=[-5., 0.], vel=[1., 0.])
        b2 = balls.Ball(pos=[5., 0.], vel=[-1., 0.])

        ttc_mock1 = MagicMock(wraps=b1.time_to_collision)
        ttc_mock2 = MagicMock(wraps=b2.time_to_collision)
        with monkeypatch.context() as m:
            m.setattr(b1, "time_to_collision", ttc_mock1)
            m.setattr(b2, "time_to_collision", ttc_mock2)
            b1.collide(b2)

        ttc_mock1.assert_not_called()
        ttc_mock2.assert_not_called()
