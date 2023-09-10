"""Test Tasks
Laura Hollister
Septemvber 2023
    """

import pytest
import numpy as np
import matplotlib.pyplot as plt
from types import FunctionType, MethodType


class TestTask1():
    def test_pos_method(self, ball_mod):
        assert isinstance(ball_mod.Ball.pos, FunctionType)

    def test_default_pos_array(self, ball_mod):
        ball = ball_mod.Ball()
        assert isinstance(ball.pos(), np.ndarray)

    def test_default_ball_pos(self, default_ball):
        assert np.allclose(default_ball.pos(), np.array([0, 0]))

    def test_set_ball_pos(self, ball_mod):
        ball = ball_mod.Ball(pos=[1., 2.])
        assert np.allclose(ball.pos(), np.array([1., 2.]))

    def test_pos_input(self, ball_mod):
        with pytest.raises(Exception):
            ball = ball_mod.Ball(pos=[1, 2, 3])
        with pytest.raises(Exception):
            ball = ball_mod.Ball(pos=[1])
        with pytest.raises(Exception):
            ball = ball_mod.Ball(pos=4)

    def test_vel_method(self, ball_mod):
        assert isinstance(ball_mod.Ball.vel, FunctionType)

    def test_default_vel_array(self, default_ball):
        assert isinstance(default_ball.vel(), np.ndarray)

    def test_default_ball_vel(self, default_ball):
        assert np.allclose(default_ball.vel(), np.array([1, 0]))

    def test_set_ball_vel(self, ball_mod):
        ball = ball_mod.Ball(vel=[1., 2.])
        assert np.allclose(ball.vel(), np.array([1., 2.]))

    def test_vel_input(self, ball_mod):
        with pytest.raises(Exception):
            ball = ball_mod.Ball(vel=[1, 2, 3])
        with pytest.raises(Exception):
            ball = ball_mod.Ball(vel=[1])
        with pytest.raises(Exception):
            ball = ball_mod.Ball(vel=4)

    def test_mass_method(self, ball_mod):
        assert isinstance(ball_mod.Ball.mass, FunctionType)

    def test_mass_type(self, default_ball):
        assert isinstance(default_ball.mass(), float)

    def test_default_mass(self, default_ball):
        assert default_ball.mass() == 1.

    def test_set_mass(self, ball_mod):
        ball = ball_mod.Ball(mass=5)
        assert ball.mass() == 5.

    def test_radius_method(self, ball_mod):
        assert isinstance(ball_mod.Ball.radius, FunctionType)

    def test_radius_type(self, default_ball):
        assert isinstance(default_ball.radius(), float)

    def test_default_radius(self, default_ball):
        assert default_ball.radius() == 0.1

    def test_set_radius(self, ball_mod):
        ball = ball_mod.Ball(radius=2)
        assert ball.radius() == 2.

    def test_move_exists(self, ball_mod):
        assert hasattr(ball_mod.Ball, 'move')

    def test_move_correct(self, default_ball):
        default_ball.move(3)
        assert np.allclose(default_ball.pos(), np.array([3, 0]))

    def move_correct(self, default_ball):
        with pytest.raises(Exception):
            default_ball.move(-0.4)

class TestTask2():
    def test_default_con_rad_type(self, default_con):
        assert isinstance(default_con.radius, MethodType)
    
    def test_default_rad_type(self, default_con):
        assert isinstance(default_con.radius(), float)
    
    def test_default_con_rad(self, default_con):
        assert default_con.radius() == 10.
        
    def test_repr(self, con_class):
        assert hasattr(con_class, '__repr__')

class TestTask3():
    def test_repr(self, ball_mod):
        assert hasattr(ball_mod.Ball, '__repr__')

    def test_get_patch_ball(self, ball_mod, default_ball):
        assert isinstance(ball_mod.Ball.get_patch, FunctionType)
        assert isinstance(default_ball.get_patch(), plt.Circle)

    def test_get_patch_container(self, ball_mod, default_con):
        assert isinstance(ball_mod.Container.get_patch, FunctionType)
        assert isinstance(default_con.get_patch(), plt.Circle)
        
        
class TestTask4():
    def test_time_to_collision_parallel(self, ball_mod):
        ball1 = ball_mod.Ball(pos=[0.1, 0], vel=[0., 2.])
        ball2 = ball_mod.Ball(pos=[0., 1.], vel=[0., 2.])
        assert ball1.time_to_collision(ball2) is None
    
    def test_time_to_collision_past(self, ball_mod):
        ball1 = ball_mod.Ball(pos=[2, 0], vel=[1, 0])
        ball2 = ball_mod.Ball(pos=[-3, 0], vel=[-1, 0])
        assert ball1.time_to_collision(ball2) is None   
    
    
    def test_time_to_collision_balls(self, ball_mod):
        ball1 = ball_mod.Ball(pos=[1, 1], vel=[1., 0])
        ball2 = ball_mod.Ball(pos=[5., 4.], vel=[0, -0.75])
        assert np.isclose(ball1.time_to_collision(ball2), 3.84)
               
    def test_time_to_collision_con(self, ball_mod, default_con):
        ball1 = ball_mod.Ball(pos=[1., -1.], vel=[0., 2.])
        assert np.isclose(ball1.time_to_collision(default_con), 5.42468273)

class TestTask5():
    def test_collide_functionality_1(self, ball_mod):
        ball1 = ball_mod.Ball(pos=[1, 1.], vel=[1, 0])
        ball2 = ball_mod.Ball(pos=[5, 4], vel=[0., -0.75])
        time = ball1.time_to_collision(ball2)
        ball1.move(time)
        ball2.move(time)
        ball1.collide(ball2)

        assert np.allclose(ball1.pos(),[4.84, 1])
        assert np.allclose(ball2.pos() ,[5, 1.12])
        assert np.allclose(ball1.vel(),[0, -0.75])
        assert np.allclose(ball2.vel(),[1,0])

    def test_collide_functionality_2(self, ball_mod, default_con):
        ball1 = ball_mod.Ball(pos=[1., -1.], vel=[0.,2.])
        time = ball1.time_to_collision(default_con)
        ball1.move(time)
        ball1.collide(default_con)

        assert np.allclose(ball1.pos() ,[1., 9.84936546])
        assert np.allclose(default_con.pos(),[0., 0.])
        assert np.allclose(ball1.vel(),[-0.4019739 , -1.95918784])
        assert np.allclose(default_con.vel() ,[0., 0.])

