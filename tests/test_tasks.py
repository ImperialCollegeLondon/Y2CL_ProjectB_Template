from fixtures import *
from inspect import signature
from types import FunctionType
from unittest.mock import MagicMock
import matplotlib as mpl
from matplotlib.testing.decorators import check_figures_equal
import pytest
from utils import check_figures_equal

import numpy as np
import pylab as pl
from types import FunctionType, MethodType
from conftest import *


class TestTask1:
    # TODO: how to check time_to_collision without defining balls?

    # --- functionality checks

    def test_correct_dt_1(self, ball_module):
        """
        check correct dt is calculated
        (balls moving towards each other on x-axis)
        """

        # TODO TEST INCORRECT - balls moving away from each other

        ball1 = ball_module.Ball(mass=0.1, radius=0.1,
                                 pos=[-0.1, 0.], vel=[-1., 0.])
        ball2 = ball_module.Ball(mass=0.1, radius=0.1, pos=[
                                 0.1, 0.], vel=[1.0, 0.])
        dt = ball1.time_to_collision(ball2)

        assert np.isclose(
            dt, 0.2), "Colliision expected, with dt=0.2. Please check time_to_collision()."

    def test_correct_dt_2(self, ball_module):
        """
        check correct dt is calculated - no collision
        (balls moving away from each other on x-axis)
        TODO BALLS ALREADY TOUCHING
        """

        ball1 = ball_module.Ball(mass=0.1, radius=0.1,
                                 pos=[-0.11, 0.], vel=[-1., 0.])
        ball2 = ball_module.Ball(mass=0.1, radius=0.1, pos=[
                                 0.11, 0.], vel=[1.0, 0.])
        dt = ball1.time_to_collision(ball2)

        assert dt is None, "No collision expected, dt should return None. Please check time_to_collision()."

    def test_correct_dt_3(self, ball_module):
        """
        check correct dt is calculated - opposite collision
        (balls moving towards each other)
        GOOD TEST
        """

        ball1 = ball_module.Ball(mass=0.1, radius=0.1, pos=[
                                 1.0, 0.], vel=[1., 3.])
        ball2 = ball_module.Ball(mass=0.1, radius=0.1, pos=[
                                 4.0, 1.], vel=[-2.0, 2.0])
        dt = ball1.time_to_collision(ball2)

        assert np.isclose(
            dt, 0.936754446), "Collision expected, with dt~0.936754446. Please check time_to_collision()."

    def test_correct_dt_4(self, ball_module):
        """
        check correct dt is calculated - correct collision
        (balls moving away from each other)
        GOOD TEST
        """

        ball1 = ball_module.Ball(mass=0.1, radius=0.1, pos=[
                                 1.0, 0.], vel=[1., 3.])
        ball2 = ball_module.Ball(mass=0.1, radius=0.1, pos=[
                                 4.0, 1.], vel=[2.0, 2.0])
        dt = ball1.time_to_collision(ball2)

        assert dt is None, "no collision expected, dt should return None. Please check time_to_collision()."



class TestTask2():
    def test_mass_exists(self, my_ball):
        assert hasattr(my_ball, '_Ball__mass')

    def test_radius_exists(self, my_ball):
        assert hasattr(my_ball, '_Ball__radius')

    def test_pos_exists(self, my_ball):
        assert hasattr(my_ball, '_Ball__pos')

    def test_pos_type(self, my_ball):
        assert isinstance(my_ball.pos(), np.ndarray)

    def test_vel_exists(self, my_ball):
        assert hasattr(my_ball, '_Ball__vel')

    def test_vel_type(self, my_ball):
        assert isinstance(my_ball.vel(), np.ndarray)

    def test_patch_exists(self, my_ball):
        assert hasattr(my_ball, '_Ball__patch')

    # --- has method checks

    def test_pos_method(self, my_ball):
        assert isinstance(my_ball.pos, MethodType)

    def test_vel_method(self, my_ball):
        assert isinstance(my_ball.vel, MethodType)

    def test_move_method(self, my_ball):
        assert isinstance(my_ball.move, MethodType)

    def test_time_to_collision_method(self, my_ball):
        assert isinstance(my_ball.time_to_collision, MethodType)

    def test_collide_method(self, my_ball):
        assert isinstance(my_ball.collide, MethodType)

    def test_get_patch_method(self, my_ball):
        assert isinstance(my_ball.get_patch, MethodType)



class TestTask3:

    # --- has attribute checks

    def test_balls_exists(self, my_simulation, return_object_var):
        balls = return_object_var(my_simulation, "balls")
        if balls is None:
            balls = return_object_var(my_simulation, "Balls")
        assert balls is not None, "balls attribute does not exist in Simulation class"

    def test_container_exists(self, my_simulation, return_object_var):
        container = return_object_var(my_simulation, "container")
        if container is None:
            container = return_object_var(my_simulation, "Container")
        assert container is not None, "container attribute does not exist in Simulation class"

    # --- has method checks

    def test_initialise_exists(self, my_simulation, return_object_method):
        assert return_object_method(
            my_ball, "initialise") is not None, "initialise() method does not exist in Simulation class"

    def test_next_collision_exists(self, my_simulation, return_object_method):
        assert return_object_method(
            my_ball, "next_collision") is not None, "next_collision() method does not exist in Simulation class"

    def test_run_exists(self, my_simulation, return_object_method):
        assert return_object_method(
            my_ball, "run") is not None, "run() method does not exist in Simulation class"


class TestTask4:

    # --- functionality checks

    def test_correct_dt(self, ball_module, my_container):
        """
        check correct dt is calculated - correct collision
        (ball collides with container)
        """

        ball = ball_module.Ball(mass=0.1, radius=0.1, pos=[
                                9.8, 0.1], vel=[-1.0, 0.])
        dt = ball.time_to_collision(my_container)

        assert np.isclose(
            dt, 19.6994949), "Collision of ball with container expected, please check time_to_collision()"

    def test_collide(self, ball_module, my_container):
        """
        check ball and container velocity after collision
        """

        ball = ball_module.Ball(mass=0.1, radius=0.1, pos=[
                                9.8, 0.1], vel=[-1.0, 0.])
        dt = ball.time_to_collision(my_container)
        ball.move(dt)
        ball.collide(container)


