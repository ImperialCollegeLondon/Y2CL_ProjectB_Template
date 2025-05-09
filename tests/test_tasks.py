"""Test Tasks
Laura Hollister
Septemvber 2023
    """
import re
from pathlib import Path
from types import FunctionType, MethodType
from inspect import signature, getsource
from base64 import b64encode, b64decode
from unittest.mock import MagicMock, patch
import pytest
import numpy as np
from scipy.constants import Boltzmann
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.figure import Figure


class TestTask2():

    def test_module_exists(self, balls):
        pass

    def test_ball_class_exists(self, balls):
        assert "Ball" in vars(balls)

    def test_init_args(self, balls):
        assert {"pos", "vel", "radius", "mass"}.issubset(signature(balls.Ball).parameters.keys())

    def test_construction(self, default_ball, custom_ball):
        pass

    def test_pos_method_exists(self, balls):
        assert isinstance(balls.Ball.pos, (FunctionType, property))

    def test_vel_method_exists(self, balls):
        assert isinstance(balls.Ball.vel, (FunctionType, property))

    def test_set_vel_method_exists(self, balls):
        ball = balls.Ball
        if hasattr(ball, "set_vel"):
            assert isinstance(balls.Ball.set_vel, FunctionType)
        else:
            assert isinstance(balls.Ball.vel, property)
            assert isinstance(balls.Ball.vel.fset, FunctionType)

    def test_mass_method_exists(self, balls):
        assert isinstance(balls.Ball.mass, (FunctionType, property))

    def test_radius_method_exists(self, balls):
        assert isinstance(balls.Ball.radius, (FunctionType, property))

    def test_move_method_exists(self, balls):
        assert isinstance(balls.Ball.move, FunctionType)

    def test_pos_returns_array(self, default_ball, custom_ball):
        default_pos = default_ball.pos
        if isinstance(default_pos, MethodType):
            default_pos = default_pos()
        assert isinstance(default_pos, np.ndarray), "Default constructed Ball.pos does not return a numpy array"

        custom_pos = custom_ball.pos
        if isinstance(custom_pos, MethodType):
            custom_pos = custom_pos()
        assert isinstance(custom_pos, np.ndarray), "Custom constructed Ball.pos does not return a numpy array"

    def test_pos_correct(self, default_ball, custom_ball):
        default_pos = default_ball.pos
        if isinstance(default_pos, MethodType):
            default_pos = default_pos()
        assert np.allclose(default_pos, [0., 0.])

        custom_pos = custom_ball.pos
        if isinstance(custom_pos, MethodType):
            custom_pos = custom_pos()
        assert np.allclose(custom_pos, [1., 2.])

    def test_vel_returns_array(self, default_ball, custom_ball):
        default_vel = default_ball.vel
        if isinstance(default_vel, MethodType):
            default_vel = default_vel()
        assert isinstance(default_vel, np.ndarray), "Default constructed Ball.vel does not return a numpy array"

        custom_vel = custom_ball.vel
        if isinstance(custom_vel, MethodType):
            custom_vel = custom_vel()
        assert isinstance(custom_vel, np.ndarray), "Custom constructed Ball.vel does not return a numpy array"

    def test_vel_correct(self, default_ball, custom_ball):
        default_vel = default_ball.vel
        if isinstance(default_vel, MethodType):
            default_vel = default_vel()
        assert np.allclose(default_vel, [1., 0.])

        custom_vel = custom_ball.vel
        if isinstance(custom_vel, MethodType):
            custom_vel = custom_vel()
        assert np.allclose(custom_vel, [3., 4.])

    def test_set_vel_sets_array(self, default_ball):
        if hasattr(default_ball, "set_vel"):
            default_ball.set_vel([8, 9])
        else:
            default_ball.vel = [8, 9]

        default_vel = default_ball.vel
        if isinstance(default_vel, MethodType):
            default_vel = default_vel()
        assert isinstance(default_vel, np.ndarray)

    def test_set_vel_correct(self, default_ball):
        if hasattr(default_ball, "set_vel"):
            default_ball.set_vel([8, 9])
        else:
            default_ball.vel = [8, 9]

        default_vel = default_ball.vel
        if isinstance(default_vel, MethodType):
            default_vel = default_vel()
        assert np.allclose(default_vel, [8., 9.])

    def test_mass_type(self, default_ball, custom_ball):
        default_mass = default_ball.mass
        if isinstance(default_mass, MethodType):
            default_mass = default_mass()
        assert isinstance(default_mass, float)

        custom_mass = custom_ball.mass
        if isinstance(custom_mass, MethodType):
            custom_mass = custom_mass()
        assert isinstance(custom_mass, float)

    def test_mass_correct(self, default_ball, custom_ball):
        default_mass = default_ball.mass
        if isinstance(default_mass, MethodType):
            default_mass = default_mass()
        assert default_mass == 1.

        custom_mass = custom_ball.mass
        if isinstance(custom_mass, MethodType):
            custom_mass = custom_mass()
        assert custom_mass == 6.

    def test_radius_type(self, default_ball, custom_ball):
        default_radius = default_ball.radius
        if isinstance(default_radius, MethodType):
            default_radius = default_radius()
        assert isinstance(default_radius, float)

        custom_radius = custom_ball.radius
        if isinstance(custom_radius, MethodType):
            custom_radius = custom_radius()
        assert isinstance(custom_radius, float)

    def test_radius_correct(self, default_ball, custom_ball):
        default_radius = default_ball.radius
        if isinstance(default_radius, MethodType):
            default_radius = default_radius()
        assert default_radius == 1.

        custom_radius = custom_ball.radius
        if isinstance(custom_radius, MethodType):
            custom_radius = custom_radius()
        assert custom_radius == 5.

    def test_move_correct(self, default_ball):
        default_ball.move(3)
        pos = default_ball.pos
        if isinstance(pos, MethodType):
            pos = pos()
        assert np.allclose(pos, [3, 0])


class TestTask3():

    def test_patch_exists(self, balls):
        assert isinstance(balls.Ball.patch, (FunctionType, property))

    def test_patch_type(self, default_ball, custom_ball):
        default_patch = default_ball.patch
        if isinstance(default_patch, MethodType):
            default_patch = default_patch()
        assert isinstance(default_patch, Circle)

        custom_patch = custom_ball.patch
        if isinstance(custom_patch, MethodType):
            custom_patch = custom_patch()
        assert isinstance(custom_patch, Circle)

    def test_patch_correct(self, default_ball, custom_ball):
        default_patch = default_ball.patch
        if isinstance(default_patch, MethodType):
            default_patch = default_patch()
        assert np.allclose(default_patch.center, [0., 0])
        assert np.isclose(default_patch.radius, 1.)

        custom_patch = custom_ball.patch
        if isinstance(custom_patch, MethodType):
            custom_patch = custom_patch()
        assert np.allclose(custom_patch.center, [1., 2])
        assert np.isclose(custom_patch.radius, 5.)

    def test_patch_moves(self, default_ball, custom_ball):
        default_ball.move(3)
        default_patch = default_ball.patch
        if isinstance(default_patch, MethodType):
            default_patch = default_patch()
        assert np.allclose(default_patch.center, [3., 0])

        custom_ball.move(4)
        custom_patch = custom_ball.patch
        if isinstance(custom_patch, MethodType):
            custom_patch = custom_patch()
        assert np.allclose(custom_patch.center, [13., 18])

    def test_single_patch(self, default_ball):
        default_patch = default_ball.patch
        if isinstance(default_patch, MethodType):
            init_patch = default_patch()
            second_patch = default_patch()
            third_patch = default_patch()
            assert init_patch is second_patch
            assert init_patch is third_patch
        else:
            second_patch = default_ball.patch
            third_patch = default_ball.patch
            assert default_patch is second_patch
            assert default_patch is third_patch

    def test_move_single_patch(self, default_ball):
        default_patch = default_ball.patch
        if isinstance(default_patch, MethodType):
            default_patch = default_patch()

        default_ball.move(3)

        new_patch = default_ball.patch
        if isinstance(new_patch, MethodType):
            new_patch = new_patch()

        assert default_patch is new_patch


class TestTask4():

    def test_ttc_exists(self, balls):
        assert isinstance(balls.Ball.time_to_collision, FunctionType)

    def test_ttc_return_type(self, balls, default_ball):
        ball1 = balls.Ball(pos=[5, 0], vel=[-1, 0])
        assert isinstance(default_ball.time_to_collision(ball1), float)

        ball2 = balls.Ball(pos=[5, 0], vel=[1, 0])
        assert isinstance(default_ball.time_to_collision(ball2), type(None))

    def test_ttc_correct(self, balls, default_ball):
        ball1 = balls.Ball(pos=[5, 0], vel=[-1, 0])
        assert np.isclose(default_ball.time_to_collision(ball1), 1.5)

        ball2 = balls.Ball(pos=[5, 1], vel=[-1, 0])
        assert np.isclose(default_ball.time_to_collision(ball2), 1.6339745962155614)

        ball1 = balls.Ball(pos=[1, 1], vel=[1., 0])
        ball2 = balls.Ball(pos=[5., 4.], vel=[0, -0.75])
        assert np.isclose(ball1.time_to_collision(ball2), 2.4)

    def test_ttc_parallel(self, balls):
        ball1 = balls.Ball(pos=[0.1, 0], vel=[0., 2.])
        ball2 = balls.Ball(pos=[0., 1.], vel=[0., 2.])
        assert ball1.time_to_collision(ball2) in (None, np.inf)

    def test_ttc_going_away(self, balls):
        ball1 = balls.Ball(pos=[2, 0], vel=[1, 0])
        ball2 = balls.Ball(pos=[-3, 0], vel=[-1, 0])
        assert ball1.time_to_collision(ball2) in (None, np.inf)

    # def test_time_to_collision_con(self, ball_mod, default_con):
    #     ball1 = ball_mod.Ball(pos=[1., -1.], vel=[0., 2.])
    #     assert np.isclose(ball1.time_to_collision(default_con), 5.42468273)


class TestTask5():

    def test_collide_exists(self, balls):
        assert isinstance(balls.Ball.collide, FunctionType)

    def test_collide_correct_1D(self, balls, default_ball):
        ball = balls.Ball(pos=[5., 0], vel=[-1, 0.])
        default_ball.collide(ball)

        default_vel = default_ball.vel
        if isinstance(default_vel, MethodType):
            default_vel = default_vel()

        ball_vel = ball.vel
        if isinstance(ball_vel, MethodType):
            ball_vel = ball_vel()

        assert np.allclose(ball_vel, [1, 0.])
        assert np.allclose(default_vel, [-1, 0.])

    def test_collide_correct_2D(self, balls):
        ball1 = balls.Ball(pos=[1, 1.], vel=[1, 0])
        ball2 = balls.Ball(pos=[5, 4], vel=[0., -0.75])
        ball1.collide(ball2)

        ball1_vel = ball1.vel
        if isinstance(ball1_vel, MethodType):
            ball1_vel = ball1_vel()

        ball2_vel = ball2.vel
        if isinstance(ball2_vel, MethodType):
            ball2_vel = ball2_vel()

        assert np.allclose(ball1_vel, [0, -0.75])
        assert np.allclose(ball2_vel, [1, 0.])

    # def test_collide_functionality_2(self, ball_mod, default_con):
    #     ball1 = ball_mod.Ball(pos=[1., -1.], vel=[0.,2.])
    #     time = ball1.time_to_collision(default_con)
    #     ball1.move(time)
    #     ball1.collide(default_con)

    #     assert np.allclose(ball1.pos() ,[1., 9.84936546])
    #     assert np.allclose(default_con.pos(),[0., 0.])
    #     assert np.allclose(ball1.vel(),[-0.4019739 , -1.95918784])
    #     assert np.allclose(default_con.vel() ,[0., 0.])


class TestTask6:

    def test_container_exists(self, balls):
        assert "Container" in vars(balls)

    def test_container_args(self, balls):
        assert {"radius", "mass"}.issubset(signature(balls.Container).parameters.keys())

    def test_container_construction(self, balls, default_container):
        balls.Container(radius=11.)
        balls.Container(radius=12., mass=10000000.)

    def test_default_args(self, default_container):
        default_radius = default_container.radius
        if isinstance(default_radius, MethodType):
            default_radius = default_radius()
        assert np.isclose(default_radius, 10.)

        default_mass = default_container.mass
        if isinstance(default_mass, MethodType):
            default_mass = default_mass()
        assert np.isclose(default_mass, 10000000.)

    def test_ttc_exists(self, container_class):
        assert isinstance(container_class.time_to_collision, FunctionType)

    def test_ttc_return_type(self, default_container, default_ball):
        assert isinstance(default_container.time_to_collision(default_ball), float)

    def test_ttc_correct(self, balls, default_container, default_ball):
        assert np.isclose(default_container.time_to_collision(default_ball), 9.)

        ball = balls.Ball(pos=[3, 5.], vel=[-1, 1.])
        assert np.isclose(default_container.time_to_collision(ball), 3.9497474683058327)

    def test_collide_exists(self, container_class):
        assert isinstance(container_class.collide, FunctionType)

    def test_collide_correct(self, default_container, colliding_ball):
        default_container.collide(colliding_ball)
        vel = colliding_ball.vel
        if isinstance(vel, MethodType):
            vel = vel()
        assert np.allclose(vel, [-0.9999998,  0.])

    def test_volume_exists(self, container_class):
        assert isinstance(container_class.volume, (FunctionType, property))

    def test_volume_correct(self, container_class):
        cont1 = container_class()
        vol1 = cont1.volume
        if isinstance(vol1, MethodType):
            vol1 = vol1()
        assert np.isclose(vol1, 100*np.pi)

        cont2 = container_class(radius=5.)
        vol2 = cont2.volume
        if isinstance(vol2, MethodType):
            vol2 = vol2()
        assert np.isclose(vol2, 25*np.pi)

    def test_sa_exists(self, container_class):
        assert isinstance(container_class.surface_area, (FunctionType, property))

    def test_sa_correct(self, container_class):
        cont1 = container_class()
        sa1 = cont1.surface_area
        if isinstance(sa1, MethodType):
            sa1 = sa1()
        assert np.isclose(sa1, 20*np.pi)

        cont2 = container_class(radius=5.)
        sa2 = cont2.surface_area
        if isinstance(sa2, MethodType):
            sa2 = sa2()
        assert np.isclose(sa2, 10*np.pi)

    def test_dp_tot_exists(self, container_class):
        assert isinstance(container_class.dp_tot, (FunctionType, property))

    def test_dp_tot_correct(self, colliding_ball, default_container):
        default_container.collide(colliding_ball)
        default_container.collide(colliding_ball)
        default_container.collide(colliding_ball)
        default_container.collide(colliding_ball)
        default_container.collide(colliding_ball)
        dp_tot = default_container.dp_tot
        if isinstance(dp_tot, MethodType):
            dp_tot = dp_tot()
        assert np.isclose(dp_tot, 9.999999000000098)


class TestTask7:

    def test_simulations_exits(self, simulations):
        pass

    def test_simulation_class_exists(self, simulations):
        assert "Simulation" in vars(simulations)

    def test_simulation_not_inherited(self, simulations):
        assert len(simulations.Simulation.__bases__) == 1
        assert object in simulations.Simulation.__bases__

    def test_run_exists(self, simulations):
        assert isinstance(simulations.Simulation.run, FunctionType)

    def test_run_signature(self, simulations):
        assert {"num_collisions",
                "animate",
                "pause_time"}.issubset(signature(simulations.Simulation.run).parameters.keys())

    def test_next_collision_exists(self, simulations):
        assert isinstance(simulations.Simulation.next_collision, FunctionType)

    def test_next_collision_not_implemented(self, simulations):
        class ConcreteSimulation(simulations.Simulation):  # incase they use ABC
            def next_collision(self):
                super().next_collision()

        with pytest.raises(NotImplementedError):
            ConcreteSimulation().next_collision()

    def test_setup_figure_exists(self, simulations):
        assert isinstance(simulations.Simulation.setup_figure, FunctionType)

    # might be call to abstract some setup features to the abstract base class
    # def test_setup_figure_not_implemented(self, simulations):
    #     sim = simulations.Simulation()
    #     with pytest.raises(NotImplementedError):
    #         sim.setup_figure()


class TestTask8:

    def test_singleballsimulation_exists(self, simulations):
        assert "SingleBallSimulation" in vars(simulations)

    def test_initialisation_args(self, simulations):
        assert {"container",
                "ball"}.issubset(signature(simulations.SingleBallSimulation).parameters.keys())

    def test_initialisation(self, balls, simulations):
        c = balls.Container(radius=10.)
        b = balls.Ball(pos=[-5, 0], vel=[1, 0.], radius=1., mass=1.)
        simulations.SingleBallSimulation(container=c, ball=b)

    def test_container_exists(self, simulations):
        assert isinstance(simulations.SingleBallSimulation.container, (FunctionType, property))

    def test_container_correct(self, balls, simulations):
        c = balls.Container(radius=10.)
        b = balls.Ball(pos=[-5, 0], vel=[1, 0.], radius=1., mass=1.)
        sim = simulations.SingleBallSimulation(container=c, ball=b)

        cont = sim.container
        if isinstance(cont, MethodType):
            cont = cont()
        assert isinstance(cont, balls.Container)
        assert cont is c

        radius = cont.radius
        if isinstance(radius, MethodType):
            radius = radius()
        assert np.fabs(radius) == 10.

    def test_ball_exists(self, simulations):
        assert isinstance(simulations.SingleBallSimulation.ball, (FunctionType, property))

    def test_ball_correct(self, balls, simulations):
        c = balls.Container(radius=10.)
        b = balls.Ball(pos=[-5, 0], vel=[1, 0.], radius=1., mass=1.)
        sim = simulations.SingleBallSimulation(container=c, ball=b)

        ball = sim.ball
        if isinstance(ball, MethodType):
            ball = ball()
        assert isinstance(ball, balls.Ball)
        assert ball is b

        radius = ball.radius
        if isinstance(radius, MethodType):
            radius = radius()
        assert radius == 1.

    def test_setup_figure_exists(self, simulations):
        assert "setup_figure" in vars(simulations.SingleBallSimulation)

    # # dulplicate of task7
    # def test_setup_figure_base_raises(self, simulations_mod):
    #     sim = simulations_mod.Simulation()
    #     with pytest.raises(NotImplementedError):
    #         sim.setup_figure()

    def test_next_collision_exists(self, simulations):
        assert "next_collision" in vars(simulations.SingleBallSimulation)

    # # dulplicate of task7
    # def test_next_collision_base_raises(self, simulations_mod):
    #     sim = simulations_mod.Simulation()
    #     with pytest.raises(NotImplementedError):
    #         sim.next_collision()

    def test_next_collision_functionality(self, balls, simulations, monkeypatch):
        c = balls.Container(radius=10.)
        b = balls.Ball(pos=[-5, 0], vel=[1, 0.], radius=1., mass=1.)
        sim = simulations.SingleBallSimulation(container=c, ball=b)
        ttc_mock = MagicMock(return_value=9.)
        collide_mock = MagicMock()
        with monkeypatch.context() as m:
            m.setattr(b, "time_to_collision", ttc_mock)
            m.setattr(c, "time_to_collision", ttc_mock)
            m.setattr(b, "collide", collide_mock)
            m.setattr(c, "collide", collide_mock)
            sim.next_collision()
        ttc_mock.assert_called_once()
        collide_mock.assert_called_once()


class TestTask9:

    TASK9_DEFAULT = b'ZGVmIHRhc2s5KCk6CiAgICAiIiIKICAgIFRhc2sgOS4KCiAgICBJbiB0aGlzIGZ1bmN0aW9uLCB5b3Ugc2hvdWxkIHRlc3QgeW91ciBhbmltYXRpb24uIFRvIGRvIHRoaXMsIGNyZWF0ZSBhIGNvbnRhaW5lcgogICAgYW5kIGJhbGwgYXMgZGlyZWN0ZWQgaW4gdGhlIHByb2plY3QgYnJpZWYuIENyZWF0ZSBhIFNpbmdsZUJhbGxTaW11bGF0aW9uIG9iamVjdCBmcm9tIHRoZXNlCiAgICBhbmQgdHJ5IHJ1bm5pbmcgeW91ciBhbmltYXRpb24uIEVuc3VyZSB0aGF0IHRoaXMgZnVuY3Rpb24gcmV0dXJucyB0aGUgYmFsbHMgZmluYWwgcG9zaXRpb24gYW5kCiAgICB2ZWxvY2l0eS4KCiAgICBSZXR1cm5zOgogICAgICAgIHR1cGxlW05EQXJyYXlbbnAuZmxvYXQ2NF0sIE5EQXJyYXlbbnAuZmxvYXQ2NF1dOiBUaGUgYmFsbHMgZmluYWwgcG9zaXRpb24gYW5kIHZlbG9jaXR5CiAgICAiIiIKICAgIHJldHVybgo='

    def test_doesnt_crash(self, sbs_run_mock, task9_output, an):
        attempted = getsource(an.task9).encode('utf-8') != b64decode(TestTask9.TASK9_DEFAULT)
        assert attempted, "Task9 not attempted."

    def test_singleballsim_created(self, sbs_run_mock, task9_output):
        sbs_mock, _ = sbs_run_mock
        sbs_mock.assert_called()

    def test_run_called(self, sbs_run_mock, task9_output):
        _, run_mock = sbs_run_mock
        run_mock.assert_called_once()

    def test_run_correct(self, sbs_run20_mock, task9_output):
        pos, vel = task9_output
        assert np.allclose(pos, [-9., 0.])
        assert np.allclose(vel, [1., 0.])


class TestTask10:

    TASK10_DEFAULT = b'ZGVmIHRhc2sxMCgpOgogICAgIiIiCiAgICBUYXNrIDEwLgoKICAgIEluIHRoaXMgZnVuY3Rpb24gd2Ugc2hhbGwgdGVzdCB5b3VyIE11bHRpQmFsbFNpbXVsYXRpb24uIENyZWF0ZSBhbiBpbnN0YW5jZSBvZiB0aGlzIGNsYXNzIHVzaW5nCiAgICB0aGUgZGVmYXVsdCB2YWx1ZXMgZGVzY3JpYmVkIGluIHRoZSBwcm9qZWN0IGJyaWVmIGFuZCBydW4gdGhlIGFuaW1hdGlvbiBmb3IgNTAwIGNvbGxpc2lvbnMuCgogICAgV2F0Y2ggdGhlIHJlc3VsdGluZyBhbmltYXRpb24gY2FyZWZ1bGx5IGFuZCBtYWtlIHN1cmUgeW91IGFyZW4ndCBzZWVpbmcgZXJyb3JzIGxpa2UgYmFsbHMgc3RpY2tpbmcKICAgIHRvZ2V0aGVyIG9yIGVzY2FwaW5nIHRoZSBjb250YWluZXIuCiAgICAiIiIK'

    def test_doesnt_crash(self, mbs_run_mock, an):
        an.task10()
        attempted = getsource(an.task10).encode('utf-8') != b64decode(TestTask10.TASK10_DEFAULT)
        assert attempted, "Task10 not attempted."

    def test_multiballsim_exists(self, simulations):
        assert "MultiBallSimulation" in vars(simulations)

    def test_multiballsim_args(self, simulations):
        args = signature(simulations.MultiBallSimulation).parameters.keys()
        assert {"c_radius", "b_radius", "b_speed", "b_mass"}.issubset(args)

    def test_construction(self, simulations):
        simulations.MultiBallSimulation()
        simulations.MultiBallSimulation(c_radius=10., b_radius=1., b_speed=10., b_mass=1.)

    def test_container_exists(self, simulations):
        assert isinstance(simulations.MultiBallSimulation.container, (FunctionType, property))

    def test_container_correct(self, balls, simulations):
        sim = simulations.MultiBallSimulation()

        cont = sim.container
        if isinstance(cont, MethodType):
            cont = cont()
        assert isinstance(cont, balls.Container)

        radius = cont.radius
        if isinstance(radius, MethodType):
            radius = radius()
        assert np.fabs(radius) == 10.

    def test_balls_exists(self, simulations):
        assert isinstance(simulations.MultiBallSimulation.balls, (FunctionType, property))

    def test_balls_correct(self, balls, simulations):
        sim = simulations.MultiBallSimulation()

        balls_list = sim.balls
        if isinstance(balls_list, MethodType):
            balls_list = balls_list()
        assert isinstance(balls_list, list)

        if balls_list:
            b = balls_list[0]
            assert isinstance(b, balls.Ball)

            radius = b.radius
            if isinstance(radius, MethodType):
                radius = radius()
            assert radius == 1.

    def test_setup_figure_exists(self, simulations):
        assert "setup_figure" in vars(simulations.MultiBallSimulation)

    def test_next_collision_exists(self, simulations):
        assert "next_collision" in vars(simulations.MultiBallSimulation)

    def test_run_called(self, an, simulations, monkeypatch):
        run_mock = MagicMock()
        with monkeypatch.context() as m:
            m.setattr(simulations.MultiBallSimulation, "run", run_mock)
            if hasattr(an, "MultiBallSimulation"):
                m.setattr(an.MultiBallSimulation, "run", run_mock)
            an.task10()
        run_mock.assert_called_once()


class TestTask11:

    TASK11_DEFAULT = b'ZGVmIHRhc2sxMSgpOgogICAgIiIiCiAgICBUYXNrIDExLgoKICAgIEluIHRoaXMgZnVuY3Rpb24gd2Ugc2hhbGwgYmUgcXVhbnRpdGF0aXZlbHkgY2hlY2tpbmcgdGhhdCB0aGUgYmFsbHMgYXJlbid0IGVzY2FwaW5nIG9yIHN0aWNraW5nLgogICAgVG8gZG8gdGhpcywgY3JlYXRlIHRoZSB0d28gaGlzdG9ncmFtcyBhcyBkaXJlY3RlZCBpbiB0aGUgcHJvamVjdCBzY3JpcHQuIEVuc3VyZSB0aGF0IHRoZXNlIHR3bwogICAgaGlzdG9ncmFtIGZpZ3VyZXMgYXJlIHJldHVybmVkLgoKICAgIFJldHVybnM6CiAgICAgICAgdHVwbGVbRmlndXJlLCBGaXJndXJlXTogVGhlIGhpc3RvZ3JhbXMgKGRpc3RhbmNlIGZyb20gY2VudHJlLCBpbnRlci1iYWxsIHNwYWNpbmcpLgogICAgIiIiCiAgICByZXR1cm4K'

    def test_doesnt_crash(self, mbs_run_mock, task11_output, an):
        attempted = getsource(an.task11).encode('utf-8') != b64decode(TestTask11.TASK11_DEFAULT)
        assert attempted, "Task11 not attempted."

    def test_running_simulation(self, mbs_run_mock, task11_output):
        mbs_mock, run_mock = mbs_run_mock
        mbs_mock.assert_called()
        run_mock.assert_called()

    def test_output(self, mbs_run_mock, task11_output):
        hist1, hist2 = task11_output
        assert isinstance(hist1, Figure)
        assert isinstance(hist2, Figure)

    def test_creating_hist(self, hist_mock, mbs_run_mock, task11_output, an):
        assert re.search(r"hist\(", getsource(an.task11)) is not None
        assert hist_mock[0].called or hist_mock[1].called


class TestTask12:

    TASK12_DEFAULT = b'ZGVmIHRhc2sxMigpOgogICAgIiIiCiAgICBUYXNrIDEyLgoKICAgIEluIHRoaXMgZnVuY3Rpb24gd2Ugc2hhbGwgY2hlY2sgdGhhdCB0aGUgZnVuZGFtZW50YWwgcXVhbnRpdGllcyBvZiBlbmVyZ3kgYW5kIG1vbWVudHVtIGFyZSBjb25zZXJ2ZWQuCiAgICBBZGRpdGlvbmFsbHkgd2Ugc2hhbGwgaW52ZXN0aWdhdGUgdGhlIHByZXNzdXJlIGV2b2x1dGlvbiBvZiB0aGUgc3lzdGVtLiBFbnN1cmUgdGhhdCB0aGUgNCBmaWd1cmVzCiAgICBvdXRsaW5lZCBpbiB0aGUgcHJvamVjdCBzY3JpcHQgYXJlIHJldHVybmVkLgoKICAgIFJldHVybnM6CiAgICAgICAgdHVwbGVbRmlndXJlLCBGaWd1cmUsIEZpZ3VyZSwgRmlndXJlXTogbWF0cGxvdGxpYiBGaWd1cmVzIG9mIHRoZSBLRSwgbW9tZW50dW1feCwgbW9tZW50dW1feSByYXRpb3MKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBhcyB3ZWxsIGFzIHByZXNzdXJlIGV2b2x1dGlvbi4KICAgICIiIgogICAgcmV0dXJuCg=='

    def test_doesnt_crash(self, mbs_run_mock, task12_output, an):
        attempted = getsource(an.task12).encode('utf-8') != b64decode(TestTask12.TASK12_DEFAULT)
        assert attempted, "Task12 not attempted."

    def test_output(self, mbs_run_mock, task12_output):
        assert len(task12_output) == 4
        assert isinstance(task12_output[0], Figure)
        assert isinstance(task12_output[1], Figure)
        assert isinstance(task12_output[2], Figure)
        assert isinstance(task12_output[3], Figure)

    def test_ke_exists(self, simulations):
        assert isinstance(simulations.MultiBallSimulation.kinetic_energy, (FunctionType, property))

    def test_ke_correct(self, simulations):
        sim = simulations.MultiBallSimulation()
        ke_tot = 0.
        balls = sim.balls
        if isinstance(balls, MethodType):
            balls = balls()

        for ball in balls:
            vel = ball.vel
            if isinstance(vel, MethodType):
                vel = vel()
            mass = ball.mass
            if isinstance(mass, MethodType):
                mass = mass()
            ke_tot += 0.5 * mass * np.dot(vel, vel)

        sim_ke = sim.kinetic_energy
        if isinstance(sim_ke, MethodType):
            sim_ke = sim_ke()
        assert np.isclose(ke_tot, sim_ke)

    def test_time_exists(self, simulations):
        assert isinstance(simulations.MultiBallSimulation.time, (FunctionType, property))

    def test_time_correct(self, balls, simulations, monkeypatch):

        time_tot = set()
        original_move = balls.Ball.move

        def move_wrapper(self, dt):
            nonlocal time_tot
            time_tot.add(dt)
            original_move(self, dt)

        with monkeypatch.context() as m:
            m.setattr(balls.Ball, "move", move_wrapper)
            if "Ball" in vars(simulations):
                m.setattr(simulations.Ball, "move", move_wrapper)

            sim = simulations.MultiBallSimulation()
            sim.run(10)

        assert len(time_tot) == 10, "Incorrect number of collisions."

        time = sim.time
        if isinstance(time, MethodType):
            time = time()
        assert np.isclose(sum(time_tot), time)

    def test_momentum_exists(self, simulations):
        assert isinstance(simulations.MultiBallSimulation.momentum, (FunctionType, property))

    def test_momentum_correct(self, simulations):
        sim = simulations.MultiBallSimulation()

        mom_tot = np.zeros(2, dtype=np.float64)
        balls = sim.balls
        if isinstance(balls, MethodType):
            balls = balls()

        for ball in balls:
            vel = ball.vel
            if isinstance(vel, MethodType):
                vel = vel()
            mass = ball.mass
            if isinstance(mass, MethodType):
                mass = mass()
            mom_tot += mass * vel

        sim_mom = sim.momentum
        if isinstance(sim_mom, MethodType):
            sim_mom = sim_mom()
        assert np.allclose(mom_tot, sim_mom)

    def test_pressure_exists(self, simulations):
        assert isinstance(simulations.MultiBallSimulation.pressure, (FunctionType, property))

    def test_pressure_correct(self, simulations):
        sim = simulations.MultiBallSimulation()
        sim.run(10)

        cont = sim.container
        if isinstance(cont, MethodType):
            cont = cont()

        sa = cont.surface_area
        if isinstance(sa, MethodType):
            sa = sa()

        dp_tot = cont.dp_tot
        if isinstance(dp_tot, MethodType):
            dp_tot = dp_tot()

        time = sim.time
        if isinstance(time, MethodType):
            time = time()

        pressure = sim.pressure
        if isinstance(pressure, MethodType):
            pressure = pressure()
        assert np.isclose(dp_tot / (time * sa), pressure)

    # @pytest.mark.skip
    # def test_test12_plots(self):
    #     pass


class TestTask13:

    TASK13_DEFAULT = b'ZGVmIHRhc2sxMygpOgogICAgIiIiCiAgICBUYXNrIDEzLgoKICAgIEluIHRoaXMgZnVuY3Rpb24gd2UgaW52ZXN0aWdhdGUgaG93IHdlbGwgb3VyIHNpbXVsYXRpb24gcmVwcm9kdWNlcyB0aGUgZGlzdHJpYnV0aW9ucyBvZiB0aGUgSUdMLgogICAgQ3JlYXRlIHRoZSAzIGZpZ3VyZXMgZGlyZWN0ZWQgYnkgdGhlIHByb2plY3Qgc2NyaXB0LCBuYW1lbHk6CiAgICAxKSBQVCBwbG90CiAgICAyKSBQViBwbG90CiAgICAzKSBQTiBwbG90CiAgICBFbnN1cmUgdGhhdCB0aGlzIGZ1bmN0aW9uIHJldHVybnMgdGhlIHRocmVlIG1hdHBsb3RsaWIgZmlndXJlcy4KCiAgICBSZXR1cm5zOgogICAgICAgIHR1cGxlW0ZpZ3VyZSwgRmlndXJlLCBGaWd1cmVdOiBUaGUgMyByZXF1ZXN0ZWQgZmlndXJlczogKFBULCBQViwgUE4pCiAgICAiIiIKICAgIHJldHVybgo='

    def test_doesnt_crash(self, mbs_run_mock, task13_output, an):
        attempted = getsource(an.task13).encode('utf-8') != b64decode(TestTask13.TASK13_DEFAULT)
        assert attempted, "Task13 not attempted."

    def test_output(self, mbs_run_mock, task13_output):
        assert len(task13_output) == 3
        assert isinstance(task13_output[0], Figure)
        assert isinstance(task13_output[1], Figure)
        assert isinstance(task13_output[2], Figure)

    def test_t_equipartition_exists(self, simulations):
        assert isinstance(simulations.MultiBallSimulation.t_equipartition, (FunctionType, property))

    def test_t_equipartition_correct(self, simulations):
        sim = simulations.MultiBallSimulation()
        sim.run(10)

        ke_tot = sim.kinetic_energy
        if isinstance(ke_tot, MethodType):
            ke_tot = ke_tot()

        balls = sim.balls
        if isinstance(balls, MethodType):
            balls = balls()
        nballs = len(balls)

        t_equi = sim.t_equipartition
        if isinstance(t_equi, MethodType):
            t_equi = t_equi()
        # assert ke_tot / (Boltzmann * nballs) == t_equi
        assert np.any(np.isclose(t_equi, [ke_tot / nballs, ke_tot / (Boltzmann * nballs)]))
        # assert t_equi in {ke_tot / nballs, ke_tot / (Boltzmann * nballs)}

    # @pytest.mark.skip
    # def test_task13_plots(self):
    #     pass


class TestTask14:

    TASK14_DEFAULT = b'ZGVmIHRhc2sxNCgpOgogICAgIiIiCiAgICBUYXNrIDE0LgoKICAgIEluIHRoaXMgZnVuY3Rpb24gd2Ugc2hhbGwgYmUgbG9va2luZyBhdCB0aGUgZGl2ZXJnZW5jZSBvZiBvdXIgc2ltdWxhdGlvbiBmcm9tIHRoZSBJR0wuIFdlIHNoYWxsCiAgICBxdWFudGlmeSB0aGUgYmFsbCByYWRpaSBkZXBlbmRlbmNlIG9mIHRoaXMgZGl2ZXJnZW5jZSBieSBwbG90dGluZyB0aGUgdGVtcGVyYXR1cmUgcmF0aW8gZGVmaW5lZCBpbgogICAgdGhlIHByb2plY3QgYnJpZWYuCgogICAgUmV0dXJuczoKICAgICAgICBGaWd1cmU6IFRoZSB0ZW1wZXJhdHVyZSByYXRpbyBmaWd1cmUuCiAgICAiIiIKICAgIHJldHVybgo='

    def test_doesnt_crash(self, mbs_run_mock, task14_output, an):
        attempted = getsource(an.task14).encode('utf-8') != b64decode(TestTask14.TASK14_DEFAULT)
        assert attempted, "Task14 not attempted."

    def test_output(self, mbs_run_mock, task14_output):
        assert isinstance(task14_output, Figure)

    def test_t_ideal_exists(self, simulations):
        assert isinstance(simulations.MultiBallSimulation.t_ideal, (FunctionType, property))

    def test_t_ideal_correct(self, simulations):
        sim = simulations.MultiBallSimulation()
        sim.run(10)

        balls = sim.balls
        if isinstance(balls, MethodType):
            balls = balls()
        nballs = len(balls)

        t_ideal = sim.t_ideal
        if isinstance(t_ideal, MethodType):
            t_ideal = t_ideal()

        pressure = sim.pressure
        if isinstance(pressure, MethodType):
            pressure = pressure()

        cont = sim.container
        if isinstance(cont, MethodType):
            cont = cont()
        volume = cont.volume
        if isinstance(volume, MethodType):
            volume = volume()

        assert np.any(np.isclose(t_ideal, [pressure * volume / nballs, pressure * volume / (Boltzmann * nballs)]))
        # assert t_ideal in {pressure * volume / nballs, pressure * volume / (Boltzmann * nballs)}

    # @pytest.mark.skip
    # def test_task14_plots(self):
    #     pass


class TestTask15:

    TASK15_DEFAULT = b'ZGVmIHRhc2sxNSgpOgogICAgIiIiCiAgICBUYXNrIDE1LgoKICAgIEluIHRoaXMgZnVuY3Rpb24gd2Ugc2hhbGwgcGxvdCBhIGhpc3RvZ3JhbSB0byBpbnZlc3RpZ2F0ZSBob3cgdGhlIHNwZWVkcyBvZiB0aGUgYmFsbHMgZXZvbHZlIGZyb20gdGhlIGluaXRpYWwKICAgIHZhbHVlLiBXZSBzaGFsbCB0aGVuIGNvbXBhcmUgdGhpcyB0byB0aGUgTWF4d2VsbC1Cb2x0em1hbm4gZGlzdHJpYnV0aW9uLiBFbnN1cmUgdGhhdCB0aGlzIGZ1bmN0aW9uIHJldHVybnMKICAgIHRoZSBjcmVhdGVkIGhpc3RvZ3JhbS4KCiAgICBSZXR1cm5zOgogICAgICAgIEZpZ3VyZTogVGhlIHNwZWVkIGhpc3RvZ3JhbS4KICAgICIiIgogICAgcmV0dXJuCg=='

    def test_doesnt_crash(self, mbs_run_mock, task15_output, an):
        attempted = getsource(an.task15).encode('utf-8') != b64decode(TestTask15.TASK15_DEFAULT)
        assert attempted, "Task15 not attempted."

    def test_output(self, mbs_run_mock, task15_output):
        assert isinstance(task15_output, Figure)

    def test_speeds_exists(self, simulations):
        assert isinstance(simulations.MultiBallSimulation.speeds, (FunctionType, property))

    def test_speeds_correct(self, simulations):
        sim = simulations.MultiBallSimulation()
        sim.run(5)

        balls = sim.balls
        if isinstance(balls, MethodType):
            balls = balls()

        b_speeds = []
        for ball in balls:
            vel = ball.vel
            if isinstance(vel, MethodType):
                vel = vel()
            b_speeds.append(np.linalg.norm(vel))

        sim_speeds = sim.speeds
        if isinstance(sim_speeds, MethodType):
            sim_speeds = sim_speeds()

        assert np.allclose(sim_speeds, b_speeds)

    def test_maxwell_exists(self, physics):
        assert isinstance(physics.maxwell, FunctionType)

    def test_maxwell_args(self, physics):
        args = signature(physics.maxwell).parameters.keys()
        assert {"speed", "kbt", "mass"}.issubset(args)

    def test_maxwell_correct(self, physics):
        mass = 2.
        speed = 0.5
        kbt = Boltzmann * 2000

        mb_prob = mass * speed * np.exp(-mass * speed * speed / (2. * kbt)) / kbt
        assert np.isclose(mb_prob, physics.maxwell(speed=speed, kbt=kbt, mass=mass))

    # @pytest.mark.skip
    # def test_task15_plots(self):
    #     pass


class TestTask16:

    TASK16_DEFAULT = b'ZGVmIHRhc2sxNigpOgogICAgIiIiCiAgICBUYXNrIDE2LgoKICAgIEluIHRoaXMgZnVuY3Rpb24gd2Ugc2hhbGwgYWxzbyBiZSBsb29raW5nIGF0IHRoZSBkaXZlcmdlbmNlIG9mIG91ciBzaW11bGF0aW9uIGZyb20gdGhlIElHTC4gV2Ugc2hhbGwKICAgIHF1YW50aWZ5IHRoZSBiYWxsIHJhZGlpIGRlcGVuZGVuY2Ugb2YgdGhpcyBkaXZlcmdlbmNlIGJ5IHBsb3R0aW5nIHRoZSB0ZW1wZXJhdHVyZSByYXRpbwogICAgYW5kIHZvbHVtZSBmcmFjdGlvbiBkZWZpbmVkIGluIHRoZSBwcm9qZWN0IGJyaWVmLiBXZSBzaGFsbCBmaXQgdGhpcyB0ZW1wZXJhdHVyZSByYXRpbyBiZWZvcmUKICAgIHBsb3R0aW5nIHRoZSBWRFcgYiBwYXJhbWV0ZXJzIHJhZGlpIGRlcGVuZGVuY2UuCgogICAgUmV0dXJuczoKICAgICAgICB0dXBsZVtGaWd1cmUsIEZpZ3VyZV06IFRoZSByYXRpbyBmaWd1cmUgYW5kIGIgcGFyYW1ldGVyIGZpZ3VyZS4KICAgICIiIgogICAgcmV0dXJuCg=='

    def test_doesnt_crash(self, mbs_run_mock, task16_output, an):
        attempted = getsource(an.task16).encode('utf-8') != b64decode(TestTask16.TASK16_DEFAULT)
        assert attempted, "Task16 not attempted."

    def test_output(self, mbs_run_mock, task16_output):
        assert len(task16_output) == 2
        assert isinstance(task16_output[0], Figure)
        assert isinstance(task16_output[1], Figure)

    def test_multiple_sims_created(self, mbs_run_mock, task16_output):
        mbs_mock, _ = mbs_run_mock
        assert mbs_mock.call_count > 3

    def test_curve_fit_called(self, curve_fit_mock, mbs_run_mock, task16_output):
        assert curve_fit_mock.called


class TestTask17:

    TASK17_DEFAULT = b'ZGVmIHRhc2sxNygpOgogICAgIiIiCiAgICBUYXNrIDE3LgoKICAgIEluIHRoaXMgZnVuY3Rpb24gd2Ugc2hhbGwgcnVuIGEgQnJvd25pYW4gbW90aW9uIHNpbXVsYXRpb24gYW5kIHBsb3QgdGhlIHJlc3VsdGluZyB0cmFqZWN0b3J5IG9mIHRoZSAnYmlnJyBiYWxsLgogICAgIiIiCg=='

    def test_doesnt_crash(self, bms_run_mock, task17_output, an):
        attempted = getsource(an.task17).encode('utf-8') != b64decode(TestTask17.TASK17_DEFAULT)
        assert attempted, "Task17 not attempted."

    def test_browniansimulation_exists(self, simulations):
        assert "BrownianSimulation" in vars(simulations)

    def test_browniansimulation_created(self, bms_run_mock, task17_output):
        bms_mock, _ = bms_run_mock
        assert bms_mock.called

    def test_bbpositions_exists(self, simulations):
        assert isinstance(simulations.BrownianSimulation.bb_positions, (FunctionType, property))

    def test_initialisation_args(self, simulations):
        assert {"bb_radius", "bb_mass"}.issubset(signature(simulations.BrownianSimulation).parameters.keys())

    def test_defaults_init_args(self, simulations):
        params = signature(simulations.BrownianSimulation).parameters
        assert params['bb_radius'].default == 2.
        assert params['bb_mass'].default == 10.
