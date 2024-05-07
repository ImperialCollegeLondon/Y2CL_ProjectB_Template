"""Test Tasks
Laura Hollister
Septemvber 2023
    """
import re
from pathlib import Path
from types import FunctionType, MethodType, NoneType
from inspect import signature, getsource
from unittest.mock import MagicMock
import pytest
import numpy as np
from scipy.constants import Boltzmann
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.figure import Figure


class TestTask2():

    def test_module_exists(self, balls_mod):
        pass

    def test_ball_class_exists(self, balls_mod):
        assert "Ball" in vars(balls_mod)

    def test_init_args(self, balls_mod):
        assert {"pos", "vel", "radius", "mass"}.issubset(signature(balls_mod.Ball).parameters.keys())

    def test_construction(self, default_ball, custom_ball):
        pass

    def test_pos_method_exists(self, balls_mod):
        assert isinstance(balls_mod.Ball.pos, (FunctionType, property))

    def test_vel_method_exists(self, balls_mod):
        assert isinstance(balls_mod.Ball.vel, (FunctionType, property))

    def test_set_vel_method_exists(self, balls_mod):
        ball = balls_mod.Ball
        if hasattr(ball, "set_vel"):
            assert isinstance(balls_mod.Ball.set_vel, FunctionType)
        else:
            assert isinstance(balls_mod.Ball.vel, property)
            assert isinstance(balls_mod.Ball.vel.fset, FunctionType)

    def test_mass_method_exists(self, balls_mod):
        assert isinstance(balls_mod.Ball.mass, (FunctionType, property))

    def test_radius_method_exists(self, balls_mod):
        assert isinstance(balls_mod.Ball.radius, (FunctionType, property))

    def test_move_method_exists(self, balls_mod):
        assert isinstance(balls_mod.Ball.move, FunctionType)

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

    def test_patch_exists(self, balls_mod):
        assert isinstance(balls_mod.Ball.patch, (FunctionType, property))

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


class TestTask4():

    def test_ttc_exists(self, balls_mod):
        assert isinstance(balls_mod.Ball.time_to_collision, FunctionType)

    def test_ttc_return_type(self, balls_mod, default_ball):
        ball1 = balls_mod.Ball(pos=[5, 0], vel=[-1, 0])
        assert isinstance(default_ball.time_to_collision(ball1), float)

        ball2 = balls_mod.Ball(pos=[5, 0], vel=[1, 0])
        assert isinstance(default_ball.time_to_collision(ball2), NoneType)

    def test_ttc_correct(self, balls_mod, default_ball):
        ball1 = balls_mod.Ball(pos=[5, 0], vel=[-1, 0])
        assert np.isclose(default_ball.time_to_collision(ball1), 1.5)

        ball2 = balls_mod.Ball(pos=[5, 1], vel=[-1, 0])
        assert np.isclose(default_ball.time_to_collision(ball2), 1.6339745962155614)

        ball1 = balls_mod.Ball(pos=[1, 1], vel=[1., 0])
        ball2 = balls_mod.Ball(pos=[5., 4.], vel=[0, -0.75])
        assert np.isclose(ball1.time_to_collision(ball2), 2.4)

    def test_ttc_parallel(self, balls_mod):
        ball1 = balls_mod.Ball(pos=[0.1, 0], vel=[0., 2.])
        ball2 = balls_mod.Ball(pos=[0., 1.], vel=[0., 2.])
        assert ball1.time_to_collision(ball2) in (None, np.inf)

    def test_ttc_going_away(self, balls_mod):
        ball1 = balls_mod.Ball(pos=[2, 0], vel=[1, 0])
        ball2 = balls_mod.Ball(pos=[-3, 0], vel=[-1, 0])
        assert ball1.time_to_collision(ball2) in (None, np.inf)

    # def test_time_to_collision_con(self, ball_mod, default_con):
    #     ball1 = ball_mod.Ball(pos=[1., -1.], vel=[0., 2.])
    #     assert np.isclose(ball1.time_to_collision(default_con), 5.42468273)


class TestTask5():

    def test_collide_exists(self, balls_mod):
        assert isinstance(balls_mod.Ball.collide, FunctionType)

    def test_collide_correct_1D(self, balls_mod, default_ball):
        ball = balls_mod.Ball(pos=[5., 0], vel=[-1, 0.])
        default_ball.collide(ball)

        default_vel = default_ball.vel
        if isinstance(default_vel, MethodType):
            default_vel = default_vel()

        ball_vel = ball.vel
        if isinstance(ball_vel, MethodType):
            ball_vel = ball_vel()

        assert np.allclose(ball_vel, [1, 0.])
        assert np.allclose(default_vel, [-1, 0.])

    def test_collide_correct_2D(self, balls_mod):
        ball1 = balls_mod.Ball(pos=[1, 1.], vel=[1, 0])
        ball2 = balls_mod.Ball(pos=[5, 4], vel=[0., -0.75])
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

    def test_container_exists(self, balls_mod):
        assert "Container" in vars(balls_mod)

    def test_container_args(self, balls_mod):
        assert {"radius", "mass"}.issubset(signature(balls_mod.Container).parameters.keys())

    def test_container_construction(self, balls_mod, default_container):
        balls_mod.Container(radius=11.)
        balls_mod.Container(radius=12., mass=10000000.)

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

    def test_ttc_correct(self, balls_mod, default_container, default_ball):
        assert np.isclose(default_container.time_to_collision(default_ball), 9.)

        ball = balls_mod.Ball(pos=[3, 5.], vel=[-1, 1.])
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

    def test_simulations_exits(self, simulations_mod):
        pass

    def test_simulation_class_exists(self, simulations_mod):
        assert "Simulation" in vars(simulations_mod)

    def test_simulation_not_inherited(self, simulations_mod):
        assert len(simulations_mod.Simulation.__bases__) == 1
        assert object in simulations_mod.Simulation.__bases__

    def test_run_exists(self, simulations_mod):
        assert isinstance(simulations_mod.Simulation.run, FunctionType)

    def test_run_signature(self, simulations_mod):
        assert {"num_collisions",
                "animate",
                "pause_time"}.issubset(signature(simulations_mod.Simulation.run).parameters.keys())

    def test_next_collision_exists(self, simulations_mod):
        assert isinstance(simulations_mod.Simulation.next_collision, FunctionType)

    def test_next_collision_not_implemented(self, simulations_mod):
        sim = simulations_mod.Simulation()
        with pytest.raises(NotImplementedError):
            sim.next_collision()

    def test_setup_figure_exists(self, simulations_mod):
        assert isinstance(simulations_mod.Simulation.setup_figure, FunctionType)

    def test_setup_figure_not_implemented(self, simulations_mod):
        sim = simulations_mod.Simulation()
        with pytest.raises(NotImplementedError):
            sim.setup_figure()


class TestTask8:

    def test_singleballsimulation_exists(self, simulations_mod):
        assert "SingleBallSimulation" in vars(simulations_mod)

    def test_initialisation_args(self, simulations_mod):
        assert {"container",
                "ball"}.issubset(signature(simulations_mod.SingleBallSimulation).parameters.keys())

    def test_initialisation(self, balls_mod, simulations_mod):
        c = balls_mod.Container(radius=10.)
        b = balls_mod.Ball(pos=[-5, 0], vel=[1, 0.], radius=1., mass=1.)
        simulations_mod.SingleBallSimulation(container=c, ball=b)

    def test_container_exists(self, simulations_mod):
        assert isinstance(simulations_mod.SingleBallSimulation.container, (FunctionType, property))

    def test_container_correct(self, balls_mod, simulations_mod):
        c = balls_mod.Container(radius=10.)
        b = balls_mod.Ball(pos=[-5, 0], vel=[1, 0.], radius=1., mass=1.)
        sim = simulations_mod.SingleBallSimulation(container=c, ball=b)

        cont = sim.container
        if isinstance(cont, MethodType):
            cont = cont()
        assert isinstance(cont, balls_mod.Container)
        assert cont is c

        radius = cont.radius
        if isinstance(radius, MethodType):
            radius = radius()
        assert np.fabs(radius) == 10.

    def test_ball_exists(self, simulations_mod):
        assert isinstance(simulations_mod.SingleBallSimulation.ball, (FunctionType, property))

    def test_ball_correct(self, balls_mod, simulations_mod):
        c = balls_mod.Container(radius=10.)
        b = balls_mod.Ball(pos=[-5, 0], vel=[1, 0.], radius=1., mass=1.)
        sim = simulations_mod.SingleBallSimulation(container=c, ball=b)

        ball = sim.ball
        if isinstance(ball, MethodType):
            ball = ball()
        assert isinstance(ball, balls_mod.Ball)
        assert ball is b

        radius = ball.radius
        if isinstance(radius, MethodType):
            radius = radius()
        assert radius == 1.

    def test_setup_figure_exists(self, simulations_mod):
        assert "setup_figure" in vars(simulations_mod.SingleBallSimulation)

    # # dulplicate of task7
    # def test_setup_figure_base_raises(self, simulations_mod):
    #     sim = simulations_mod.Simulation()
    #     with pytest.raises(NotImplementedError):
    #         sim.setup_figure()

    def test_next_collision_exists(self, simulations_mod):
        assert "next_collision" in vars(simulations_mod.SingleBallSimulation)

    # # dulplicate of task7
    # def test_next_collision_base_raises(self, simulations_mod):
    #     sim = simulations_mod.Simulation()
    #     with pytest.raises(NotImplementedError):
    #         sim.next_collision()

    def test_next_collision_functionality(self, balls_mod, simulations_mod, monkeypatch):
        c = balls_mod.Container(radius=10.)
        b = balls_mod.Ball(pos=[-5, 0], vel=[1, 0.], radius=1., mass=1.)
        sim = simulations_mod.SingleBallSimulation(container=c, ball=b)
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

    def test_run_called(self, an, simulations_mod, monkeypatch):
        run_mock = MagicMock()
        with monkeypatch.context() as m:
            m.setattr(simulations_mod.SingleBallSimulation, "run", run_mock)
            if hasattr(an, "SingleBallSimulation"):
                m.setattr(an.SingleBallSimulation, "run", run_mock)
            an.task9()
        run_mock.assert_called_once()

    def test_run_correct(self, an, monkeypatch):
        show_mock = MagicMock()
        with monkeypatch.context() as m:
            m.setattr(an.plt, "show", show_mock)
            pos, vel = an.task9()
        assert np.allclose(pos, [-9., 0.])
        assert np.allclose(vel, [1., 0.])


class TestTask10:

    def test_multiballsim_exists(self, simulations_mod):
        assert "MultiBallSimulation" in vars(simulations_mod)

    def test_multiballsim_args(self, simulations_mod):
        args = signature(simulations_mod.MultiBallSimulation).parameters.keys()
        assert {"c_radius", "b_radius", "b_speed", "b_mass"}.issubset(args)

    def test_construction(self, simulations_mod):
        simulations_mod.MultiBallSimulation()
        simulations_mod.MultiBallSimulation(c_radius=10., b_radius=1., b_speed=10., b_mass=1.)

    def test_container_exists(self, simulations_mod):
        assert isinstance(simulations_mod.MultiBallSimulation.container, (FunctionType, property))

    def test_container_correct(self, balls_mod, simulations_mod):
        sim = simulations_mod.MultiBallSimulation()

        cont = sim.container
        if isinstance(cont, MethodType):
            cont = cont()
        assert isinstance(cont, balls_mod.Container)

        radius = cont.radius
        if isinstance(radius, MethodType):
            radius = radius()
        assert np.fabs(radius) == 10.

    def test_balls_exists(self, simulations_mod):
        assert isinstance(simulations_mod.MultiBallSimulation.balls, (FunctionType, property))

    def test_balls_correct(self, balls_mod, simulations_mod):
        sim = simulations_mod.MultiBallSimulation()

        balls = sim.balls
        if isinstance(balls, MethodType):
            balls = balls()
        assert isinstance(balls, list)

        if balls:
            b = balls[0]
            assert isinstance(b, balls_mod.Ball)

            radius = b.radius
            if isinstance(radius, MethodType):
                radius = radius()
            assert radius == 1.

    def test_setup_figure_exists(self, simulations_mod):
        assert "setup_figure" in vars(simulations_mod.MultiBallSimulation)

    def test_next_collision_exists(self, simulations_mod):
        assert "next_collision" in vars(simulations_mod.MultiBallSimulation)

    def test_run_called(self, an, simulations_mod, monkeypatch):
        run_mock = MagicMock()
        with monkeypatch.context() as m:
            m.setattr(simulations_mod.MultiBallSimulation, "run", run_mock)
            if hasattr(an, "MultiBallSimulation"):
                m.setattr(an.MultiBallSimulation, "run", run_mock)
            an.task10()
        run_mock.assert_called_once()


class TestTask11:

    def test_running_simulation(self, an):
        assert re.search(r"MultiBallSimulation", getsource(an.task11)) is not None
        assert re.search(r"\.run", getsource(an.task11))

    def test_creating_hist(self, an):
        assert re.search(r"hist\(", getsource(an.task11)) is not None


class TestTask12:

    def test_ke_exists(self, simulations_mod):
        assert isinstance(simulations_mod.MultiBallSimulation.kinetic_energy, (FunctionType, property))

    def test_ke_correct(self, simulations_mod):
        sim = simulations_mod.MultiBallSimulation()
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

    def test_time_exists(self, simulations_mod):
        assert isinstance(simulations_mod.MultiBallSimulation.time, (FunctionType, property))

    def test_time_correct(self, balls_mod, simulations_mod, monkeypatch):

        time_tot = set()
        original_move = balls_mod.Ball.move

        def move_wrapper(self, dt):
            nonlocal time_tot
            time_tot.add(dt)
            original_move(self, dt)

        with monkeypatch.context() as m:
            m.setattr(balls_mod.Ball, "move", move_wrapper)
            if "Ball" in vars(simulations_mod):
                m.setattr(simulations_mod.Ball, "move", move_wrapper)

            sim = simulations_mod.MultiBallSimulation()
            sim.run(10)

        assert len(time_tot) == 10, "Incorrect number of collisions."

        time = sim.time
        if isinstance(time, MethodType):
            time = time()
        assert np.isclose(sum(time_tot), time)

    def test_momentum_exists(self, simulations_mod):
        assert isinstance(simulations_mod.MultiBallSimulation.momentum, (FunctionType, property))

    def test_momentum_correct(self, simulations_mod):
        sim = simulations_mod.MultiBallSimulation()

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

    def test_pressure_exists(self, simulations_mod):
        assert isinstance(simulations_mod.MultiBallSimulation.pressure, (FunctionType, property))

    def test_pressure_correct(self, simulations_mod):
        sim = simulations_mod.MultiBallSimulation()
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

    def test_t_equipartition_exists(self, simulations_mod):
        assert isinstance(simulations_mod.MultiBallSimulation.t_equipartition, (FunctionType, property))

    def test_t_equipartition_correct(self, simulations_mod):
        sim = simulations_mod.MultiBallSimulation()
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

    def test_t_ideal_exists(self, simulations_mod):
        assert isinstance(simulations_mod.MultiBallSimulation.t_ideal, (FunctionType, property))

    def test_t_ideal_correct(self, simulations_mod):
        sim = simulations_mod.MultiBallSimulation()
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

    def test_speeds_exists(self, simulations_mod):
        assert isinstance(simulations_mod.MultiBallSimulation.speeds, (FunctionType, property))

    def test_speeds_correct(self, simulations_mod):
        sim = simulations_mod.MultiBallSimulation()
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

    def test_maxwell_exists(self, physics_mod):
        assert isinstance(physics_mod.maxwell, FunctionType)

    def test_maxwell_args(self, physics_mod):
        args = signature(physics_mod.maxwell).parameters.keys()
        assert {"speed", "kbt", "mass"}.issubset(args)

    def test_maxwell_correct(self, physics_mod):
        mass = 2.
        speed = 0.5
        kbt = Boltzmann * 2000

        mb_prob = mass * speed * np.exp(-mass * speed * speed / (2. * kbt)) / kbt
        assert np.isclose(mb_prob, physics_mod.maxwell(speed=speed, kbt=kbt, mass=mass))

    # @pytest.mark.skip
    # def test_task15_plots(self):
    #     pass
