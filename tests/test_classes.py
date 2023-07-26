# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 12:38:51 2023

@author: nikic
"""

import pylab as pl
from utils import *
from conftest import *

class TestBall:
    
    # --- has attribute check
    
    def test_mass_attribute_exists(self, my_ball, return_object_var):
        mass = return_object_var(my_ball, "mass")
        assert mass is not None, "mass attribute does not exist in Ball class"

    def test_radius_attribute_exists(self, my_ball, return_object_var):
        radius = return_object_var(my_ball, "radius")
        assert radius is not None, "radius attribute does not exist in Ball class"

    def test_pos_attribute_exists(self, my_ball, return_object_var):
        pos = return_object_var(my_ball, "pos")
        assert pos is not None, "position attribute does not exist in Ball class"
        assert isinstance(pos, np.ndarray), "position attribute must be of type np.ndarray"

    def test_vel_attribute_exists(self, my_ball, return_object_var):
        vel = return_object_var(my_ball, "vel")
        assert vel is not None, "velocity attribute does not exist in Ball class"
        assert isinstance(vel, np.ndarray), "velocity attribute must be of type np.ndarray"

    def test_patch_attribute_exists(self, my_ball, return_object_var):
        patch = return_object_var(my_ball, "patch")
        assert patch is not None, "patch attribute does not exist in Ball class"
        assert isinstance(patch, pl.Circle), "patch attribute must be of pl.Circle"
        
    # --- has method check

    def test_pos_method_exists(self, my_ball, return_object_method):
        assert return_object_method(my_ball, "pos") is not None, "pos() method does not exist in Ball class"

    def test_vel_method_exists(self, my_ball, return_object_method):
        assert return_object_method(my_ball, "vel") is not None, "vel() method does not exist in Ball class"

    def test_move_method_exists(self, my_ball, return_object_method):
        assert return_object_method(my_ball, "move") is not None, "move() method does not exist in Ball class"

    def test_time_to_collision_method_exists(self, my_ball, return_object_method):
        assert return_object_method(my_ball, "time_to_collision") is not None, "time_to_collision() method does not exist in Ball class"

    def test_collide_method_exists(self, my_ball, return_object_method):
        assert return_object_method(my_ball, "collide") is not None, "collide() method does not exist in Ball class"

    # --- check method functionality

    def test_time_to_collision_functionality(self, ball_module, my_container):
        
        # --- 1. balls do not collide
        ball1 = ball_module.Ball(pos=[0.1], vel=[0., 2.])
        ball2 = ball_module.Ball(pos=[0., 1.], vel=[0., 2.])
        time = ball1.time_to_collision(ball2)
        #assert almost_equal(time, some_value), "Incorrect dt obtained from time_to_collision"
                
        # --- 2. balls collide
        ball1 = ball_module.Ball(pos=[1., -1.], vel=[0., 2.])
        ball2 = ball_module.Ball(pos=[0., 1.], vel=[0., 2.])
        time = ball1.time_to_collision(ball2)
        #assert almost_equal(time, some_value), "Incorrect dt obtained from time_to_collision"
               
        # --- 3. ball and container
        ball1 = ball_module.Ball(pos=[1., -1.], vel=[0., 2.])
        time = ball1.time_to_collision(my_container)
        #assert almost_equal(time, some_value), "Incorrect dt obtained from time_to_collision"
        
    def test_collide_functionality_1(self, ball_module):
        
        ball1 = ball_module.Ball(pos=[1., -1.], vel=[0.,2.])
        ball2 = ball_module.Ball(pos=[0., 1.], vel=[0.,-2.])
        ball1.collide(ball2)
        assert ball1.pos() == [0., 0.], "Incorrect final ball position after collision"
        assert ball2.pos() == [0., 0.], "Incorrect final ball position after collision"
        assert ball1.vel() == [0., 0.], "Incorrect final ball velocity after collision"
        assert ball2.vel() == [0., 0.], "Incorrect final ball velocity after collision"
        
    def test_collide_functionality_2(self, ball_module, my_container):
        
        ball1 = Ball(pos=[1., -1.], vel=[0.,2.])
        ball1.collide(my_container)
        assert ball1.pos() == [0., 0.], "Incorrect final ball position after collision"
        assert my_container.pos() == [0., 0.], "Incorrect final container position after collision"
        assert ball1.vel() == [0., 0.], "Incorrect final ball velocity after collision"
        assert my_container.vel() == [0., 0.], "Incorrect final container velocity after collision"
        
    def test_move_functionality(self, ball_module):
        
        # dt = 0.1
        
        # ball1 = Ball(pos=[1., -1.], vel=[0.,2.])
        # ball.move(dt)
        # assert ball.pos() == some_val, "Incorrect final ball position after being moved"
        # assert ball.vel() == some_val, "Incorrect final "
        pass
        
    
class TestSimulation:
        
    # --- has attribute check
    
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
        
    def test_time_exists(self, my_simulation, return_object_var):
        time = return_object_var(my_simulation, "time")
        
    # --- check method functionality

    def test_initialise_exists(self, my_simulation, return_object_method):
        assert return_object_method(my_ball, "initialise") is not None, "initialise() method does not exist in Simulation class"
        
    def test_initialise_exists(self, my_simulation, return_object_method):
        assert return_object_method(my_ball, "next_collision") is not None, "next_collision() method does not exist in Simulation class"
        
    def test_initialise_exists(self, my_simulation, return_object_method):
        assert return_object_method(my_ball, "run") is not None, "run() method does not exist in Simulation class"
    