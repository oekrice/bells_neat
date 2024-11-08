"""
Test the performance of the best genome produced by evolve-feedforward.py.
"""

import os
import pickle

import neat
#from cart_pole import CartPole, discrete_actuator_force
from main import run_bell, discrete_actuator_force, continuous_actuator_force, probably_actuator_force

import random
from random import uniform, gauss

import matplotlib.pyplot as plt

import numpy as np
#from movie import make_movie

# load the winner
with open('winner_bell', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config_bell')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

#c is the best loaded genome. Easy to miss that...

net = neat.nn.FeedForwardNetwork.create(c, config)
sim = run_bell()

print()
print("Initial conditions:")
print("    angle = {0:.4f}".format(sim.bell.bell_angle))
print(" velocity = {0:.4f}".format(sim.bell.velocity))

# Run the given simulation for up to 120 seconds.

sim.bell.bell_angle = uniform(-2,2)
angles = [sim.bell.bell_angle]

while sim.phy.time < 60.0:
    inputs = sim.get_scaled_state()
    action = net.activate(inputs)
    
    # Apply action to the simulated cart-pole
    #force = discrete_actuator_force(action)
    force = discrete_actuator_force(action)
    #force = discrete_actuator_force(action)
    # Doesn't matter what you do with this as long as it's consistent. 
    sim.step(force)
    angles.append(sim.bell.bell_angle)
    
fitness = np.sum(np.array(angles)**2/np.pi**2)/len(np.array(angles))/(sim.bell.stay_hit + 1)
#db = (np.array(np.abs(angles)) - np.pi)**2/(np.pi**2)
#fitness = 1.0 - np.sum(db)/len(db)

print('fitness', fitness)

plt.plot(sim.bell.times, angles)
plt.show()

print()
print("Final conditions:")
print("    angle = {0:.4f}".format(sim.bell.bell_angle))
print(" velocity = {0:.4f}".format(sim.bell.velocity))
print()


#make_movie(net, discrete_actuator_force, 15.0, "feedforward-movie.mp4")
