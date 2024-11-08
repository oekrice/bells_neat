"""
Single-pole balancing experiment using a feed-forward neural network.
"""

import multiprocessing
import os
import pickle

from main import run_bell, discrete_actuator_force, continuous_actuator_force

import main
import neat
import numpy as np
from random import uniform, gauss

runs_per_net = 10
simulation_seconds = 60.0
ngenerations = 50

# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        sim = run_bell()   #all the physics in here
        sim.bell.bell_angle = uniform(-1,1)
        sim.bell.velocity = 0.0
        # Run the given simulation for up to num_steps time steps.
        angles = [sim.bell.bell_angle]
        velocities = [sim.bell.bell_angle]

        while sim.phy.time < simulation_seconds:
            #Inputs are the things we can know -- in my case it is the angle and speed of the bell (for now)
            #Do try to remember to get inputs in the range (0,1). Can do easily enough.
            inputs = sim.get_scaled_state()
            #This is just a list.
            action = net.activate(inputs)            
            # Apply action to the simulated cart-pole
            force = continuous_actuator_force(action)
        
            sim.step(force)

            angles.append(sim.bell.bell_angle)
                
        angles = np.array(angles)
        if True:   #Ringing up
            if False:#sim.bell.stay_hit:
                fitness = 0.0
            else:
                fitness = np.sum(np.array(angles)**2/np.pi**2)/len(np.array(angles))
        elif False:   #Ringing down
            fitness = np.sum((np.pi + sim.bell.stay_angle - np.abs(np.array(angles)))**2/np.pi**2)/len(np.array(angles))
        if False:   #Hold on the balance?
            db = (np.array(np.abs(angles)) - np.pi)**2/(np.pi**2)
            fitness = 1.0 - np.sum(db)/len(db)
    
        fitness = np.sum(np.array(angles)**2/np.pi**2)/len(np.array(angles))/(sim.bell.stay_hit + 1)
        fitnesses.append(fitness)
    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config_bell')
    #Load in config file. Will tweak in due course.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    pop = neat.Population(config)
    #These just print some things out. But keep on...
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    #pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count()-1, eval_genome)

    winner = pop.run(pe.evaluate,n=ngenerations)

    # Save the winner.
    with open('winner_bell', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

if __name__ == '__main__':
    run()
