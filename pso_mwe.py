#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 19:55:39 2018

@author: herman
"""

import pyswarms as ps
import numpy as np

n_particles = 40
dimensions = 190

init_pos  = np.random.random([dimensions, n_particles])

options = {'c1': 2, #  cognitive parameter (weight of personal best)
           'c2': 2, #  social parameter (weight of swarm best)
           'v': 0, #  initial velocity
           'w': 0.4, #  inertia
           'k': 2, #  Number of neighbours. Ring topology seems popular
           'p': 2}  #  Distance function (Minkowski p-norm). 1 for abs, 2 for Euclidean
iterations = 10
swarm = ps.backend.generators.create_swarm(n_particles,
             options=options,
             dimensions=dimensions)
top = ps.backend.topology.Ring()

def objective(pos):
    return 100 - np.sum(pos, axis=1)

for i in range(iterations):
    # Compute cost for current position and personal best
    swarm.current_cost = p.pso_objective(swarm.position)
    swarm.pbest_cost = p.pso_objective(swarm.pbest_pos)
    swarm.pbest_pos, swarm.pbest_cost = ps.backend.operators.compute_pbest(
        swarm
    )
    best_cost_yet_found = np.min(swarm.best_cost)
    # Update gbest from neighborhood
    swarm.best_pos, swarm.best_cost = top.compute_gbest(
        swarm, options['p'], options['k']
    )

    swarm.velocity = top.compute_velocity(
        swarm)
    swarm.position = top.compute_position(
        swarm)
    print(best_cost_yet_found)
# Obtain the final best_cost and the final best_position
final_best_cost = swarm.best_cost.copy()
final_best_pos = swarm.best_pos.copy()