"""Optimization module using various algorithms via PyGMO.

This module adapts the PPNO problem to the PyGMO framework for multi-objective
optimization, supporting algorithms like NSGA-II, MOEAD, MACO, and PSO.
"""

import logging
from time import perf_counter
from typing import Any, List, Optional, Tuple

import numpy as np
import pygmo as pg

from .constants import (
    GENERATIONS,
    MAX_ALGORITHM_TIME,
    MAX_TRIALS,
    PATIENCE,
    POPULATION_SIZE,
)

# Logger configuration
logger = logging.getLogger(__name__)

class PPNOProblem:
    """User-Defined Problem (UDP) for PyGMO integration.

    Adapts the network optimization problem to a format compatible with PyGMO
    algorithms, defining objectives, bounds, and variable types.

    Attributes:
        optimization_instance (Any): The main Optimization object.
    """

    def __init__(self, optimization_instance: Any):
        """Initialize the problem adapter.

        Args:
            optimization_instance (Any): An instance of the Optimization class.
        """
        self.optimization_instance = optimization_instance

    def fitness(self, x: np.ndarray) -> List[float]:
        """Calculate fitness values for a given solution vector.

        Objectives:
            1. Total network cost (to be minimized).
            2. Maximum pressure deficit (to be minimized towards <= 0).

        Args:
            x (np.ndarray): Vector of diameter indexes.

        Returns:
            List[float]: A list containing `[cost, max_deficit]`.
        """
        diameter_indexes = np.array(x).astype(np.int32)
        self.optimization_instance.set_x(diameter_indexes)

        # Objective 1: Investment Cost
        cost = float(self.optimization_instance.get_cost())

        # Objective 2: Feasibility (Pressure Deficit)
        # We look for the maximum deficit across all nodes.
        # A value <= 0 means all nodes satisfy pressure requirements.
        nodal_deficits = self.optimization_instance.check(mode='PD')
        max_deficit = float(np.max(nodal_deficits))

        return [cost, max_deficit]

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the lower and upper bounds for each variable.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: Lower and upper bounds.
        """
        return self.optimization_instance.lbound, self.optimization_instance.ubound

    def get_nobj(self) -> int:
        """Return the number of objectives (Cost and Max Deficit)."""
        return 2

    def get_nix(self) -> int:
        """Return the number of integer variables."""
        return self.optimization_instance.dimension

    def get_name(self) -> str:
        """Return the descriptive name of the problem."""
        return "Pressurized Pipe Network Optimization (Multi-objective)"


def evolve_ppno(optimization_instance: Any, 
                algorithm_factory: Any, 
                name: str,
                initial_x: Optional[np.ndarray] = None) -> Tuple[Optional[List[float]], Optional[np.ndarray]]:
    """Execute generic evolution loop for PyGMO algorithms.
    
    Evolves a population of solutions to find the best valid (max_deficit <= 0)
    solution with the minimum cost. This loop handles the multi-objective nature
    of the problem (Cost vs. Feasibility).

    Seeding Strategy:
        If `initial_x` is provided, the initial population is seeded. The first
        individual becomes the exact initial solution. A subset of the population 
        is then filled with mutated variations of this initial solution (changing 
        ~5% of pipes) to provide genetic diversity around a known good region.

    Args:
        optimization_instance (Any): The main Optimization object.
        algorithm_factory (Any): A callable returning a PyGMO algorithm instance.
        name (str): The human-readable name of the algorithm (for logging).
        initial_x (Optional[np.ndarray]): Optional starting solution vector.

    Returns:
        Tuple[Optional[List[float]], Optional[np.ndarray]]: A tuple containing 
            (`[Cost, Deficit]`, Best Solution Vector) or `(None, None)` if the 
            optimization fails or finds no valid solutions.
    """
    logger.info(f'*** {name} OPTIMIZATION ***')

    start_time = perf_counter()

    # PyGMO copies the UDP internally, so the main Optimization instance cannot
    # observe each check() call directly. Estimate simulation cycles per fitness
    # evaluation once, then scale it by the final PyGMO evaluation count.
    initial_cycles = int(optimization_instance.simulation_cycles)
    optimization_instance.check(mode='TF')
    cycles_per_eval = max(1, int(optimization_instance.simulation_cycles) - initial_cycles)

    prob = pg.problem(PPNOProblem(optimization_instance))

    total_generations = 0
    trials = 0
    consecutive_no_changes = 0

    best_cost_found: float = float('inf')
    best_valid_fitness: Optional[List[float]] = None
    best_valid_x: Optional[np.ndarray] = None

    # Initialize algorithm and population
    algorithm = pg.algorithm(algorithm_factory())
    pop_size = optimization_instance.config.get('PopulationSize', POPULATION_SIZE)
    population = pg.population(prob, size=pop_size)

    # Seed population with the initial solution and feasible variations
    if initial_x is not None:
        logger.info(f"      [SEEDED] Injecting initial solution into {name} population.")
        # Replace the first individual with the exact initial solution
        population.set_x(0, initial_x)
        
        # Initialize tracking with the seed if it's valid
        seed_fit = population.get_f()[0]
        if seed_fit[1] <= 0:
            best_cost_found = float(seed_fit[0])
            best_valid_fitness = list(seed_fit)
            best_valid_x = initial_x.copy()

        # Add light mutations around the seed so the population starts near a
        # feasible region without becoming a set of identical individuals.
        n_vars = len(initial_x)
        for i in range(1, min(10, pop_size)): # Seed first 10
            variant = initial_x.copy()
            n_change = max(1, int(n_vars * 0.05))
            idx_change = np.random.choice(n_vars, n_change, replace=False)
            for idx in idx_change:
                variant[idx] = np.clip(variant[idx] + np.random.randint(-1, 2), 
                                       optimization_instance.lbound[idx], 
                                       optimization_instance.ubound[idx])
            population.set_x(i, variant)

    max_trials = optimization_instance.config.get('MaxTrials', MAX_TRIALS)
    
    while True:
        trials += 1
        try:
            population = algorithm.evolve(population)
        except Exception as e:
            logger.error(f"  FAILED: {e}")
            if trials >= max_trials:
                logger.warning("Terminated: Maximum trials reached.")
                break
            continue
        total_generations += optimization_instance.config.get('Generations', GENERATIONS)

        # Search for the best valid solution in the current population
        fitness_values = population.get_f()
        solution_vectors = population.get_x()

        # Filter valid solutions and find the one with the minimum cost
        valid_solutions = [
            (fit, x) for fit, x in zip(fitness_values, solution_vectors)
            if fit[1] <= 0
        ]

        if valid_solutions:
            # Get valid solution with minimum cost
            current_best_fit, current_best_x = min(valid_solutions, key=lambda sol: sol[0][0])

            if current_best_fit[0] < best_cost_found:
                best_cost_found = current_best_fit[0]
                best_valid_fitness = list(current_best_fit)
                best_valid_x = current_best_x.copy()
                consecutive_no_changes = 0
            else:
                consecutive_no_changes += 1
        else:
            consecutive_no_changes += 1


        # Stopping criteria check
        elapsed_time = perf_counter() - start_time
        max_time = optimization_instance.config.get('MaxTime', MAX_ALGORITHM_TIME)
        patience = optimization_instance.config.get('Patience', PATIENCE)
        
        if elapsed_time >= max_time:
            logger.warning("Terminated: Maximum time reached.")
            break
        if trials >= max_trials:
            logger.warning("Terminated: Maximum trials reached.")
            break
        if consecutive_no_changes >= patience:
            logger.info(f"Terminated: Converged after {consecutive_no_changes} trials without improvement.")
            break

    # Final sync of simulation cycles back to the main instance
    optimization_instance.simulation_cycles = population.problem.get_fevals() * cycles_per_eval

    return best_valid_fitness, best_valid_x


def nsga2(optimization_instance: Any, initial_x: Optional[np.ndarray] = None) -> Tuple[Optional[List[float]], Optional[np.ndarray]]:
    """Run the Non-dominated Sorting Genetic Algorithm II."""
    gens = optimization_instance.config.get('Generations', GENERATIONS)
    return evolve_ppno(optimization_instance, lambda: pg.nsga2(gen=gens), "NSGA-II", initial_x)


def moead(optimization_instance: Any, initial_x: Optional[np.ndarray] = None) -> Tuple[Optional[List[float]], Optional[np.ndarray]]:
    """Run Multi-Objective Evolutionary Algorithm based on Decomposition."""
    gens = optimization_instance.config.get('Generations', GENERATIONS)
    return evolve_ppno(optimization_instance, lambda: pg.moead(gen=gens), "MOEAD", initial_x)


def maco(optimization_instance: Any, initial_x: Optional[np.ndarray] = None) -> Tuple[Optional[List[float]], Optional[np.ndarray]]:
    """Run Multi-objective Ant Colony Optimizer."""
    gens = optimization_instance.config.get('Generations', GENERATIONS)
    return evolve_ppno(optimization_instance, lambda: pg.maco(gen=gens), "MACO", initial_x)


def nspso(optimization_instance: Any, initial_x: Optional[np.ndarray] = None) -> Tuple[Optional[List[float]], Optional[np.ndarray]]:
    """Run Non-dominated Sorting Particle Swarm Optimizer."""
    gens = optimization_instance.config.get('Generations', GENERATIONS)
    return evolve_ppno(optimization_instance, lambda: pg.nspso(gen=gens), "NSP-SO (PSO)", initial_x)

