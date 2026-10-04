"""Pressurized Pipe Network Optimizer (PPNO) core module.

This module provides the main `Optimization` class, which serves as the central
coordinator for the two-stage pipe sizing optimization pipeline. It handles file
parsing, semantic validation, EPANET hydraulic simulation state, and delegates
the optimization process to specific heuristic and metaheuristic solvers.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from time import localtime, perf_counter, strftime
from typing import Any, Dict, List, NoReturn, Optional, Tuple, Union, cast

import numpy as np
import pygmo as pg

try:
    from entoolkit import toolkit as et
    # Support for newer versions where functions moved to legacy
    if not hasattr(et, 'ENopen'):
        from entoolkit import legacy as et
except ImportError:
    try:
        from entoolkit import legacy as et
    except ImportError:
        import entoolkit as et

if __package__:
    from . import section_parser as sp
    from .local_refiner import LocalRefiner
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ppno import section_parser as sp
    from ppno.local_refiner import LocalRefiner

# Logger configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
if __package__:
    from .constants import (
        ALGORITHM_DA,
        ALGORITHM_DE,
        ALGORITHM_MACO,
        ALGORITHM_MOEAD,
        ALGORITHM_NSGA2,
        ALGORITHM_PSO,
        ALGORITHM_UH,
        DEFAULT_CONFIG,
        MAX_RETRIES,
    )
else:
    from ppno.constants import (
        ALGORITHM_DA,
        ALGORITHM_DE,
        ALGORITHM_MACO,
        ALGORITHM_MOEAD,
        ALGORITHM_NSGA2,
        ALGORITHM_PSO,
        ALGORITHM_UH,
        DEFAULT_CONFIG,
        MAX_RETRIES,
    )

ALGORITHM_BY_NAME = {
    'UH': ALGORITHM_UH,
    'DE': ALGORITHM_DE,
    'DA': ALGORITHM_DA,
    'NSGA2': ALGORITHM_NSGA2,
    'MOEAD': ALGORITHM_MOEAD,
    'MACO': ALGORITHM_MACO,
    'PSO': ALGORITHM_PSO,
}

ALGORITHM_NAMES = {value: key for key, value in ALGORITHM_BY_NAME.items()}
SCIPY_ALGORITHMS = {ALGORITHM_DE, ALGORITHM_DA}
PYGMO_ALGORITHMS = {ALGORITHM_NSGA2, ALGORITHM_MOEAD, ALGORITHM_MACO, ALGORITHM_PSO}


class Optimization:
    """Core class for pipe network optimization.

    Coordinates the hydraulic simulation and the optimization algorithms to
        find the best pipe diameters from the available pipe-size groups.

    Attributes:
        problem_file (Path): Path to the .ext problem definition file.
        inp_file (Path): Path to the original EPANET .inp file.
        rpt_file (Path): Path to the generated EPANET .rpt file.
        algorithm (int): Chosen optimization algorithm ID.
        pipes (np.ndarray): Array of pipes to be sized.
        nodes (np.ndarray): Array of nodes to check for pressure requirements.
        pipe_catalog_file (Path): Path to the external pipe catalog file.
        pipe_sizes (Dict[str, np.ndarray]): Available pipe-size groups and their properties.
        dimension (int): Number of variable pipes.
        lbound (np.ndarray): Lower bounds for variable indexes.
        ubound (np.ndarray): Upper bounds for variable indexes.
        algorithms (List[int]): List of Stage 2 metaheuristic algorithms to execute.
        max_retries (int): Maximum retry attempts for failed algorithms.
        simulation_cycles (int): Counter for network hydraulic simulation runs.
        results (List[Dict[str, Any]]): Collected performance data.
    """

    def __init__(self, problem_file: Union[str, Path]):
        """Initialize the optimization problem and perform full semantic validation.

        Reads the problem definition from the `.ext` file, locates the associated EPANET 
        `.inp` model, and validates all entities (pipes, nodes, pipe sizes) to ensure 
        hydraulic and logical consistency before proceeding with optimization.

        Args:
            problem_file (Union[str, Path]): Absolute or relative path to the `.ext` 
                configuration file containing the problem definition.

        Raises:
            FileNotFoundError: If the specified `.ext` file or its referenced `.inp` model 
                cannot be found.
            ValueError: If the configuration validation fails due to syntax errors, missing 
                hydraulic entities, or logical inconsistencies (e.g., non-monotonic pipe sizes).
        """
        self.problem_file = Path(problem_file)
        if not self.problem_file.exists():
            raise FileNotFoundError(f"Problem file not found: {self.problem_file}")

        parser = sp.SectionParser(self.problem_file)
        sections = parser.read()

        # 1. Verify INP Section
        if 'INP' not in sections or not sections['INP']:
            raise ValueError(f"Line 1: Mandatory [INP] section is missing or empty in {self.problem_file.name}")
        
        inp_line_num, inp_raw_path = sections['INP'][0]
        inp_path = self._resolve_problem_path(inp_raw_path)
            
        if not inp_path.exists():
             raise FileNotFoundError(f"Line {inp_line_num}: EPANET INP file not found: {inp_raw_path}")

        self.inp_file = inp_path
        self.pipe_catalog_file = self._load_pipe_catalog_file(sections)
        pipe_catalog_parser = sp.SectionParser(self.pipe_catalog_file)
        self.pipe_catalog_lines = pipe_catalog_parser.read_rows()
        self.algorithm = ALGORITHM_UH
        self.best_algorithm_name = ALGORITHM_NAMES[ALGORITHM_UH]
        self.config = DEFAULT_CONFIG.copy()

        # 2. Open Toolkit for entity validation
        rpt = os.devnull
        try:
            et.ENopen(str(self.inp_file), rpt)
        except Exception as e:
            raise ValueError(f"Line {inp_line_num}: Failed to load EPANET model {self.inp_file.name} ({str(e)})") from e
        
        et.ENopenH()
        try:
            et.ENsetstatusreport(0)
        except Exception:
            pass

        try:
            # 3. Comprehensive Validation
            self._validate_config(sections, parser, self.pipe_catalog_lines)

            # 4. Load Data
            self._load_options(sections.get('OPTIONS', []), parser)
            self._apply_random_seed()

            logger.info(f"Loading optimization problem: {self.problem_file}")
            logger.info("-" * 80)
            logger.info(f"NETWORK DATA: {self.inp_file}")

            self._load_pipes(sections.get('PIPES', []), parser)
            self._load_pressures(sections.get('PRESSURES', []), parser)
            self._load_pipe_sizes(self.pipe_catalog_lines, parser)

            self.dimension = len(self.pipes)
            self._current_x = np.zeros(self.dimension, dtype=np.int32)
            self.lbound = np.zeros(self.dimension, dtype=np.int32)
            self.ubound = np.array([len(self.pipe_sizes[str(p['group'])]) - 1 for p in self.pipes], dtype=np.int32)
        except Exception:
            self.close()
            raise

        self.simulation_cycles = 0
        self.results: List[Dict[str, Any]] = []
        logger.info("-" * 80)

    def _resolve_problem_path(self, raw_path: str) -> Path:
        """Resolve a path from CWD, the .ext directory, or by basename there."""
        resolved_path = Path(raw_path)
        if resolved_path.is_absolute():
            return resolved_path

        candidates = [
            resolved_path,
            self.problem_file.parent / resolved_path,
            self.problem_file.parent / resolved_path.name,
        ]
        return next((candidate for candidate in candidates if candidate.exists()), candidates[-1])

    def _load_pipe_catalog_file(self, sections: Dict[str, List[Tuple[int, str]]]) -> Path:
        """Resolve the external pipe catalog referenced by [PIPE_CATALOG]."""
        catalog_lines = sections.get('PIPE_CATALOG', [])
        if not catalog_lines:
            raise ValueError(f"Mandatory [PIPE_CATALOG] section is missing or empty in {self.problem_file.name}")
        if len(catalog_lines) > 1:
            line_num, _ = catalog_lines[1]
            raise ValueError(f"Line {line_num}: [PIPE_CATALOG] accepts exactly one file path")

        catalog_line_num, catalog_raw_path = catalog_lines[0]
        catalog_path = self._resolve_problem_path(catalog_raw_path)
        if not catalog_path.exists():
            raise FileNotFoundError(f"Line {catalog_line_num}: Pipe catalog file not found: {catalog_raw_path}")
        return catalog_path

    def _validate_config(
            self,
            sections: Dict[str, List[Tuple[int, str]]],
            parser: sp.SectionParser,
            pipe_catalog_lines: List[Tuple[int, str]]
            ) -> None:
        """Performs semantic validation of the configuration."""
        errors = []

        # Validate option values early so loading can stay simple.
        algorithm_names = set(ALGORITHM_BY_NAME)
        supported_options = {'ALGORITHM', 'ALGORITHMS'}

        options = sections.get('OPTIONS', [])
        for line_num, content in options:
            tokens = parser.line_to_tuple(content)
            if not tokens:
                continue
            tokens = tuple(t for t in tokens if t != '=')
            if not tokens:
                continue
            key = tokens[0].upper().replace('_', '')
            if key in supported_options:
                for val in tokens[1:]:
                    if val.upper() not in algorithm_names:
                        errors.append(f"Line {line_num}: Unknown algorithm '{val}'")
            else:
                errors.append(f"Line {line_num}: Unsupported option '{tokens[0]}'. [OPTIONS] only supports Algorithm")

        # Check pipe existence and pipe-size group references
        pipes_lines = sections.get('PIPES', [])
        pipe_size_groups = set()
        for line_num, content in pipe_catalog_lines:
            tokens = parser.line_to_tuple(content)
            if tokens and tokens[0].startswith('['):
                errors.append(f"Line {line_num}: Pipe catalog files must not contain section headers")
                continue
            if len(tokens) >= 4:
                pipe_size_groups.add(tokens[0])

        for line_num, content in pipes_lines:
            tokens = parser.line_to_tuple(content)
            if len(tokens) < 2:
                errors.append(f"Line {line_num}: Invalid pipe definition. Expected 'ID GROUP'")
                continue
            pipe_id, group_name = tokens[0], tokens[1]
            try:
                et.ENgetlinkindex(pipe_id)
            except Exception:
                errors.append(f"Line {line_num}: Pipe '{pipe_id}' not found in hydraulic model")
            
            if group_name not in pipe_size_groups:
                errors.append(f"Line {line_num}: Pipe-size group '{group_name}' not defined in pipe catalog")

        # Check Nodes and Pressures
        pressures_lines = sections.get('PRESSURES', [])
        for line_num, content in pressures_lines:
            tokens = parser.line_to_tuple(content)
            if len(tokens) < 2:
                errors.append(f"Line {line_num}: Invalid pressure definition. Expected 'ID MIN_PRESSURE'")
                continue
            node_id, min_p = tokens[0], tokens[1]
            try:
                et.ENgetnodeindex(node_id)
            except Exception:
                errors.append(f"Line {line_num}: Node '{node_id}' not found in hydraulic model")
            try:
                float(min_p)
            except ValueError:
                errors.append(f"Line {line_num}: Invalid pressure value '{min_p}'")

        # Check pipe-size group consistency (strictly increasing diameter)
        temp_sizes: Dict[str, List[Dict[str, Any]]] = {}
        for line_num, content in pipe_catalog_lines:
            tokens = parser.line_to_tuple(content)
            if not tokens:
                continue
            if tokens[0].startswith('['):
                continue
            if len(tokens) < 4:
                errors.append(f"Line {line_num}: Invalid pipe catalog entry. Expected 'GROUP DIAMETER ROUGHNESS PRICE'")
                continue
            try:
                sn, d, _r, p = tokens[0], float(tokens[1]), float(tokens[2]), float(tokens[3])
            except ValueError:
                errors.append(f"Line {line_num}: Invalid numeric values in pipe catalog group '{tokens[0]}'")
                continue
            
            if sn not in temp_sizes:
                temp_sizes[sn] = []
            temp_sizes[sn].append({'d': d, 'p': p, 'line': line_num})

        for sn, items in temp_sizes.items():
            for i in range(1, len(items)):
                if items[i]['d'] <= items[i-1]['d']:
                    errors.append(f"Line {items[i]['line']}: Diameter must be strictly increasing in group '{sn}'")
                if items[i]['p'] <= items[i-1]['p']:
                     logger.warning(f"Line {items[i]['line']}: Price {items[i]['p']} is not greater than {items[i-1]['p']} for larger diameter (Anomalous)")

        if errors:
            raise ValueError("Configuration Validation Failed:\n" + "\n".join(errors))

    def _load_options(self, options_lines: List[Tuple[int, str]], parser: sp.SectionParser) -> None:
        """Parses the OPTIONS section."""
        self.algorithms = []  # Stage 2 metaheuristics; empty = only UH + FLS-H
        self.max_retries = MAX_RETRIES
        seen_algorithms = set()

        for _line_num, content in options_lines:
            tokens = parser.line_to_tuple(content)
            if not tokens:
                continue
            
            # Filter out '=' if present
            tokens = tuple(t for t in tokens if t != '=')
            if not tokens:
                continue

            key = tokens[0].upper().replace('_', '') # Normalize: remove underscores
            values = tokens[1:]

            if key in ['ALGORITHM', 'ALGORITHMS'] and values:
                for value in values:
                    algorithm_id = ALGORITHM_BY_NAME[value.upper()]
                    if algorithm_id == ALGORITHM_UH or algorithm_id in seen_algorithms:
                        continue
                    self.algorithms.append(algorithm_id)
                    seen_algorithms.add(algorithm_id)
                logger.info(f"Optional Metaheuristics: {', '.join(values)}")

    def _apply_random_seed(self) -> None:
        """Apply the optional global random seed configured in constants.py."""
        seed = self.config.get('RandomSeed')
        if seed is None:
            return
        np.random.seed(seed)
        pg.set_global_rng_seed(seed)
        logger.info(f"RandomSeed: {seed}")

    def _load_pipes(self, pipe_lines: List[Tuple[int, str]], parser: sp.SectionParser) -> None:
        """Parses the PIPES section."""
        dt = np.dtype([('link_idx', 'i4'), ('id', 'U16'), ('length', 'f4'), ('group', 'U16')])
        data = []
        for _line_num, content in pipe_lines:
            tokens = parser.line_to_tuple(content)
            pipe_id, group_name = tokens[0], tokens[1]
            link_idx = et.ENgetlinkindex(pipe_id)
            length = et.ENgetlinkvalue(link_idx, et.EN_LENGTH)
            data.append((link_idx, pipe_id, length, group_name))

        self.pipes = np.array(data, dt)
        logger.info(f"Loaded {len(self.pipes)} pipes for sizing.")

    def _load_pressures(self, pressure_lines: List[Tuple[int, str]], parser: sp.SectionParser) -> None:
        """Parses the PRESSURES section."""
        dt = np.dtype([('node_idx', 'i4'), ('id', 'U16'), ('min_pressure', 'f4')])
        data = []
        for _line_num, content in pressure_lines:
            tokens = parser.line_to_tuple(content)
            node_id, min_p = tokens[0], tokens[1]
            node_idx = et.ENgetnodeindex(node_id)
            data.append((node_idx, node_id, float(min_p)))

        self.nodes = np.array(data, dtype=dt)
        logger.info(f"Loaded {len(self.nodes)} pressure constraints.")

    def _load_pipe_sizes(self, pipe_size_lines: List[Tuple[int, str]], parser: sp.SectionParser) -> None:
        """Parses the external pipe catalog rows."""
        dt = np.dtype([('diameter', 'f4'), ('roughness', 'f4'), ('price', 'f4')])
        required_groups = set(str(p['group']) for p in self.pipes)

        raw_data: Dict[str, List[Tuple[float, float, float]]] = {s: [] for s in required_groups}
        for _line_num, content in pipe_size_lines:
            tokens = parser.line_to_tuple(content)
            if tokens and tokens[0].startswith('['):
                continue
            if len(tokens) < 4:
                continue
            sn, d, r, p = tokens[0], tokens[1], tokens[2], tokens[3]
            if sn in required_groups:
                raw_data[sn].append((float(d), float(r), float(p)))

        self.pipe_sizes = {
            name: np.sort(np.array(data, dtype=dt), order='diameter')
            for name, data in raw_data.items()
        }
        logger.info(f"Loaded {len(self.pipe_sizes)} pipe-size groups.")

    def set_x(self, x: np.ndarray) -> None:
        """Updates the hydraulic model with the new diameter indexes."""
        self._current_x = x.astype(np.int32)
        for i, pipe in enumerate(self.pipes):
            link_idx = int(pipe['link_idx'])
            pipe_id = str(pipe['id'])
            group = self.pipe_sizes[str(pipe['group'])]
            size_idx = int(self._current_x[i])

            try:
                et.ENsetlinkvalue(link_idx, et.EN_DIAMETER, float(group[size_idx]['diameter']))
                et.ENsetlinkvalue(link_idx, et.EN_ROUGHNESS, float(group[size_idx]['roughness']))
            except Exception as e:
                logger.error(f"Failed to set hydraulic values for pipe {pipe_id} (index {link_idx}): {e}")

    def get_x(self) -> np.ndarray:
        """Retrieve a copy of the current vector of diameter indexes.

        Returns:
            np.ndarray: A 1D integer array representing the active sizes for each pipe.
        """
        return self._current_x.copy()

    def check(self, mode: str = 'TF') -> Union[bool, Tuple[bool, np.ndarray], np.ndarray]:
        """Simulate the network and evaluate pressure constraints across all time steps.

        This method triggers an EPANET Extended Period Simulation (EPS). It tracks the 
        maximum pressure deficits at critical nodes and (optionally) the headloss gradients
        for heuristic algorithms.

        Args:
            mode (str, optional): Evaluation mode. 
                - 'TF': Returns a boolean indicating strict overall feasibility (True/False).
                - 'UH': Returns a tuple `(status, sorted_indices)` for the Unit Headloss heuristic.
                - 'PD': Returns an array of maximum pressure deficits for each node.
                Defaults to 'TF'.

        Returns:
            Union[bool, Tuple[bool, np.ndarray], np.ndarray]: The evaluation result 
            structured according to the requested `mode`.
        """
        deficits = np.full(len(self.nodes), -1e10, dtype=np.float32) if mode == 'PD' else None
        max_hls = np.zeros(len(self.pipes), dtype=np.float32) if mode == 'UH' else None
        overall_status = True

        try:
            self.simulation_cycles += 1
            et.ENinitH(0)
            while True:
                et.ENrunH()

                # Check nodal pressures
                for i, node in enumerate(self.nodes):
                    calculated_p = et.ENgetnodevalue(int(node['node_idx']), et.EN_PRESSURE)
                    required_p = float(node['min_pressure'])
                    deficit = required_p - calculated_p

                    if deficit > 0:
                        overall_status = False

                    if mode == 'PD' and deficits is not None:
                        if deficits[i] < deficit:
                            deficits[i] = deficit

                # Track link headlosses for UH mode
                if mode == 'UH' and max_hls is not None:
                    for i in range(len(self.pipes)):
                        hl = et.ENgetlinkvalue(int(self.pipes[i]['link_idx']), et.EN_HEADLOSS)
                        gradient = abs(hl) / float(self.pipes[i]['length'])
                        if max_hls[i] < gradient:
                            max_hls[i] = gradient

                if et.ENnextH() == 0:
                    break
        except Exception as e:
            logger.debug(f"Hydraulic simulation error in check(): {e}")
            overall_status = False

        if mode == 'UH':
            sorted_indices = np.argsort(max_hls)[::-1]
            return overall_status, sorted_indices
        if mode == 'PD':
            return deficits if deficits is not None else np.array([])
        return overall_status

    def get_cost(self) -> float:
        """Calculate the total network investment cost based on current sizing.

        Returns:
            float: The total cost computed by summing (length * price) for all pipes.
        """
        total = 0.0
        x = self.get_x()
        for i, pipe in enumerate(self.pipes):
            total += float(pipe['length']) * float(self.pipe_sizes[str(pipe['group'])][int(x[i])]['price'])
        return total

    def solve(self) -> Optional[np.ndarray]:
        """Execute the full two-stage optimization pipeline.

        Orchestrates the mandatory Heuristic Foundation (Unit Headloss + FLS-H) 
        and the optional Global Exploration using the configured metaheuristics.

        Returns:
            Optional[np.ndarray]: The best discrete solution vector found, or None 
            if the heuristic stage fails to find any feasible configuration.
        """
        self.results = []
        self.simulation_cycles = 0
        logger.info(f"Optimization pipeline started at: {strftime('%H:%M:%S', localtime())}")

        # --- STAGE 1: Mandatory Foundation (UH + LS) ---
        stage1_res = self._run_stage_1()
        if stage1_res is None:
            return None
        
        overall_best_solution, overall_best_cost = stage1_res

        # --- STAGE 2: Optional Global Exploration (Metaheuristics) ---
        if self.algorithms:
            overall_best_solution, overall_best_cost = self._run_stage_2(overall_best_solution, overall_best_cost)
        else:
            logger.info("\n>>> STAGE 2 Skipped: No global exploration requested <<<")

        self._print_summary()
        self.set_x(overall_best_solution)
        self._handle_success(overall_best_solution)
        
        return overall_best_solution

    def _run_stage_1(self) -> Optional[Tuple[np.ndarray, float]]:
        """Runs the first stage of optimization: Heuristic + Refinement."""
        logger.info("\n" + ">>> STAGE 1: HEURISTIC FOUNDATION (UH + FLS-H) <<<")
        start_time = perf_counter()
        
        # 1a. UH Heuristic
        solution = self._solve_uh()
        if solution is None:
            logger.error("Stage 1 failed: Unit Headloss Heuristic could not find a solution.")
            return None
        
        # 1b. Refinement (FLS-H) - always applied
        solution = self._apply_refinement(solution)
        
        duration = perf_counter() - start_time
        self.set_x(solution)
        cost = self.get_cost()
        
        logger.info(f"Stage 1 Complete | Cost: {cost:.2f} | Time: {duration:.2f}s")
        
        self.results.append({
            'Algorithm': 'UH',
            'Attempt': 1,
            'Success': "YES",
            'Time (s)': f"{duration:.2f}",
            'Simulations': self.simulation_cycles,
            'Cost': f"{cost:.2f}"
        })
        self.algorithm = ALGORITHM_UH
        self.best_algorithm_name = ALGORITHM_NAMES[ALGORITHM_UH]
        return solution, cost

    def _run_stage_2(self, best_sol: np.ndarray, best_cost: float) -> Tuple[np.ndarray, float]:
        """Runs the second stage of optimization: Global Exploration."""
        logger.info("\n" + ">>> STAGE 2: OPTIONAL GLOBAL EXPLORATION <<<")
        
        for alg_id in self.algorithms:
            alg_name = ALGORITHM_NAMES.get(alg_id, 'UNKNOWN')
            best_sol, best_cost = self._execute_metaheuristic(alg_id, alg_name, best_sol, best_cost)

        # Final refinement to the absolute best solution found
        logger.info("\n>>> REFINING BEST SOLUTION (FLS-H) <<<")
        start_time_ref = perf_counter()
        best_sol = self._apply_refinement(best_sol)
        duration_ref = perf_counter() - start_time_ref
        
        self.set_x(best_sol)
        refined_cost = self.get_cost()
        logger.info(f"Refinement Complete | Cost: {refined_cost:.2f} | Time: {duration_ref:.2f}s")
        
        return best_sol, refined_cost

    def _execute_metaheuristic(self, alg_id: int, alg_name: str, 
                               overall_best_sol: np.ndarray, overall_best_cost: float) -> Tuple[np.ndarray, float]:
        """Handles the retry loop and performance tracking for a single metaheuristic."""
        best_sol = overall_best_sol
        best_cost = overall_best_cost
        algorithm_best_sol = None
        algorithm_best_cost = float('inf')

        for attempt in range(1, self.max_retries + 1):
            logger.info("-" * 40)
            logger.info(f"ALGORITHM: {alg_name} (Attempt {attempt}/{self.max_retries})")
            
            start_time = perf_counter()
            self.simulation_cycles = 0

            logger.info(f"      [SEED] Starting with best cost: {best_cost:.2f}")
            meta_solution = self._run_meta_algorithm(alg_id, best_sol)
            duration = perf_counter() - start_time
            
            if meta_solution is not None:
                self.set_x(meta_solution)
                meta_cost = self.get_cost()
                if meta_cost < algorithm_best_cost:
                    algorithm_best_cost = meta_cost
                    algorithm_best_sol = meta_solution.copy()
                
                if meta_cost < best_cost:
                    logger.info(f"      [ACCEPTED] Improved cost found: {meta_cost:.2f} (Previous: {best_cost:.2f})")
                    best_cost = meta_cost
                    best_sol = meta_solution.copy()
                    self.algorithm = alg_id
                    self.best_algorithm_name = alg_name
                else:
                    logger.info(f"      [DISCARDED] Cost {meta_cost:.2f} is not an improvement over {best_cost:.2f}")
                
                self.results.append({
                    'Algorithm': alg_name, 'Attempt': attempt, 'Success': "YES",
                    'Time (s)': f"{duration:.2f}", 'Simulations': self.simulation_cycles,
                    'Cost': f"{meta_cost:.2f}"
                })
                if meta_cost < overall_best_cost:
                    break
            else:
                self.results.append({
                    'Algorithm': alg_name, 'Attempt': attempt, 'Success': "NO",
                    'Time (s)': f"{duration:.2f}", 'Simulations': self.simulation_cycles, 'Cost': "-"
                })

        if algorithm_best_sol is not None:
            self.set_x(algorithm_best_sol)
            self._save_scn_result(alg_name)
            self.set_x(best_sol)
        
        return best_sol, best_cost

    def _run_meta_algorithm(self, alg_id: int, initial_x: np.ndarray) -> Optional[np.ndarray]:
        """Dispatches to the specific SciPy or PyGMO solver."""
        if alg_id in SCIPY_ALGORITHMS:
            if __package__:
                from . import scipy_solver
            else:
                from ppno import scipy_solver
            return scipy_solver.solve_scipy(self, alg_id, initial_x=initial_x)

        if alg_id in PYGMO_ALGORITHMS:
            if __package__:
                from . import pygmo_solver
            else:
                from ppno import pygmo_solver
            sol_x = None
            if alg_id == ALGORITHM_NSGA2:
                _, sol_x = pygmo_solver.nsga2(self, initial_x=initial_x)
            elif alg_id == ALGORITHM_MOEAD:
                _, sol_x = pygmo_solver.moead(self, initial_x=initial_x)
            elif alg_id == ALGORITHM_MACO:
                _, sol_x = pygmo_solver.maco(self, initial_x=initial_x)
            elif alg_id == ALGORITHM_PSO:
                _, sol_x = pygmo_solver.nspso(self, initial_x=initial_x)
            
            return np.array(sol_x, dtype=np.int32) if sol_x is not None else None
        
        return None

    def _print_summary(self) -> None:
        """Displays a summary of all optimization runs (aggregated by algorithm)."""
        logger.info("\n" + "=" * 80)
        logger.info(f"{'ALGORITHM SUMMARY':^80}")
        logger.info("=" * 80)
        header = f"{'Algorithm':<14} {'Tries':<7} {'Success':<8} {'Time(s)':>10} {'Sims':>10} {'Best Cost':>15}"
        logger.info(header)
        logger.info("-" * 80)
        
        aggregated: Dict[str, Dict[str, Any]] = {}
        for res in self.results:
            name = res['Algorithm']
            if name not in aggregated:
                aggregated[name] = {
                    'tries': 0,
                    'success': "NO",
                    'time': 0.0,
                    'sims': 0,
                    'cost': float('inf')
                }
            
            stats = aggregated[name]
            stats['tries'] += 1
            if res['Success'] == "YES":
                stats['success'] = "YES"
            
            stats['time'] += float(res['Time (s)'])
            stats['sims'] += int(res['Simulations'])
            
            if res['Cost'] != "-":
                stats['cost'] = min(stats['cost'], float(res['Cost']))

        for name, stats in aggregated.items():
            cost_str = f"{stats['cost']:.2f}" if stats['cost'] != float('inf') else "-"
            row = (f"{name:<14} {stats['tries']:<7} {stats['success']:<8} "
                   f"{stats['time']:>10.2f} {stats['sims']:>10} {cost_str:>15}")
            logger.info(row)
        logger.info("=" * 80 + "\n")

    def _solve_uh(self) -> Optional[np.ndarray]:
        """Execute the Unit Headloss (UH) heuristic.

        Greedily increases the diameter of pipes with the highest hydraulic gradient 
        until all pressure constraints are satisfied, establishing a feasible baseline.

        Returns:
            Optional[np.ndarray]: A feasible diameter index vector, or None if the 
            algorithm exhausts all maximum diameters without achieving feasibility.
        """
        logger.info("*** UNIT HEADLOSS HEURISTIC ***")
        self.set_x(np.zeros(self.dimension, dtype=np.int32))
        while True:
            status, sorted_hls = cast(Tuple[bool, np.ndarray], self.check(mode='UH'))
            if status:
                return self.get_x().copy()

            expanded = False
            for idx in sorted_hls:
                x = self.get_x().copy()
                if x[idx] < self.ubound[idx]:
                    x[idx] += 1
                    self.set_x(x)
                    expanded = True
                    break
            if not expanded:
                return None


    def _apply_refinement(self, solution: np.ndarray) -> np.ndarray:
        """Applies FLS-H refined local search to the current solution."""
        config = {
            'max_iter': self.config['RefinerIters'],
            'acceptance_threshold': self.config['RefinerWorsening'],
            'neighborhood_size': self.config['RefinerNeighbors']
        }
        
        refiner = LocalRefiner(self, config)
        refined_solution = refiner.refine(solution)
        
        return refined_solution

    def _save_scn_result(self, algorithm_name: str) -> None:
        """Saves a .scn file with the diameters and roughnesses obtained by the algorithm.

        The filename format: <inp_name>_result_<algorithm_name>.scn
        """
        scn_filename = f"{self.inp_file.stem}_result_{algorithm_name}.scn"
        scn_path = self.inp_file.parent / scn_filename

        try:
            with open(scn_path, 'w', encoding='utf-8') as f:
                x = self.get_x()

                f.write("[DIAMETERS]\n")
                f.write(f"; Result for algorithm: {algorithm_name}\n")
                f.write(f";{'Pipe':<16}\tDiameter\n")
                for i, pipe in enumerate(self.pipes):
                    pipe_id = str(pipe['id'])
                    group = self.pipe_sizes[str(pipe['group'])]
                    size_idx = int(x[i])
                    diameter = float(group[size_idx]['diameter'])
                    f.write(f" {pipe_id:<16}\t{diameter:.4f}\n")

                f.write("\n\n[ROUGHNESS]\n")
                f.write(f";{'Pipe':<16}\tRoughness\n")
                for i, pipe in enumerate(self.pipes):
                    pipe_id = str(pipe['id'])
                    group = self.pipe_sizes[str(pipe['group'])]
                    size_idx = int(x[i])
                    roughness = float(group[size_idx]['roughness'])

                    f.write(f" {pipe_id:<16}\t{roughness:.6f}\n")

            logger.info(f"Results saved to SCN file: {scn_path}")
        except Exception as e:
            logger.error(f"Failed to save SCN file {scn_path}: {e}")

    def _handle_success(self, solution: np.ndarray) -> None:
        """Saves and prints results for a successful run."""
        self.set_x(solution)
        cost = self.get_cost()
        logger.info(f"Success! Final Network Cost: {cost:.2f}")
        logger.info(f"Best source algorithm: {self.best_algorithm_name}")

    def pretty_print(self, x: np.ndarray) -> None:
        """Displays the solution in a formatted table."""
        logger.info("\n" + "*" * 20 + " FINAL SOLUTION " + "*" * 20)
        logger.info(f"{'Pipe ID':>16} {'Series':>12} {'Diam':>8} {'Rough':>8} {'Len':>8} {'Price':>8} {'Total':>10}")
        logger.info("-" * 80)

        for i, pipe in enumerate(self.pipes):
            size = int(x[i])
            cat = self.pipe_sizes[str(pipe['group'])][size]
            d, r, p = float(cat['diameter']), float(cat['roughness']), float(cat['price'])
            length = float(pipe['length'])
            logger.info(f"{str(pipe['id']):>16} {str(pipe['group']):>12} {d:8.1f} {r:8.4f} {length:8.1f} {p:8.2f} {length * p:10.2f}")

        logger.info("-" * 80)
        logger.info(f"TOTAL NETWORK COST: {self.get_cost():.2f}")
        logger.info("*" * 56 + "\n")

    def close(self) -> None:
        """Safely closes the EPANET toolkit."""
        try:
            et.ENcloseH()
            et.ENclose()
        except Exception:
            pass


class PPNOArgumentParser(argparse.ArgumentParser):
    """Argument parser that prints full help when command-line parsing fails."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: {message}\n\n{self.format_help()}")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the PPNO executable."""
    return PPNOArgumentParser(
        prog="ppno",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Pressurized Pipe Network Optimizer (PPNO)\n\n"
            "Optimize pipe diameters for an EPANET model using a .ext problem file."
        ),
        epilog=(
            "Examples:\n"
            "  ppno ppno/examples/example_1.ext\n"
            "  python -m ppno.ppno ppno/examples/example_1.ext"
        ),
    )


def _normalize_argv(argv: Optional[List[str]]) -> List[str]:
    """Return command arguments without the program name."""
    if argv is None:
        return sys.argv[1:]
    if argv and Path(argv[0]).name.lower() in {"ppno", "ppno.exe", "ppno.py"}:
        return argv[1:]
    return argv


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the PPNO command-line tool."""
    parser = build_parser()
    parser.add_argument(
        "problem_file",
        nargs="?",
        help="Path to the .ext problem definition file.",
    )

    args = parser.parse_args(_normalize_argv(argv))
    if args.problem_file is None:
        parser.print_help()
        sys.exit(0)

    logger.info("=" * 80)
    logger.info(" PRESSURIZED PIPE NETWORK OPTIMIZER ")
    logger.info("=" * 80)

    opt = None
    try:
        opt = Optimization(args.problem_file)
        solution = opt.solve()
        if solution is not None:
            opt.pretty_print(solution)
    except Exception:
        logger.exception("A fatal error occurred during optimization:")
        parser.print_help(sys.stderr)
        sys.exit(1)
    finally:
        if opt is not None:
            opt.close()


if __name__ == "__main__":
    main()
