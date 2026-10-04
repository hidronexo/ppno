from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ppno.pygmo_solver import PPNOProblem, evolve_ppno, maco, moead, nsga2, nspso


@pytest.fixture(autouse=True)
def mock_pg():
    with patch('ppno.pygmo_solver.pg', new_callable=MagicMock) as m:
        yield m

@pytest.fixture
def mock_opt():
    opt = MagicMock()
    opt.pipes = np.array([(0, 'p1', 100.0, 's1')], dtype=[('link_idx', 'i4'), ('id', 'U16'), ('length', 'f4'), ('group', 'U16')])
    # pipe sizes with 2 entries to allow variety
    opt.pipe_sizes = {'s1': np.array([(100.0, 0.1, 10.0), (200.0, 0.1, 25.0)], dtype=[('diameter', 'f4'), ('roughness', 'f4'), ('price', 'f4')])}
    opt.lbound = np.array([0])
    opt.ubound = np.array([1])
    opt.simulation_cycles = 10
    opt.dimension = 1
    opt.config = {
        'PopulationSize': 100,
        'MaxTrials': 250,
        'Patience': 10,
        'Generations': 100,
        'MaxTime': 120
    }
    return opt

def test_ppno_problem_getters(mock_opt):
    problem = PPNOProblem(mock_opt)
    assert problem.get_nobj() == 2
    assert problem.get_nix() == 1

def test_ppno_problem_fitness(mock_opt):
    problem = PPNOProblem(mock_opt)
    mock_opt.get_cost.return_value = 500.0
    mock_opt.check.return_value = np.array([0.0])
    f = problem.fitness([0.0])
    assert f[0] == 500.0
    
    # Test min_deficit logic (line 62)
    mock_opt.check.return_value = np.array([5.0, 2.0])
    f_infeasible = problem.fitness([0.0])
    assert f_infeasible[1] == 5.0 # Max deficit

def test_evolve_ppno_complete_loop_logic(mock_opt, mock_pg):
    m_prob = MagicMock()
    mock_pg.problem.return_value = m_prob
    
    m_pop = MagicMock()
    # Varying fitness across trials: 100, 50, 50, ...
    m_pop.get_f.side_effect = [np.array([[100.0, 0.0]]), np.array([[50.0, 0.0]]), np.array([[50.0, 0.0]])]
    m_pop.get_x.side_effect = [np.array([[0]]), np.array([[1]]), np.array([[1]])]
    m_pop.problem = m_prob
    
    m_alg = MagicMock()
    # Provide plenty of pops to avoid StopIteration
    m_alg.evolve.return_value = m_pop
    mock_pg.algorithm.return_value = m_alg
    mock_pg.population.return_value = m_pop
    
    with patch('ppno.pygmo_solver.perf_counter', side_effect=range(100)):
        mock_opt.config['MaxTrials'] = 2
        f, x = evolve_ppno(mock_opt, lambda: MagicMock(), "TEST")
        assert f[0] == 50.0

def test_algorithm_wrappers_execution(mock_opt, mock_pg):
    m_prob = MagicMock()
    mock_pg.problem.return_value = m_prob
    m_pop = MagicMock()
    m_pop.get_f.return_value = np.array([[100.0, 0.0]])
    m_pop.get_x.return_value = np.array([[0]])
    m_pop.problem = m_prob
    mock_pg.population.return_value = m_pop
    m_alg = MagicMock()
    m_alg.evolve.return_value = m_pop
    mock_pg.algorithm.return_value = m_alg
    
    with patch('ppno.pygmo_solver.perf_counter', side_effect=range(1000)):
        mock_opt.config['MaxTrials'] = 1
        nsga2(mock_opt)
        assert mock_pg.nsga2.called
        moead(mock_opt)
        assert mock_pg.moead.called
        maco(mock_opt)
        assert mock_pg.maco.called
        nspso(mock_opt)
        assert mock_pg.nspso.called

def test_evolve_ppno_early_exit(mock_opt, mock_pg):
    # Coverage for the patience-based convergence break
    m_prob = MagicMock()
    mock_pg.problem.return_value = m_prob
    m_pop = MagicMock()
    m_pop.get_f.return_value = np.array([[100.0, 0.0]])
    m_pop.get_x.return_value = np.array([[0]])
    m_pop.problem = m_prob
    mock_pg.population.return_value = m_pop
    m_alg = MagicMock()
    m_alg.evolve.return_value = m_pop
    mock_pg.algorithm.return_value = m_alg
    
    with patch('ppno.pygmo_solver.perf_counter', side_effect=range(100)):
        mock_opt.config['Patience'] = 1
        evolve_ppno(mock_opt, lambda: MagicMock(), "TEST")
        assert m_alg.evolve.called
