from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ppno.local_refiner import LocalRefiner


@pytest.fixture
def mock_sim():
    sim = MagicMock()
    size = 10
    sim.pipes = np.array([
        (i, f'p{i}', 100.0, 's1') for i in range(size)
    ], dtype=[('link_idx', 'i4'), ('id', 'U16'), ('length', 'f4'), ('group', 'U16')])
    
    sim.pipe_sizes = {
        's1': np.array([
            (100.0, 0.1, 10.0),
            (200.0, 0.1, 25.0)
        ], dtype=[('diameter', 'f4'), ('roughness', 'f4'), ('price', 'f4')])
    }
    sim.lbound = np.zeros(size, dtype=np.int32)
    sim.ubound = np.ones(size, dtype=np.int32)
    return sim

def test_init():
    sim = MagicMock()
    refiner = LocalRefiner(sim, {'max_iter': 10, 'acceptance_threshold': 0.05, 'neighborhood_size': 5})
    assert refiner.max_iter == 10

def test_evaluate_cache(mock_sim):
    refiner = LocalRefiner(mock_sim)
    x = np.zeros(10, dtype=np.int32)
    mock_sim.get_cost.return_value = 1000.0
    mock_sim.check.return_value = True
    
    res1 = refiner.evaluate(x)
    mock_sim.get_cost.reset_mock()
    res2 = refiner.evaluate(x)
    assert res2 == res1
    mock_sim.get_cost.assert_not_called()

def test_repair(mock_sim):
    refiner = LocalRefiner(mock_sim)
    x = np.array([-1, 10] + [0]*8)
    repaired = refiner.repair(x)
    assert repaired[0] == 0
    assert repaired[1] == 1

def test_is_promising(mock_sim):
    refiner = LocalRefiner(mock_sim, {'acceptance_threshold': 0.1})
    x_cheap = np.zeros(10, dtype=np.int32)
    current_cost = 15000.0
    assert refiner.is_promising(x_cheap, current_cost) is True

def test_accept_or_reject(mock_sim):
    refiner = LocalRefiner(mock_sim, {'acceptance_threshold': 0.1})
    curr_x = np.zeros(10, dtype=np.int32)
    curr_cost = 100.0
    new_x = np.ones(10, dtype=np.int32)
    eval_better = {'cost': 90.0, 'feasible': True}
    res_x, res_cost = refiner.accept_or_reject(curr_x, curr_cost, new_x, eval_better)
    assert res_cost == 90.0

def test_refine_infeasible_start(mock_sim):
    refiner = LocalRefiner(mock_sim)
    x0 = np.zeros(10, dtype=np.int32)
    with patch.object(LocalRefiner, 'evaluate', side_effect=[{'cost': 100, 'feasible': False}, {'cost': 100, 'feasible': False}]):
        refined = refiner.refine(x0)
        assert np.array_equal(refined, x0)

def test_refine_loop_stagnation(mock_sim):
    # Coverage for lines 79-80 and 90-91 (diversify/continue)
    refiner = LocalRefiner(mock_sim, {'max_iter': 2, 'neighborhood_size': 1})
    x0 = np.zeros(10, dtype=np.int32)
    
    with patch.object(LocalRefiner, 'evaluate', return_value={'cost': 1000, 'feasible': True}), \
         patch.object(LocalRefiner, 'is_promising', side_effect=[False, True]), \
         patch.object(LocalRefiner, 'generate_neighborhood', return_value=[np.ones(10, dtype=np.int32)]), \
         patch.object(LocalRefiner, 'diversify', side_effect=[np.ones(10, dtype=np.int32), np.zeros(10, dtype=np.int32)]):
        
        # Iteration 1: is_promising False -> trigger diversification (79-80)
        # Iteration 2: eval results empty? We mock evaluate to ensure feasibility logic
        with patch.object(LocalRefiner, 'evaluate', side_effect=[{'cost': 1000, 'feasible': True}, {'cost': 1100, 'feasible': False}]):
            # First evaluate is for initial x0. Second is for candidate.
            refiner.refine(x0)

def test_diversify_guaranteed(mock_sim):
    refiner = LocalRefiner(mock_sim)
    x = np.zeros(10, dtype=np.int32)
    # Patch random to ensure jump is positive and idx is selected
    with patch('ppno.local_refiner.np.random.random', return_value=0.6), \
         patch('ppno.local_refiner.np.random.randint', return_value=1):
        div = refiner.diversify(x)
        assert not np.array_equal(div, x)
