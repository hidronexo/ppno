import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ppno import section_parser as sp
from ppno.local_refiner import LocalRefiner
from ppno.ppno import Optimization, main
from ppno.pygmo_solver import PPNOProblem, evolve_ppno


@pytest.fixture
def mock_et():
    with patch('ppno.ppno.et') as m:
        m.ENgetlinkindex.return_value = 1
        m.ENgetnodeindex.return_value = 1
        m.ENgetlinkvalue.return_value = 100.0
        m.ENgetnodevalue.return_value = 25.0
        m.ENopen.return_value = 0
        m.ENopenH.return_value = 0
        m.ENinitH.return_value = 0
        m.ENrunH.return_value = 0
        m.ENnextH.return_value = 0
        m.ENclose.return_value = 0
        m.ENsaveinpfile.return_value = 0
        m.ENreport.return_value = 0
        m.EN_LENGTH = 1
        m.EN_PRESSURE = 11
        m.EN_HEADLOSS = 4
        m.EN_DIAMETER = 0
        m.EN_ROUGHNESS = 1
        yield m

def test_ppno_missing_branches_corrected(mock_et, tmp_path):
    ext = tmp_path / "comp.ext"; (tmp_path / "t.inp").write_text("")
    # Valid minimal content
    ext.write_text("[INP]\nt.inp\n[PIPES]\np1 s1\n[PIPE_CATALOG]\npipe.cat\n[PRESSURES]\nn1 20")
    
    opt = Optimization(ext)
    assert opt.config['MaxTime'] == 120
    
    # Heuristic expansion failure
    opt.pipes = np.array([(0, 'p1', 100.0, 's1')], dtype=[('link_idx', 'i4'), ('id', 'U16'), ('length', 'f4'), ('group', 'U16')])
    opt.pipe_sizes = {'s1': np.sort(np.array([(100, 0.1, 10)], dtype=[('diameter', 'f4'), ('roughness', 'f4'), ('price', 'f4')]), order='diameter')}
    opt.dimension = 1
    opt.ubound = np.array([0])
    opt._current_x = np.zeros(1, dtype=int)
    
    with patch.object(Optimization, 'check', return_value=(False, [0])):
        assert opt._solve_uh() is None

def test_pygmo_evolve_recovers_after_failed_trial():
    opt = MagicMock()
    opt.pipes = np.array([(0, 'p1', 100.0, 's1')], dtype=[('link_idx', 'i4'), ('id', 'U16'), ('length', 'f4'), ('group', 'U16')])
    opt.pipe_sizes = {'s1': np.sort(np.array([(100, 0.1, 10), (200, 0.1, 20)], dtype=[('diameter', 'f4'), ('roughness', 'f4'), ('price', 'f4')]), order='diameter')}
    opt.lbound = np.array([0]); opt.ubound = np.array([1]); opt.dimension = 1
    opt.simulation_cycles = 0
    opt.config = {'PopulationSize': 100, 'MaxTrials': 2, 'Patience': 10, 'Generations': 100, 'MaxTime': 120}
    
    prob = PPNOProblem(opt)
    assert "Pressurized" in prob.get_name()
    
    m_pop = MagicMock()
    m_pop.get_f.return_value = np.array([[100, 0]])
    m_pop.get_x.return_value = np.array([[0]])
    m_pop.problem = MagicMock()

    with patch('ppno.pygmo_solver.pg') as mock_pg:
        m_alg = MagicMock()
        m_alg.evolve.side_effect = [Exception("Crash"), m_pop]
        mock_pg.algorithm.return_value = m_alg
        mock_pg.population.return_value = m_pop
        mock_pg.problem.return_value = m_pop.problem
        
        with patch('ppno.pygmo_solver.perf_counter', side_effect=range(10)):
             f, x = evolve_ppno(opt, lambda: None, "TEST")
             assert f[0] == 100.0

def test_validations(mock_et, tmp_path):
    ext = tmp_path / "valid.ext"; (tmp_path / "t.inp").write_text("")
    
    # 1. Missing INP
    ext.write_text("[OPTIONS]\nALGORITHM DE")
    with pytest.raises(ValueError, match="Mandatory .INP. section is missing"):
        Optimization(ext)
    
    # 2. Unknown Algorithm
    ext.write_text("[INP]\nt.inp\n[OPTIONS]\nALGORITHM MAGIC\n[PIPE_CATALOG]\npipe.cat")
    with pytest.raises(ValueError, match="Unknown algorithm 'MAGIC'"):
        Optimization(ext)
    
    # 3. Missing Pipe/Node
    ext.write_text("[INP]\nt.inp\n[PIPES]\nmissing_p s1\n[PIPE_CATALOG]\npipe.cat\n[PRESSURES]\nmissing_n 20")
    mock_et.ENgetlinkindex.side_effect = Exception("Not found")
    mock_et.ENgetnodeindex.side_effect = Exception("Not found")
    with pytest.raises(ValueError, match="Pipe 'missing_p' not found"):
        Optimization(ext)
    mock_et.ENgetlinkindex.side_effect = None
    mock_et.ENgetnodeindex.side_effect = None

    # 4. Monotonicity Diameter
    (tmp_path / "pipe.cat").write_text("s1 200 0.1 10\ns1 100 0.1 20\n")
    ext.write_text("[INP]\nt.inp\n[PIPES]\np1 s1\n[PIPE_CATALOG]\npipe.cat")
    with pytest.raises(ValueError, match="Diameter must be strictly increasing"):
        Optimization(ext)

    # 5. Price Anomaly (Warning)
    (tmp_path / "pipe.cat").write_text("s1 100 0.1 20\ns1 200 0.1 10\n")
    ext.write_text("[INP]\nt.inp\n[PIPES]\np1 s1\n[PIPE_CATALOG]\npipe.cat")
    with patch('ppno.ppno.logger.warning') as m_warn:
        Optimization(ext)
        m_warn.assert_called()

def test_cli_help_and_errors(tmp_path):
    with patch.object(sys, 'argv', ['ppno', '-h']), pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 0
    
    p = tmp_path / "fatal.ext"
    p.write_text("[INP]\nmissing.inp")
    with patch.object(sys, 'argv', ['ppno', str(p)]), pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1

def test_section_parser_extra(tmp_path):
    p = tmp_path / "enc.txt"
    p.write_text("[UTF16]\nData", encoding='utf-16')
    parser = sp.SectionParser(p)
    assert 'UTF16' in parser.read()
    
    p.write_bytes("[CP1252]\nccent".encode('cp1252'))
    assert 'CP1252' in parser.read()

    assert sp.SectionParser.line_to_tuple("a, b\tc") == ("a", "b", "c")
    assert sp.SectionParser.tuple_to_line(("a", 1)) == "a    1"

def test_optimization_full_flow(mock_et, tmp_path):
    ext = tmp_path / "full.ext"; (tmp_path / "t.inp").write_text("")
    (tmp_path / "pipe.cat").write_text("s1 100 0.1 20\ns1 200 0.1 10\n")
    ext.write_text("[INP]\nt.inp\n[OPTIONS]\nALGORITHMS DE\n[PIPES]\np1 s1\n[PIPE_CATALOG]\npipe.cat\n[PRESSURES]\nn1 20")
    
    opt = Optimization(ext)
    opt.set_x(np.array([0]))
    assert opt.get_cost() == 100.0 * 20.0 
    
    opt.check(mode='TF')
    opt.check(mode='UH')
    opt.check(mode='PD')
    
    with patch('ppno.scipy_solver.solve_scipy', return_value=np.array([1])):
        sol = opt.solve()
        assert sol[0] == 1
    
    with patch.object(Optimization, '_solve_uh', return_value=None):
        assert opt.solve() is None

def test_optimization_options_variants(mock_et, tmp_path):
    ext = tmp_path / "opt.ext"; (tmp_path / "t.inp").write_text("")
    ext.write_text("[INP]\nt.inp\n[OPTIONS]\nALGORITHMS DE PSO\n[PIPES]\np1 s1\n[PIPE_CATALOG]\npipe.cat\n[PRESSURES]\nn1 20")
    opt = Optimization(ext)
    assert len(opt.algorithms) == 2
    assert opt.config['MaxTime'] == 120

def test_all_pygmo_algorithms(mock_et, tmp_path):
    ext = tmp_path / "pygmo.ext"; (tmp_path / "t.inp").write_text("")
    ext.write_text("[INP]\nt.inp\n[PIPES]\np1 s1\n[PIPE_CATALOG]\npipe.cat\n[PRESSURES]\nn1 20")
    opt = Optimization(ext)
    
    from ppno import pygmo_solver
    with patch('ppno.pygmo_solver.evolve_ppno', return_value=([100.0, 0.0], np.array([0]))):
        pygmo_solver.nsga2(opt)
        pygmo_solver.moead(opt)
        pygmo_solver.maco(opt)
        pygmo_solver.nspso(opt)

def test_all_scipy_algorithms(mock_et, tmp_path):
    ext = tmp_path / "scipy.ext"; (tmp_path / "t.inp").write_text("")
    ext.write_text("[INP]\nt.inp\n[PIPES]\np1 s1\n[PIPE_CATALOG]\npipe.cat\n[PRESSURES]\nn1 20")
    opt = Optimization(ext)
    
    from ppno import scipy_solver
    from ppno.constants import ALGORITHM_DA, ALGORITHM_DE
    
    with patch('scipy.optimize.dual_annealing') as m_da, \
         patch('scipy.optimize.differential_evolution') as m_de:
        
        m_da.return_value = MagicMock(success=True, x=[0.0])
        m_de.return_value = MagicMock(success=True, x=[0.0])
        
        scipy_solver.solve_scipy(opt, ALGORITHM_DA)
        scipy_solver.solve_scipy(opt, ALGORITHM_DE)
        m_da.return_value.success = False
        assert np.array_equal(scipy_solver.solve_scipy(opt, ALGORITHM_DA), [0])

def test_scipy_timeout(mock_et, tmp_path):
    ext = tmp_path / "timeout.ext"; (tmp_path / "t.inp").write_text("")
    ext.write_text("[INP]\nt.inp\n[PIPES]\np1 s1\n[PIPE_CATALOG]\npipe.cat\n[PRESSURES]\nn1 20")
    opt = Optimization(ext)
    from ppno import scipy_solver
    from ppno.constants import ALGORITHM_DE
    
    with patch('ppno.scipy_solver.perf_counter', side_effect=[0, 1000000]), \
         patch('scipy.optimize.differential_evolution') as m_de:
        m_de.side_effect = lambda obj, *args, **kwargs: obj(np.array([0.0]))
        assert scipy_solver.solve_scipy(opt, ALGORITHM_DE) is None

def test_pygmo_problem_methods(mock_et):
    opt = MagicMock()
    opt.lbound = np.array([0]); opt.ubound = np.array([1]); opt.dimension = 1
    opt.config = {'PopulationSize': 100, 'MaxTrials': 2, 'Patience': 10, 'Generations': 100, 'MaxTime': 120}
    from ppno.pygmo_solver import PPNOProblem
    prob = PPNOProblem(opt)
    assert prob.get_nobj() == 2
    assert prob.get_nix() == 1
    assert prob.get_name() == "Pressurized Pipe Network Optimization (Multi-objective)"
    prob.get_bounds()
    
    opt.get_cost.return_value = 100.0
    opt.check.return_value = np.array([0.0])
    fit = prob.fitness(np.array([0.0]))
    assert fit == [100.0, 0.0]

def test_section_parser_end_tag(tmp_path):
    p = tmp_path / "end.txt"
    p.write_text("[S1]\n1\n[END]\n[S2]\n2")
    parser = sp.SectionParser(p)
    sections = parser.read()
    assert 'S1' in sections
    assert 'S2' not in sections

def test_section_parser_read_section(tmp_path):
    p = tmp_path / "sect.txt"
    p.write_text("[S1]\n1 2\n[S2]\n3 4")
    parser = sp.SectionParser(p)
    assert len(parser.read_section("S1")) == 1
    assert parser.read_section("S3") == []

def test_print_methods(mock_et, tmp_path, caplog):
    ext = tmp_path / "print.ext"; (tmp_path / "t.inp").write_text("")
    ext.write_text("[INP]\nt.inp\n[PIPES]\np1 s1\n[PIPE_CATALOG]\npipe.cat\n[PRESSURES]\nn1 20")
    opt = Optimization(ext)
    opt.results = [{'Algorithm': 'Test', 'Attempt': 1, 'Success': 'YES', 'Time (s)': '1.0', 'Simulations': 10, 'Cost': '100.0'}]
    opt._print_summary()
    opt.pretty_print(np.array([0]))
    opt.close()

def test_local_refiner_improvement(mock_et):
    opt = MagicMock()
    opt.pipes = np.array([(0, 'p1', 100.0, 's1')], dtype=[('link_idx', 'i4'), ('id', 'U16'), ('length', 'f4'), ('group', 'U16')])
    opt.pipe_sizes = {'s1': np.sort(np.array([(100, 0.1, 20), (200, 0.1, 10)], dtype=[('diameter', 'f4'), ('roughness', 'f4'), ('price', 'f4')]), order='diameter')}
    opt.lbound = np.array([0])
    opt.ubound = np.array([1])
    opt.get_x.return_value = np.array([0])
    opt.check.return_value = True
    opt.get_cost.side_effect = [2000.0, 1000.0, 1000.0]
    
    with patch('numpy.random.random', return_value=1.0), \
         patch('numpy.random.randint', return_value=1):
        refiner = LocalRefiner(opt, {'max_iter': 1, 'acceptance_threshold': 0.05, 'neighborhood_size': 1})
        sol = refiner.refine(np.array([0]))
        assert sol[0] == 1

def test_ppno_validation_errors(mock_et, tmp_path):
    ext = tmp_path / "errors.ext"; (tmp_path / "t.inp").write_text("")
    # 1. Invalid pipe definition
    ext.write_text("[INP]\nt.inp\n[PIPES]\np1\n[PIPE_CATALOG]\npipe.cat\n[PRESSURES]\nn1 20")
    with pytest.raises(ValueError, match="Invalid pipe definition"):
        Optimization(ext)
    # 2. group not in pipe sizes
    ext.write_text("[INP]\nt.inp\n[PIPES]\np1 s2\n[PIPE_CATALOG]\npipe.cat\n[PRESSURES]\nn1 20")
    with pytest.raises(ValueError, match="Pipe-size group .s2. not defined in pipe catalog"):
        Optimization(ext)
    # 3. Invalid pressure definition
    ext.write_text("[INP]\nt.inp\n[PIPES]\np1 s1\n[PIPE_CATALOG]\npipe.cat\n[PRESSURES]\nn1")
    with pytest.raises(ValueError, match="Invalid pressure definition"):
        Optimization(ext)
    # 4. Invalid pressure value
    ext.write_text("[INP]\nt.inp\n[PIPES]\np1 s1\n[PIPE_CATALOG]\npipe.cat\n[PRESSURES]\nn1 X")
    with pytest.raises(ValueError, match="Invalid pressure value .X."):
        Optimization(ext)

def test_ppno_file_not_found():
    with pytest.raises(FileNotFoundError, match="Problem file not found"):
        Optimization("non_existent.ext")

def test_scipy_objective_penalty(mock_et, tmp_path):
    ext = tmp_path / "penalty.ext"; (tmp_path / "t.inp").write_text("")
    ext.write_text("[INP]\nt.inp\n[PIPES]\np1 s1\n[PIPE_CATALOG]\npipe.cat\n[PRESSURES]\nn1 20")
    opt = Optimization(ext)
    from ppno import scipy_solver
    from ppno.constants import ALGORITHM_DE
    
    with patch.object(Optimization, 'check', return_value=np.array([5.0])):
        with patch('scipy.optimize.differential_evolution') as m_de:
            m_de.side_effect = lambda obj, *args, **kwargs: MagicMock(success=True, x=np.array([0.0]), fun=obj(np.array([0.0])))
            scipy_solver.solve_scipy(opt, ALGORITHM_DE)

def test_new_robust_validations(mock_et, tmp_path):
    ext = tmp_path / "new_errors.ext"; (tmp_path / "t.inp").write_text("")
    
    # 1. Unsupported numeric option
    ext.write_text("[INP]\nt.inp\n[OPTIONS]\nMAXRETRIES abc\n[PIPE_CATALOG]\npipe.cat")
    with pytest.raises(ValueError, match="Unsupported option 'MAXRETRIES'"):
        Optimization(ext)
        
    # 2. Unsupported refiner option
    ext.write_text("[INP]\nt.inp\n[OPTIONS]\nREFINER_WORSENING xyz\n[PIPE_CATALOG]\npipe.cat")
    with pytest.raises(ValueError, match="Unsupported option 'REFINER_WORSENING'"):
        Optimization(ext)
        
    # 3. Malformed pipe sizes line (too short)
    (tmp_path / "pipe.cat").write_text("s1 100 0.1\n")
    ext.write_text("[INP]\nt.inp\n[PIPES]\np1 s1\n[PIPE_CATALOG]\npipe.cat\n[PRESSURES]\nn1 20")
    with pytest.raises(ValueError, match="Invalid pipe catalog entry"):
        Optimization(ext)

    # 4. Invalid numeric value in pipe sizes
    (tmp_path / "pipe.cat").write_text("s1 100 0.1 error\n")
    ext.write_text("[INP]\nt.inp\n[PIPES]\np1 s1\n[PIPE_CATALOG]\npipe.cat\n[PRESSURES]\nn1 20")
    with pytest.raises(ValueError, match="Invalid numeric values in pipe catalog group 's1'"):
        Optimization(ext)



