import ppno.constants as c


def test_algorithm_ids():
    assert c.ALGORITHM_UH == 0
    assert c.ALGORITHM_DE == 1
    assert c.ALGORITHM_DA == 2
    assert c.ALGORITHM_NSGA2 == 3
    assert c.ALGORITHM_MOEAD == 4
    assert c.ALGORITHM_MACO == 5
    assert c.ALGORITHM_PSO == 6

def test_params():
    assert c.PENALTY_VALUE == 1e9
    assert c.MAX_RETRIES == 3
    assert c.MAX_ALGORITHM_TIME == 120
    assert c.RANDOM_SEED is None
    assert c.POPULATION_SIZE == 100
    assert c.GENERATIONS == 100
    assert c.PATIENCE == 10
    assert c.MAX_TRIALS == 250

def test_ls_settings():
    assert c.LS_MAX_ITER == 50
    assert c.LS_ACCEPTANCE_THRESHOLD == 0.01
    assert c.LS_NEIGHBORHOOD_SIZE == 20
    assert c.DEFAULT_CONFIG['MaxTime'] == c.MAX_ALGORITHM_TIME
    assert c.DEFAULT_CONFIG['PopulationSize'] == c.POPULATION_SIZE
    assert c.DEFAULT_CONFIG['RefinerIters'] == c.LS_MAX_ITER
