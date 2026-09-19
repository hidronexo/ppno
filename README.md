# PPNO - Pressurized Pipe Network Optimizer

PPNO is a command-line optimizer for pressurized water distribution networks. It
uses EPANET as the hydraulic simulator and searches for cost-effective pipe
diameter assignments that satisfy minimum pressure constraints.

The optimizer is built around a two-stage workflow:

1. Stage 1 always runs a Unit Headloss heuristic followed by FLS-H local
   refinement. This produces a feasible baseline solution when the available
   pipe sizes and hydraulic model allow one.
2. Stage 2 runs only when requested in the `[OPTIONS]` section. It seeds global
   metaheuristics with the Stage 1 solution, then applies a final FLS-H polish.

## Features

- EPANET-based pressure feasibility checks over all hydraulic time steps.
- Mandatory Unit Headloss heuristic for fast feasible design.
- FLS-H local refinement with an evaluation cache.
- Optional Stage 2 global search with SciPy and PyGMO algorithms.
- Feasible-solution seeding for metaheuristics.
- `.scn` result files for successful Stage 2 algorithms.
- Semantic validation for missing entities, invalid pipe catalog entries, and option names.
- Multi-encoding support for `.ext` files: UTF-8, UTF-16, CP1252, then Latin-1 fallback.

## Installation

Install from source:

```bash
pip install .
```

For development:

```bash
pip install -e ".[dev]"
```

PyGMO is required by the code. On Linux, the normal install command installs
the full solver stack from PyPI: NumPy, SciPy, PyGMO, and entoolkit.

On Windows, PyGMO is not currently distributed as a PyPI wheel. Install PyGMO
in the Python environment first, typically with conda, then install PPNO:

```bash
conda install -c conda-forge pygmo
pip install .
```

## Usage

Run an optimization problem:

```bash
ppno path/to/problem.ext
```

Equivalent module form:

```bash
python -m ppno.ppno path/to/problem.ext
```

If no argument is provided, PPNO prints the command help and exits successfully.
Fatal optimization or validation errors exit with status code `1`.

## Output Files

PPNO writes scenario files next to the EPANET `.inp` model for successful Stage
2 algorithm runs:

```text
<inp_name>_result_<algorithm>.scn
```

Each `.scn` file stores a result scenario with `[DIAMETERS]` and `[ROUGHNESS]`
sections:

```ini
[DIAMETERS]
; Result for algorithm: DE
;Pipe            	Diameter
 1               	386.6000
 2               	289.9000


[ROUGHNESS]
;Pipe            	Roughness
 1               	130.000000
 2               	130.000000
```

The `.inp` file remains unchanged.

## Problem File Format

PPNO uses a plain-text `.ext` file with bracketed sections. Semicolon comments
are supported.

### `[INP]`

Required. Path to the EPANET `.inp` model. Relative paths are resolved from the
current working directory, the `.ext` file directory, or the `.ext` file
directory using only the `.inp` file name.

```ini
[INP]
./examples/HAN.inp
```

### `[OPTIONS]`

Optional. This section only accepts `Algorithm` or its alias `Algorithms`. If it
is omitted, empty, or does not list any Stage 2 algorithm, PPNO runs only Stage
1: Unit Headloss plus FLS-H refinement.

You may list one or more algorithms on the same line. Values can be separated by
spaces, commas, or tabs.

```ini
[OPTIONS]
Algorithm NSGA2 DE
```

Recognized option names:

- `Algorithm` or `Algorithms`

Solver tuning values are not read from `[OPTIONS]`. They are configured in
`ppno/constants.py`.

### Supported Algorithms

Stage 1:

- `UH` - Unit Headloss heuristic. This is always run internally and is ignored
  if listed under `Algorithm`.

Stage 2 with SciPy:

- `DE` - Differential Evolution
- `DA` - Dual Annealing

Stage 2 with PyGMO:

- `NSGA2` - Non-dominated Sorting Genetic Algorithm II
- `MOEAD` - Multi-objective Evolutionary Algorithm based on Decomposition
- `MACO` - Multi-objective Ant Colony Optimizer
- `PSO` - Non-dominated Sorting Particle Swarm Optimizer

### Solver Constants

Edit `ppno/constants.py` to adjust global solver behavior:

Global optimization parameters:

- `PENALTY_VALUE` - base penalty added to infeasible SciPy objective values.
- `MAX_RETRIES` - maximum attempts for each Stage 2 algorithm.
- `MAX_ALGORITHM_TIME` - time limit, in seconds, for each SciPy or PyGMO
  algorithm attempt.
- `RANDOM_SEED` - optional integer seed for NumPy and PyGMO; `None` leaves runs
  stochastic.
- `POPULATION_SIZE` - number of individuals in PyGMO populations.
- `GENERATIONS` - PyGMO generations per evolution trial.
- `PATIENCE` - PyGMO trials without improvement before convergence is assumed.
- `MAX_TRIALS` - maximum PyGMO evolution trials per algorithm attempt.

Local search (FLS-H) settings:

- `LS_MAX_ITER` - maximum iterations in the FLS-H refinement loop.
- `LS_ACCEPTANCE_THRESHOLD` - allowed temporary cost worsening in FLS-H, as a
  decimal fraction; `0.01` means 1%.
- `LS_NEIGHBORHOOD_SIZE` - candidate solutions generated per FLS-H iteration.

### `[PIPES]`

Required. Maps EPANET pipe/link IDs to PPNO pipe-size group names.

```ini
[PIPES]
1    PVC-SDR41
2    PVC-SDR41
```

The first column must match a link ID in the EPANET model. The second column
must match a group defined in the external pipe catalog.

### `[PRESSURES]`

Required. Minimum pressure constraints.

```ini
[PRESSURES]
2    30.0
3    30.0
```

The first column must match an EPANET node ID. The second column is the minimum
required pressure in the same units used by the EPANET model.

### `[PIPE_CATALOG]`

Required. Path to the external pipe catalog file. Relative paths are resolved
from the current working directory, the `.ext` file directory, or the `.ext`
file directory using only the catalog file name.

```ini
[PIPE_CATALOG]
example.cat
```

### Pipe Catalog (`.cat`)

Required. Plain text table with one pipe option per row. The file does not use
section headers.

```text
; group    diameter    roughness    unit price
PVC-SDR41     289.9    130.0     45.73
PVC-SDR41     386.6    130.0     70.40
```

Columns are:

1. group name
2. diameter
3. roughness
4. unit price

Diameters in each group must be strictly increasing. Prices are allowed to be
non-monotonic, but PPNO logs a warning when a larger diameter is not more
expensive.

## Example

```ini
[INP]
./examples/HAN.inp

[OPTIONS]
Algorithm DE NSGA2

[PIPE_CATALOG]
example.cat

[PIPES]
1    PVC-SDR41
2    PVC-SDR41

[PRESSURES]
2    30.0
3    30.0
```

`example.cat`:

```text
PVC-SDR41     289.9    130.0     45.73
PVC-SDR41     386.6    130.0     70.40
PVC-SDR41     483.2    130.0     98.39
```

More complete examples are available under `ppno/examples/`.

## Pipeline Details

### Stage 1: Unit Headloss and FLS-H

PPNO starts from the smallest available diameter for each pipe. It repeatedly runs
the hydraulic simulation, identifies pipes with the highest unit headloss, and
increases diameters until all pressure constraints are satisfied or all
available diameters are exhausted.

The resulting solution is refined with FLS-H, a feasible local search procedure
that tries to reduce cost while preserving hydraulic feasibility.

### Stage 2: Global Search

When algorithms are listed in `[OPTIONS]`, PPNO runs them after Stage 1. The
Stage 1 solution is used as a seed for compatible solvers. Each successful
candidate is compared against the current best solution, and the final best
solution is polished again with FLS-H.

## License and Citation

Apache License 2.0

If you use PPNO in research:

```text
García Martínez, A. (2019-2026).
PPNO: Pressurized Pipe Network Optimizer
https://github.com/andresgciamtez/ppno
```

## Interfaz gráfica

PPNO incluye una interfaz nativa para preparar, validar y ejecutar problemas de
optimización sin editar manualmente el fichero `.ext`. La aplicación mantiene el
flujo de trabajo del programa de consola: el modelo EPANET y el catálogo externo
siguen siendo las fuentes de datos, y se usa exactamente el mismo motor de
validación y optimización.

La interfaz está pensada para trabajar dentro del entorno Conda de PPNO. En
Windows, active el entorno, instale el proyecto en modo editable y arranque la
aplicación:

```powershell
conda activate ppno
python -m pip install -e ".[dev]"
ppno-gui
```

También puede iniciarse sin usar el comando instalado:

```powershell
conda activate ppno
python -m ppno.gui
```

El uso recomendado es el siguiente:

1. Cree un proyecto nuevo o abra un problema `.ext` existente.
2. Seleccione el modelo EPANET `.inp` y el catálogo de tuberías `.cat`. La GUI
   obtiene del INP los nudos y las tuberías, y muestra las alternativas del
   catálogo sin modificar ese fichero externo.
3. En **Tuberías**, active los conductos que se optimizarán y asigne su grupo de
   catálogo. En **Presiones**, active los nudos de control y defina la presión
   mínima. Se pueden seleccionar varias filas para aplicar valores en bloque.
4. Marque, si procede, los algoritmos opcionales de la etapa 2. La etapa 1
   (Unit Headloss y FLS-H) se ejecuta siempre.
5. Guarde y pulse **Validar**. La optimización permanece deshabilitada hasta que
   PPNO valida el `.ext`, el INP y el catálogo. Cualquier edición posterior exige
   validar de nuevo.
6. Pulse **Optimizar**. El cálculo se ejecuta en segundo plano; las pestañas
   **Ejecución** y **Solución** muestran respectivamente el registro, el resumen
   de algoritmos, el coste final y los diámetros seleccionados.

Los resultados `.scn` de las metaheurísticas se generan junto al modelo INP con
el mismo nombre y formato que en la interfaz de línea de comandos. La GUI no
crea ni utiliza un entorno virtual independiente: siempre ejecuta PPNO con el
intérprete del entorno Conda desde el que se inició.
