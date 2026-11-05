## v0.4.0 (2026-11-05)

### Feat

- **gui**: add structured PPNO desktop workflow

## v0.3.3 (2026-05-21)

### Feat

- expand configuration options, reorganize [OPTIONS] and remove RPT generation
- implement incremental .scn result saving and update to v0.3.3
- implement project structure including core solvers, section parser, and local refiner with unit tests
- implement SciPy and PyGMO solver modules for pipe network optimization
- implement SciPy-based global optimization solvers and add automated example execution scripts
- implement optimization solvers using SciPy and PyGMO libraries
- add PyGMO-based multi-objective optimization solver and update version to 0.3.2
- implement core pipe network optimization engine with multi-algorithm support
- implement MOEAD, MACO, and PSO (NSPSO) algorithms from PyGMO
- implement SHGO and DIRECT optimization algorithms from SciPy
- implement
- add project metadata, dependency requirements, and example files to support package distribution and documentation
- implement Python wrapper for EPANET toolkit with cross-platform binary support

### Fix

- correct pygmo_solver loop limits and resolve parser edge cases
- improve GA seeding logic and add explicit acceptance/discard logging
- improve GA seeding logic and add explicit acceptance/discard logging

### Refactor

- externalize pipe catalog and remove DIRECT
- rename pipe size groups and clarify outputs
- make pygmo required and refresh documentation
- overhaul docstrings, harden validation and format README
- remove installer build infrastructure and update installation documentation
- update installer build process to use pip wheel and add core optimization logic

## v0.1.8 (2019-11-21)

## v0.1.3 (2019-11-19)

## v0.1.2 (2019-11-07)

## v0.1.1 (2019-10-31)

## v0.1.0 (2019-09-20)

## v0.0.0 (2019-09-20)
