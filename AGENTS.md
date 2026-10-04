# PPNO — agent instructions

Follow the central policy in `hidronexo-meta/AGENTS.md`.

- Visibility and license: public, `Apache-2.0` (see `repos.toml` in `hidronexo-meta`). Authors, commits, and public metadata use `HIDRONEXO <opensource@hidronexo.com>`.
- Core paths (no UI imports): `ppno/ppno.py`, `ppno/constants.py`, `ppno/local_refiner.py`, `ppno/pygmo_solver.py`, `ppno/scipy_solver.py`, `ppno/section_parser.py`.
- `ppno/gui.py` is the only Tk-dependent module; it may import the core, never the reverse.
- Runtime dependency: `entoolkit` (public). Keep `ppno` free of private runtime dependencies.
- Exception: `pygmo` is installed through conda-forge by Pixi; `pip install` skips it on Windows.
