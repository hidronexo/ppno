"""Native structured editor and runner for PPNO optimization projects."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import queue
import tempfile
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .ppno import Optimization
from .section_parser import SectionParser


ALGORITHMS = ("DE", "DA", "NSGA2", "MOEAD", "MACO", "PSO")
TEMPLATE = """; Proyecto PPNO
[TITLE]
Nuevo problema de optimización

[INP]

[OPTIONS]
; Algorithm DE DA NSGA2 MOEAD MACO PSO

[PIPE_CATALOG]

[PIPES]

[PRESSURES]

[END]
"""


def read_text(path: Path) -> str:
    """Read a PPNO text input using the same encoding fallback as its parser."""
    raw = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", raw, 0, 1, "unsupported encoding")


def save_text(path: Path, text: str) -> None:
    """Atomically replace a UTF-8 text document."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _data_lines(values: Iterable[Sequence[object]]) -> List[str]:
    return ["    ".join(str(value) for value in row) for row in values]


def render_project(
    source: str,
    inp_reference: str,
    catalog_reference: str,
    algorithms: Sequence[str],
    pipes: Sequence[Tuple[str, str]],
    pressures: Sequence[Tuple[str, str]],
) -> str:
    """Update the editable sections while retaining comments and unknown sections."""
    replacements = {
        "[INP]": [inp_reference],
        "[OPTIONS]": ["Algorithm " + " ".join(algorithms)] if algorithms else [],
        "[PIPE_CATALOG]": [catalog_reference],
        "[PIPES]": _data_lines(pipes),
        "[PRESSURES]": _data_lines(pressures),
    }
    output: List[str] = []
    seen = set()
    section: Optional[str] = None
    inserted = False

    for raw in source.splitlines():
        data = raw.split(";", 1)[0].strip()
        if data.startswith("[") and data.endswith("]"):
            if section in replacements and not inserted:
                output.extend(replacements[section])
            section = data.upper()
            inserted = False
            output.append(raw)
            if section in replacements:
                seen.add(section)
            continue

        if section in replacements and data:
            # Standalone and trailing comments survive structured editing.
            if ";" in raw:
                output.append(";" + raw.split(";", 1)[1])
            if not inserted:
                output.extend(replacements[section])
                inserted = True
            continue
        output.append(raw)

    if section in replacements and not inserted:
        output.extend(replacements[section])

    end_index = next(
        (i for i, line in enumerate(output) if line.split(";", 1)[0].strip().upper() == "[END]"),
        len(output),
    )
    missing: List[str] = []
    for header, values in replacements.items():
        if header not in seen:
            missing.extend([header, *values, ""])
    if missing:
        output[end_index:end_index] = missing
    if not any(line.split(";", 1)[0].strip().upper() == "[END]" for line in output):
        output.extend(["", "[END]"])
    return "\n".join(output).rstrip() + "\n"


def _section_rows(path: Path, section: str) -> List[Tuple[str, ...]]:
    parser = SectionParser(path)
    return [parser.line_to_tuple(value) for _, value in parser.read().get(section, [])]


def network_ids(path: Path) -> Tuple[List[str], List[str]]:
    """Return junction and pipe IDs from an EPANET INP without opening the toolkit."""
    result: Dict[str, List[str]] = {"JUNCTIONS": [], "PIPES": []}
    seen = {name: set() for name in result}
    section = ""
    for raw in read_text(Path(path)).splitlines():
        data = raw.split(";", 1)[0].strip()
        if not data:
            continue
        if data.startswith("[") and data.endswith("]"):
            section = data.strip("[] ").upper()
            continue
        if section in result:
            identifier = data.split()[0]
            if identifier in seen[section]:
                raise ValueError(f"Identificador duplicado en [{section}]: {identifier}")
            seen[section].add(identifier)
            result[section].append(identifier)
    if not result["JUNCTIONS"] or not result["PIPES"]:
        raise ValueError("El INP debe contener las secciones [JUNCTIONS] y [PIPES].")
    return result["JUNCTIONS"], result["PIPES"]


def catalog_rows(path: Path) -> List[Tuple[str, str, str, str]]:
    """Read the four visible columns of a PPNO pipe catalog."""
    parser = SectionParser(path)
    rows: List[Tuple[str, str, str, str]] = []
    for line_number, raw in parser.read_rows():
        values = parser.line_to_tuple(raw)
        if len(values) != 4:
            raise ValueError(
                f"Línea {line_number}: se esperaban grupo, diámetro, rugosidad y precio."
            )
        try:
            float(values[1])
            float(values[2])
            float(values[3])
        except ValueError as exc:
            raise ValueError(f"Línea {line_number}: valores numéricos no válidos.") from exc
        rows.append((values[0], values[1], values[2], values[3]))
    if not rows:
        raise ValueError("El catálogo no contiene alternativas de tubería.")
    return rows


def resolve_reference(reference: str, project_path: Path) -> Path:
    """Resolve a project reference with the same practical fallbacks as PPNO."""
    candidate = Path(reference)
    if candidate.is_absolute():
        return candidate
    choices = (
        Path.cwd() / candidate,
        project_path.parent / candidate,
        project_path.parent / candidate.name,
    )
    return next((path for path in choices if path.exists()), choices[-1])


def portable_reference(path: Path, project_path: Path) -> str:
    """Prefer a project-relative reference and use forward slashes in the document."""
    try:
        return Path(os.path.relpath(path, project_path.parent)).as_posix()
    except ValueError:
        return str(path)


def project_payload(path: Path) -> Dict[str, object]:
    """Load the fields edited by the GUI from an existing .ext project."""
    path = Path(path)
    parser = SectionParser(path)
    sections = parser.read()

    def first_value(name: str) -> str:
        values = sections.get(name, [])
        return values[0][1] if values else ""

    algorithms: List[str] = []
    for _, raw in sections.get("OPTIONS", []):
        tokens = [token for token in parser.line_to_tuple(raw) if token != "="]
        if tokens and tokens[0].upper().replace("_", "") in {"ALGORITHM", "ALGORITHMS"}:
            algorithms.extend(token.upper() for token in tokens[1:])

    pipes = []
    for _, raw in sections.get("PIPES", []):
        tokens = parser.line_to_tuple(raw)
        if len(tokens) >= 2:
            pipes.append((tokens[0], tokens[1]))
    pressures = []
    for _, raw in sections.get("PRESSURES", []):
        tokens = parser.line_to_tuple(raw)
        if len(tokens) >= 2:
            pressures.append((tokens[0], tokens[1]))

    inp_raw = first_value("INP")
    catalog_raw = first_value("PIPE_CATALOG")
    return {
        "source": read_text(path),
        "inp": resolve_reference(inp_raw, path) if inp_raw else None,
        "catalog": resolve_reference(catalog_raw, path) if catalog_raw else None,
        "algorithms": list(dict.fromkeys(algorithms)),
        "pipes": pipes,
        "pressures": pressures,
    }


class QueueLogHandler(logging.Handler):
    """Forward formatted optimization log records to the Tk event queue."""

    def __init__(self, events: "queue.Queue[Tuple[str, object]]") -> None:
        super().__init__()
        self.events = events
        self.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self.events.put(("log", self.format(record)))


def run_optimization(path: Path, events: "queue.Queue[Tuple[str, object]]") -> None:
    """Run PPNO off the Tk thread and return a structured result through events."""
    handler = QueueLogHandler(events)
    root_logger = logging.getLogger()
    old_level = root_logger.level
    if old_level > logging.INFO:
        root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)
    optimization: Optional[Optimization] = None
    try:
        optimization = Optimization(path)
        solution = optimization.solve()
        if solution is None:
            events.put(("error", "No se encontró una solución hidráulicamente factible."))
            return
        optimization.pretty_print(solution)
        rows = []
        for index, pipe in enumerate(optimization.pipes):
            size_index = int(solution[index])
            item = optimization.pipe_sizes[str(pipe["group"])][size_index]
            length = float(pipe["length"])
            price = float(item["price"])
            rows.append(
                (
                    str(pipe["id"]),
                    str(pipe["group"]),
                    f"{float(item['diameter']):.4f}",
                    f"{float(item['roughness']):.6f}",
                    f"{length:.2f}",
                    f"{price:.2f}",
                    f"{length * price:.2f}",
                )
            )
        payload = {
            "best_algorithm": optimization.best_algorithm_name,
            "cost": optimization.get_cost(),
            "results": [dict(item) for item in optimization.results],
            "pipes": rows,
        }
        events.put(("done", payload))
    except Exception as exc:
        logging.getLogger(__name__).exception("La optimización terminó con un error")
        events.put(("error", str(exc)))
    finally:
        if optimization is not None:
            optimization.close()
        root_logger.removeHandler(handler)
        root_logger.setLevel(old_level)


class ProjectEditor:
    """Structured Tk editor with explicit edit, validation and results modes."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.path: Optional[Path] = None
        self.source = TEMPLATE
        self.dirty = False
        self.validated_fingerprint: Optional[str] = None
        self.running = False
        self.events: "queue.Queue[Tuple[str, object]]" = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.inp_path: Optional[Path] = None
        self.catalog_path: Optional[Path] = None
        self.algorithms = {name: tk.BooleanVar(value=False) for name in ALGORITHMS}
        self.status = tk.StringVar(value="Cree o abra un problema PPNO.")
        self.cost_text = tk.StringVar(value="Sin resultados")

        root.title("Hidronexo PPNO - Optimizador de redes a presión")
        root.geometry("1120x760")
        root.minsize(880, 600)
        root.protocol("WM_DELETE_WINDOW", self.close)

        self._build_menu()
        self.edit_toolbar = ttk.Frame(root, padding=8)
        self.edit_toolbar.pack(fill="x")
        for label, command in (
            ("Nuevo", self.new),
            ("Abrir .ext…", self.open),
            ("Guardar", self.save),
            ("Guardar como…", self.save_as),
            ("Validar", self.validate),
        ):
            ttk.Button(self.edit_toolbar, text=label, command=command).pack(side="left", padx=3)

        self.result_toolbar = ttk.Frame(root, padding=8)
        ttk.Button(self.result_toolbar, text="Volver a edición", command=self.back_to_edit).pack(
            side="left", padx=3
        )
        self.run_button = ttk.Button(
            self.result_toolbar, text="Optimizar", command=self.optimize, state="disabled"
        )
        self.run_button.pack(side="left", padx=3)

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=5)
        self._build_project_tab()
        self.pipe_tree = self._build_assignment_tab(
            "Tuberías", ("use", "id", "group"), (70, 220, 300)
        )
        self.pressure_tree = self._build_assignment_tab(
            "Presiones", ("use", "id", "pressure"), (70, 220, 220)
        )
        self._build_catalog_tab()
        self.edit_tabs = list(self.notebook.tabs())
        self._build_result_tabs()
        self.back_to_edit(initial=True)

        ttk.Label(root, textvariable=self.status, relief="sunken", anchor="w", padding=5).pack(
            fill="x"
        )

    def _build_menu(self) -> None:
        menu = tk.Menu(self.root)
        file_menu = tk.Menu(menu, tearoff=False)
        file_menu.add_command(label="Nuevo", command=self.new)
        file_menu.add_command(label="Abrir…", command=self.open)
        file_menu.add_separator()
        file_menu.add_command(label="Guardar", command=self.save)
        file_menu.add_command(label="Guardar como…", command=self.save_as)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.close)
        menu.add_cascade(label="Archivo", menu=file_menu)
        run_menu = tk.Menu(menu, tearoff=False)
        run_menu.add_command(label="Validar", command=self.validate)
        run_menu.add_command(label="Optimizar", command=self.optimize)
        menu.add_cascade(label="Ejecutar", menu=run_menu)
        self.root.config(menu=menu)

    def _build_project_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=18)
        self.notebook.add(tab, text="Proyecto")
        self.inp_text = tk.StringVar(value="No seleccionado")
        self.catalog_text = tk.StringVar(value="No seleccionado")
        for row, (label, variable, command) in enumerate(
            (
                ("Modelo EPANET (.inp)", self.inp_text, self.choose_inp),
                ("Catálogo de tuberías (.cat)", self.catalog_text, self.choose_catalog),
            )
        ):
            ttk.Label(tab, text=label).grid(row=row, column=0, sticky="w", pady=9)
            ttk.Entry(tab, textvariable=variable, state="readonly").grid(
                row=row, column=1, sticky="ew", padx=12
            )
            ttk.Button(tab, text="Seleccionar…", command=command).grid(row=row, column=2)
        tab.columnconfigure(1, weight=1)

        algorithms = ttk.LabelFrame(tab, text="Etapa 2 - metaheurísticas opcionales", padding=12)
        algorithms.grid(row=2, column=0, columnspan=3, sticky="ew", pady=18)
        for column, name in enumerate(ALGORITHMS):
            ttk.Checkbutton(
                algorithms, text=name, variable=self.algorithms[name], command=self.mark_dirty
            ).grid(row=0, column=column, padx=10, pady=4)
        ttk.Label(
            tab,
            text=(
                "La etapa 1 (Unit Headloss + FLS-H) siempre se ejecuta. "
                "Las metaheurísticas seleccionadas se añaden como etapa 2."
            ),
            wraplength=760,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=6)

    def _build_assignment_tab(
        self, title: str, columns: Sequence[str], widths: Sequence[int]
    ) -> ttk.Treeview:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text=title)
        frame = ttk.Frame(tab)
        frame.pack(fill="both", expand=True)
        tree = ttk.Treeview(frame, columns=columns, show="headings", selectmode="extended")
        labels = {
            "use": "Usar",
            "id": "ID EPANET",
            "group": "Grupo de catálogo",
            "pressure": "Presión mínima",
        }
        for name, width in zip(columns, widths):
            tree.heading(name, text=labels[name])
            tree.column(name, width=width, anchor="center" if name == "use" else "w")
        scroll = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        tree.bind("<Double-1>", lambda event, widget=tree: self._edit_tree_cell(widget, event))
        tree.bind("<space>", lambda event, widget=tree: self._toggle_selected(widget))

        controls = ttk.Frame(tab, padding=(0, 8))
        controls.pack(fill="x")
        ttk.Button(controls, text="Activar selección", command=lambda: self._set_use(tree, True)).pack(
            side="left", padx=3
        )
        ttk.Button(
            controls, text="Desactivar selección", command=lambda: self._set_use(tree, False)
        ).pack(side="left", padx=3)
        if "group" in columns:
            self.bulk_group = tk.StringVar()
            self.group_combo = ttk.Combobox(
                controls, textvariable=self.bulk_group, state="readonly", width=24
            )
            self.group_combo.pack(side="left", padx=(18, 3))
            ttk.Button(controls, text="Aplicar grupo", command=self.apply_group).pack(side="left")
        else:
            self.bulk_pressure = tk.StringVar(value="30.0")
            ttk.Entry(controls, textvariable=self.bulk_pressure, width=12).pack(
                side="left", padx=(18, 3)
            )
            ttk.Button(controls, text="Aplicar presión", command=self.apply_pressure).pack(side="left")
        return tree

    def _build_catalog_tab(self) -> None:
        tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(tab, text="Catálogo")
        columns = ("group", "diameter", "roughness", "price")
        self.catalog_tree = ttk.Treeview(tab, columns=columns, show="headings")
        for name, label in zip(
            columns, ("Grupo", "Diámetro", "Rugosidad", "Precio unitario")
        ):
            self.catalog_tree.heading(name, text=label)
            self.catalog_tree.column(name, width=190, anchor="w")
        scroll = ttk.Scrollbar(tab, orient="vertical", command=self.catalog_tree.yview)
        self.catalog_tree.configure(yscrollcommand=scroll.set)
        self.catalog_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _build_result_tabs(self) -> None:
        run_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(run_tab, text="Ejecución")
        self.log = ScrolledText(run_tab, wrap="none", font=("Consolas", 10), state="disabled")
        self.log.pack(fill="both", expand=True)
        self.run_tab = str(run_tab)

        result_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(result_tab, text="Solución")
        ttk.Label(result_tab, textvariable=self.cost_text, font=("Segoe UI", 11, "bold")).pack(
            anchor="w", pady=(0, 8)
        )
        panes = ttk.Panedwindow(result_tab, orient="vertical")
        panes.pack(fill="both", expand=True)
        summary_frame = ttk.LabelFrame(panes, text="Resumen de algoritmos", padding=6)
        solution_frame = ttk.LabelFrame(panes, text="Diámetros seleccionados", padding=6)
        panes.add(summary_frame, weight=1)
        panes.add(solution_frame, weight=3)
        summary_columns = ("algorithm", "attempt", "success", "time", "simulations", "cost")
        self.summary_tree = ttk.Treeview(summary_frame, columns=summary_columns, show="headings")
        for name, label in zip(
            summary_columns, ("Algoritmo", "Intento", "Éxito", "Tiempo (s)", "Simulaciones", "Coste")
        ):
            self.summary_tree.heading(name, text=label)
            self.summary_tree.column(name, width=120, anchor="center")
        self.summary_tree.pack(fill="both", expand=True)

        pipe_columns = ("id", "group", "diameter", "roughness", "length", "price", "total")
        self.solution_tree = ttk.Treeview(solution_frame, columns=pipe_columns, show="headings")
        for name, label in zip(
            pipe_columns, ("Tubería", "Grupo", "Diámetro", "Rugosidad", "Longitud", "Precio", "Total")
        ):
            self.solution_tree.heading(name, text=label)
            self.solution_tree.column(name, width=120, anchor="e" if name not in {"id", "group"} else "w")
        scroll = ttk.Scrollbar(solution_frame, orient="vertical", command=self.solution_tree.yview)
        self.solution_tree.configure(yscrollcommand=scroll.set)
        self.solution_tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self.result_tab = str(result_tab)

    def mark_dirty(self) -> None:
        self.dirty = True
        self.validated_fingerprint = None
        self.run_button.configure(state="disabled")
        self._update_title()

    def _update_title(self) -> None:
        name = self.path.name if self.path else "sin título.ext"
        marker = " *" if self.dirty else ""
        self.root.title(f"Hidronexo PPNO - {name}{marker}")

    def _clear_tree(self, tree: ttk.Treeview) -> None:
        tree.delete(*tree.get_children())

    def _tree_values(self, tree: ttk.Treeview, selected_only: bool = False) -> List[Tuple[str, ...]]:
        rows = []
        for item in tree.get_children():
            values = tuple(str(value) for value in tree.item(item, "values"))
            if not selected_only or (values and values[0] == "✓"):
                rows.append(values)
        return rows

    def _populate_network(
        self,
        selected_pipes: Optional[Dict[str, str]] = None,
        selected_pressures: Optional[Dict[str, str]] = None,
    ) -> None:
        if self.inp_path is None:
            return
        nodes, pipes = network_ids(self.inp_path)
        selected_pipes = selected_pipes or {}
        selected_pressures = selected_pressures or {}
        default_group = self.bulk_group.get() or (self.group_combo["values"][0] if self.group_combo["values"] else "")
        self._clear_tree(self.pipe_tree)
        for identifier in pipes:
            self.pipe_tree.insert(
                "", "end", values=("✓" if identifier in selected_pipes else "", identifier, selected_pipes.get(identifier, default_group))
            )
        self._clear_tree(self.pressure_tree)
        for identifier in nodes:
            self.pressure_tree.insert(
                "", "end", values=("✓" if identifier in selected_pressures else "", identifier, selected_pressures.get(identifier, "30.0"))
            )

    def _load_catalog(self) -> None:
        self._clear_tree(self.catalog_tree)
        if self.catalog_path is None:
            self.group_combo.configure(values=())
            return
        rows = catalog_rows(self.catalog_path)
        groups = list(dict.fromkeys(row[0] for row in rows))
        self.group_combo.configure(values=groups)
        if groups and self.bulk_group.get() not in groups:
            self.bulk_group.set(groups[0])
        for row in rows:
            self.catalog_tree.insert("", "end", values=row)

    def _edit_tree_cell(self, tree: ttk.Treeview, event: tk.Event) -> None:
        item = tree.identify_row(event.y)
        column_name = tree.identify_column(event.x)
        if not item or not column_name:
            return
        column_index = int(column_name[1:]) - 1
        if column_index == 0:
            current = list(tree.item(item, "values"))
            current[0] = "" if current[0] == "✓" else "✓"
            tree.item(item, values=current)
            self.mark_dirty()
            return
        if column_index == 1:
            return
        x, y, width, height = tree.bbox(item, column_name)
        values = list(tree.item(item, "values"))
        entry = ttk.Entry(tree)
        entry.insert(0, values[column_index])
        entry.select_range(0, "end")
        entry.place(x=x, y=y, width=width, height=height)
        entry.focus_set()

        def finish(_event: Optional[tk.Event] = None) -> None:
            values[column_index] = entry.get().strip()
            tree.item(item, values=values)
            entry.destroy()
            self.mark_dirty()

        entry.bind("<Return>", finish)
        entry.bind("<FocusOut>", finish)
        entry.bind("<Escape>", lambda _event: entry.destroy())

    def _toggle_selected(self, tree: ttk.Treeview) -> None:
        for item in tree.selection():
            values = list(tree.item(item, "values"))
            values[0] = "" if values[0] == "✓" else "✓"
            tree.item(item, values=values)
        if tree.selection():
            self.mark_dirty()

    def _set_use(self, tree: ttk.Treeview, enabled: bool) -> None:
        items = tree.selection() or tree.get_children()
        for item in items:
            values = list(tree.item(item, "values"))
            values[0] = "✓" if enabled else ""
            tree.item(item, values=values)
        if items:
            self.mark_dirty()

    def apply_group(self) -> None:
        group = self.bulk_group.get().strip()
        if not group:
            return
        items = self.pipe_tree.selection() or self.pipe_tree.get_children()
        for item in items:
            values = list(self.pipe_tree.item(item, "values"))
            values[2] = group
            self.pipe_tree.item(item, values=values)
        self.mark_dirty()

    def apply_pressure(self) -> None:
        value = self.bulk_pressure.get().strip()
        try:
            float(value)
        except ValueError:
            messagebox.showerror("Presión no válida", "Introduzca una presión numérica.")
            return
        items = self.pressure_tree.selection() or self.pressure_tree.get_children()
        for item in items:
            values = list(self.pressure_tree.item(item, "values"))
            values[2] = value
            self.pressure_tree.item(item, values=values)
        self.mark_dirty()

    def choose_inp(self) -> None:
        filename = filedialog.askopenfilename(filetypes=(("Modelo EPANET", "*.inp"), ("Todos", "*.*")))
        if not filename:
            return
        path = Path(filename)
        try:
            network_ids(path)
        except Exception as exc:
            messagebox.showerror("INP no válido", str(exc))
            return
        self.inp_path = path
        self.inp_text.set(str(path))
        self._populate_network()
        self.mark_dirty()

    def choose_catalog(self) -> None:
        filename = filedialog.askopenfilename(filetypes=(("Catálogo PPNO", "*.cat"), ("Todos", "*.*")))
        if not filename:
            return
        old_path = self.catalog_path
        self.catalog_path = Path(filename)
        try:
            self._load_catalog()
        except Exception as exc:
            self.catalog_path = old_path
            messagebox.showerror("Catálogo no válido", str(exc))
            return
        self.catalog_text.set(str(self.catalog_path))
        self.mark_dirty()

    def confirm_discard(self) -> bool:
        if not self.dirty:
            return True
        answer = messagebox.askyesnocancel(
            "Cambios sin guardar", "¿Desea guardar los cambios antes de continuar?"
        )
        if answer is None:
            return False
        return self.save() if answer else True

    def new(self) -> None:
        if not self.confirm_discard():
            return
        self.path = None
        self.source = TEMPLATE
        self.inp_path = None
        self.catalog_path = None
        self.inp_text.set("No seleccionado")
        self.catalog_text.set("No seleccionado")
        for variable in self.algorithms.values():
            variable.set(False)
        for tree in (self.pipe_tree, self.pressure_tree, self.catalog_tree):
            self._clear_tree(tree)
        self.group_combo.configure(values=())
        self.dirty = False
        self.validated_fingerprint = None
        self.back_to_edit()
        self.status.set("Nuevo problema PPNO.")
        self._update_title()

    def open(self) -> None:
        if not self.confirm_discard():
            return
        filename = filedialog.askopenfilename(filetypes=(("Problema PPNO", "*.ext"), ("Todos", "*.*")))
        if filename:
            self.load(Path(filename))

    def load(self, path: Path) -> None:
        try:
            payload = project_payload(path)
            self.path = Path(path)
            self.source = str(payload["source"])
            self.inp_path = payload["inp"]  # type: ignore[assignment]
            self.catalog_path = payload["catalog"]  # type: ignore[assignment]
            self.inp_text.set(str(self.inp_path) if self.inp_path else "No seleccionado")
            self.catalog_text.set(str(self.catalog_path) if self.catalog_path else "No seleccionado")
            for name, variable in self.algorithms.items():
                variable.set(name in payload["algorithms"])
            self._load_catalog()
            self._populate_network(dict(payload["pipes"]), dict(payload["pressures"]))
        except Exception as exc:
            messagebox.showerror("No se pudo abrir el proyecto", str(exc))
            return
        self.dirty = False
        self.validated_fingerprint = None
        self.back_to_edit()
        self.status.set(f"Proyecto cargado: {self.path}")
        self._update_title()

    def _collect(self, destination: Path) -> str:
        if self.inp_path is None or not self.inp_path.exists():
            raise ValueError("Seleccione un modelo EPANET .inp existente.")
        if self.catalog_path is None or not self.catalog_path.exists():
            raise ValueError("Seleccione un catálogo .cat existente.")
        pipe_rows = self._tree_values(self.pipe_tree, selected_only=True)
        pressure_rows = self._tree_values(self.pressure_tree, selected_only=True)
        pipes = [(row[1], row[2]) for row in pipe_rows]
        pressures = [(row[1], row[2]) for row in pressure_rows]
        if not pipes:
            raise ValueError("Seleccione al menos una tubería para optimizar.")
        if not pressures:
            raise ValueError("Seleccione al menos un nudo con presión mínima.")
        for _, value in pressures:
            float(value)
        algorithms = [name for name in ALGORITHMS if self.algorithms[name].get()]
        return render_project(
            self.source,
            portable_reference(self.inp_path, destination),
            portable_reference(self.catalog_path, destination),
            algorithms,
            pipes,
            pressures,
        )

    def save(self) -> bool:
        if self.path is None:
            return self.save_as()
        try:
            content = self._collect(self.path)
            save_text(self.path, content)
        except Exception as exc:
            messagebox.showerror("No se pudo guardar", str(exc))
            return False
        self.source = content
        self.dirty = False
        self.validated_fingerprint = None
        self.status.set(f"Guardado: {self.path}")
        self._update_title()
        return True

    def save_as(self) -> bool:
        filename = filedialog.asksaveasfilename(
            defaultextension=".ext", filetypes=(("Problema PPNO", "*.ext"),)
        )
        if not filename:
            return False
        self.path = Path(filename)
        return self.save()

    def _fingerprint(self) -> str:
        if self.path is None or self.catalog_path is None or self.inp_path is None:
            return ""
        digest = hashlib.sha256()
        for path in (self.path, self.catalog_path, self.inp_path):
            digest.update(Path(path).read_bytes())
        return digest.hexdigest()

    def validate(self) -> None:
        if self.running:
            return
        if self.dirty or self.path is None:
            if not self.save():
                return
        optimization: Optional[Optimization] = None
        try:
            optimization = Optimization(self.path)
        except Exception as exc:
            self.validated_fingerprint = None
            messagebox.showerror("Validación fallida", str(exc))
            self.status.set("El problema contiene errores.")
            return
        finally:
            if optimization is not None:
                optimization.close()
        self.validated_fingerprint = self._fingerprint()
        self._show_results_mode()
        self.run_button.configure(state="normal")
        self.status.set("Problema validado. Ya puede iniciar la optimización.")
        messagebox.showinfo("Validación", "El problema PPNO es válido.")

    def _show_results_mode(self) -> None:
        self.edit_toolbar.pack_forget()
        self.result_toolbar.pack(fill="x", before=self.notebook)
        for tab in self.edit_tabs:
            self.notebook.tab(tab, state="hidden")
        self.notebook.tab(self.run_tab, state="normal")
        self.notebook.tab(self.result_tab, state="normal")
        self.notebook.select(self.run_tab)

    def back_to_edit(self, initial: bool = False) -> None:
        if self.running:
            return
        self.result_toolbar.pack_forget()
        if not initial:
            self.edit_toolbar.pack(fill="x", before=self.notebook)
        for tab in getattr(self, "edit_tabs", []):
            self.notebook.tab(tab, state="normal")
        if hasattr(self, "run_tab"):
            self.notebook.tab(self.run_tab, state="hidden")
            self.notebook.tab(self.result_tab, state="hidden")
        if getattr(self, "edit_tabs", []):
            self.notebook.select(self.edit_tabs[0])

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def optimize(self) -> None:
        if self.running or self.path is None:
            return
        if self.validated_fingerprint != self._fingerprint():
            self.validated_fingerprint = None
            self.run_button.configure(state="disabled")
            messagebox.showwarning(
                "Validación caducada", "El proyecto, el INP o el catálogo han cambiado. Valide de nuevo."
            )
            return
        self.running = True
        self.run_button.configure(state="disabled")
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        for tree in (self.summary_tree, self.solution_tree):
            self._clear_tree(tree)
        self.cost_text.set("Optimizando…")
        self.status.set("Optimización en curso. La interfaz seguirá respondiendo.")
        self.worker = threading.Thread(
            target=run_optimization, args=(self.path, self.events), daemon=True
        )
        self.worker.start()
        self.root.after(100, self._poll_events)

    def _poll_events(self) -> None:
        terminal = False
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            if kind == "log":
                self._append_log(str(payload))
            elif kind == "done":
                self._show_payload(payload)  # type: ignore[arg-type]
                terminal = True
            elif kind == "error":
                self.running = False
                self.cost_text.set("Optimización sin resultado")
                self.status.set("La optimización terminó con un error.")
                messagebox.showerror("Error de optimización", str(payload))
                terminal = True
        if not terminal and self.running:
            self.root.after(100, self._poll_events)

    def _show_payload(self, payload: Dict[str, object]) -> None:
        self.running = False
        for result in payload["results"]:  # type: ignore[union-attr]
            row = result
            self.summary_tree.insert(
                "",
                "end",
                values=(
                    row["Algorithm"],
                    row["Attempt"],
                    row["Success"],
                    row["Time (s)"],
                    row["Simulations"],
                    row["Cost"],
                ),
            )
        for row in payload["pipes"]:  # type: ignore[union-attr]
            self.solution_tree.insert("", "end", values=row)
        self.cost_text.set(
            f"Coste total: {float(payload['cost']):.2f}   |   Mejor origen: {payload['best_algorithm']}"
        )
        self.status.set("Optimización finalizada correctamente.")
        self.run_button.configure(state="normal")
        self.notebook.select(self.result_tab)

    def close(self) -> None:
        if self.running and not messagebox.askyesno(
            "Optimización en curso",
            "La optimización continúa en segundo plano. ¿Desea cerrar la interfaz?",
        ):
            return
        if not self.running and not self.confirm_discard():
            return
        self.root.destroy()


def main() -> None:
    """Start the native PPNO GUI in the active Python/Conda environment."""
    root = tk.Tk()
    ProjectEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
