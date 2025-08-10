from __future__ import annotations
import re
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Dict, List, Tuple, Optional, Callable
from collections import defaultdict

import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# Configuración y utilidades
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

TEMP_RE = re.compile(r"(\d+(?:\.\d+)?)\s*K", re.IGNORECASE)

def extraer_temperatura(x: Path | str) -> Optional[float]:
    """Extrae la temperatura (en K) desde un nombre/ruta usando ...K."""
    name = Path(x).name
    m = TEMP_RE.search(name)
    return float(m.group(1)) if m else None

def build_ranges(min_t: float, max_t: float, *, step: Optional[float] = None, bins: Optional[int] = None
                ) -> List[Tuple[float, float]]:
    """
    Construye rangos cerrados por izquierda y abiertos por derecha: [a, b).
    - Si step está definido: usa paso fijo.
    - Si bins está definido: divide en número de bins iguales.
    - Si ninguno: devuelve lista vacía ⇒ se agrupa por temperatura exacta.
    """
    if step is None and bins is None:
        return []

    if step is not None:
        edges = np.arange(min_t, max_t + step, step, dtype=float)
    else:
        edges = np.linspace(min_t, max_t, bins + 1)

    ranges = [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]
    # Ajuste para incluir el extremo superior en el último rango
    if ranges:
        a, b = ranges[-1]
        ranges[-1] = (a, max_t + 1e-12)
    return ranges

def asignar_rango(t: float, ranges: List[Tuple[float, float]]) -> Tuple[float, float] | float:
    """Devuelve el rango [a,b) que contiene t, o t si ranges está vacío (agrupación exacta)."""
    if not ranges:
        return t
    for a, b in ranges:
        if a <= t < b:
            return (a, b)
    # Por si cae exactamente en el borde superior por numérica
    a, b = ranges[-1]
    if np.isclose(t, b):
        return (a, b)
    raise ValueError(f"Temperatura {t} fuera de los rangos definidos.")

# -----------------------------------------------------------------------------
# Parámetros del proceso
# -----------------------------------------------------------------------------
@dataclass
class ColumnsCfg:
    x: str = "pos"
    y: str = "X"
    sep: str | None = None          # None + delim_whitespace=True
    delim_whitespace: bool = True

@dataclass
class AveragingCfg:
    scale_x: float = 1.0            # p.ej. 2/c
    apply_scale_after_mean: bool = True  # True equivale a (mean(pos))*scale_x

@dataclass
class GroupingCfg:
    step: Optional[float] = None     # tamaño de paso de temperatura (e.g. 10.0)
    bins: Optional[int] = None       # número de bins
    round_output_temp: int = 2       # decimales para nombre de salida

@dataclass
class ProcessCfg:
    input_dir: Path
    output_subdir: str = "carpeta1"
    pattern: str = "*.dat"
    columns: ColumnsCfg = ColumnsCfg()
    avg: AveragingCfg = AveragingCfg()
    group: GroupingCfg = GroupingCfg()
    output_prefix: str = "Average"
    output_ext: str = ".dat"         # espacio separado
    keep_empty: bool = False         # si no hay temperaturas, seguir o no

# -----------------------------------------------------------------------------
# Núcleo de procesamiento
# -----------------------------------------------------------------------------
def clean_folder(folder: Path, pattern: str = "*.dat") -> int:
    """Elimina archivos que coinciden con `pattern` en `folder`."""
    n = 0
    for p in folder.glob(pattern):
        try:
            p.unlink()
            n += 1
        except Exception as e:
            logging.warning(f"No se pudo eliminar {p}: {e}")
    if n:
        logging.info(f"Eliminados {n} archivos en {folder}")
    return n

def list_dat_files(folder: Path, pattern: str) -> List[Path]:
    return sorted(folder.glob(pattern), key=lambda p: p.name)

def read_dat(path: Path, cfg: ColumnsCfg) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=cfg.sep,
        delim_whitespace=cfg.delim_whitespace,
        engine="python"
    )

def convert_dats(cfg: ProcessCfg) -> List[Path]:
    """
    Procesa todos los .dat en cfg.input_dir y guarda promedios en output_subdir.
    Agrupa por temperatura (exacta o en rangos).
    Devuelve la lista de rutas de salida.
    """
    in_dir = cfg.input_dir
    out_dir = in_dir / cfg.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Limpia salidas previas
    clean_folder(out_dir, cfg.pattern)

    archivos = list_dat_files(in_dir, cfg.pattern)
    if not archivos:
        logging.info(f"No se encontraron archivos {cfg.pattern} en {in_dir}.")
        return []

    # Agrupar por temperatura base
    por_temp: Dict[float, List[Path]] = defaultdict(list)
    temps: List[float] = []
    for f in archivos:
        t = extraer_temperatura(f)
        if t is not None:
            por_temp[t].append(f)
            temps.append(t)

    if not temps:
        msg = "No se encontraron temperaturas en los archivos."
        if cfg.keep_empty:
            logging.warning(msg + " Continuando sin procesar.")
            return []
        else:
            logging.error(msg)
            return []

    # Ordenar por temperatura
    temps_sorted = sorted(set(temps))
    por_temp = {t: por_temp[t] for t in temps_sorted}

    # Construir rangos (si aplica)
    ranges = build_ranges(min(temps_sorted), max(temps_sorted),
                          step=cfg.group.step, bins=cfg.group.bins)

    # Reagrupar por rango o dejar por temperatura exacta
    grupos: Dict[float | Tuple[float, float], List[Path]] = defaultdict(list)
    for t, files in por_temp.items():
        key = asignar_rango(t, ranges)
        grupos[key].extend(files)

    salidas: List[Path] = []

    # Procesar cada grupo
    for key, files in grupos.items():
        try:
            n = len(files)
            if n == 0:
                continue

            # Acumuladores (usaremos suma para luego promediar)
            sum_x = None
            sum_y = None
            temps_group = []

            for f in files:
                df = read_dat(f, cfg.columns)
                # Validación mínima de columnas
                if cfg.columns.x not in df.columns or cfg.columns.y not in df.columns:
                    raise ValueError(
                        f"Archivo {f} no contiene columnas '{cfg.columns.x}' y '{cfg.columns.y}'."
                    )
                if sum_x is None:
                    sum_x = df[cfg.columns.x].astype(float).copy()
                    sum_y = df[cfg.columns.y].astype(float).copy()
                else:
                    sum_x = sum_x.add(df[cfg.columns.x].astype(float), fill_value=0.0)
                    sum_y = sum_y.add(df[cfg.columns.y].astype(float), fill_value=0.0)

                t = extraer_temperatura(f)
                if t is not None:
                    temps_group.append(t)

            # Promedio
            mean_x = sum_x / n
            mean_y = sum_y / n

            # Escala (tu 2/c) — por defecto después del promedio
            if cfg.avg.apply_scale_after_mean:
                mean_x = mean_x * cfg.avg.scale_x
            else:
                mean_x = (sum_x * cfg.avg.scale_x) / n

            # Temperatura “representativa” del grupo para el nombre
            if isinstance(key, tuple):
                a, b = key
                mean_temp = np.mean(temps_group) if temps_group else (a + b) / 2
            else:
                mean_temp = float(key)

            out_name = f"{cfg.output_prefix}_{round(mean_temp, cfg.group.round_output_temp)}K{cfg.output_ext}"
            out_path = out_dir / out_name
            out_df = pd.DataFrame({cfg.columns.x: mean_x, cfg.columns.y: mean_y})
            # Guardado con separador de espacio
            out_df.to_csv(out_path, index=False, sep=" ")
            salidas.append(out_path)
            logging.info(f"Generado: {out_path}")

        except Exception as e:
            # Usamos round seguro si podemos
            try:
                tshow = round(mean_temp, cfg.group.round_output_temp)  # type: ignore
            except Exception:
                tshow = str(key)
            logging.error(f"Error al procesar el grupo {tshow}: {e}")

    # Orden final por temperatura extraída desde el nombre
    salidas = sorted(salidas, key=lambda p: (extraer_temperatura(p.name) or float("inf")))
    return salidas

# -----------------------------------------------------------------------------
# Orquestador de sample/reference
# -----------------------------------------------------------------------------
@dataclass
class PairProcessCfg:
    base_dir: Path                      # p.ej. Path.cwd()
    project_rel: Path = Path("EuZn2P2/src")
    sample_id: int = 6
    sample_fmt: str = "sample{n}_ang"
    ref_fmt: str = "reference{n}"
    output_subdir: str = "carpeta1"
    pattern: str = "*.dat"
    columns: ColumnsCfg = ColumnsCfg()
    scale_x: float = 1.0                # tu 2/c
    group_step: Optional[float] = None  # e.g. 10.0 para [min, max] con paso 10
    group_bins: Optional[int] = None    # o por número de bins
    round_output_temp: int = 2

def process_sample_and_ref(cfg: PairProcessCfg) -> Tuple[List[Path], List[Path]]:
    """
    Procesa sample y reference, limpiando salidas previas y generando promedios.
    Devuelve (salidas_reference, salidas_sample) — ambas listas ordenadas por temperatura.
    """
    sample_dir = cfg.base_dir / cfg.project_rel / cfg.sample_fmt.format(n=cfg.sample_id)
    ref_dir    = cfg.base_dir / cfg.project_rel / cfg.ref_fmt.format(n=cfg.sample_id)

    # Config común para ambos
    proc_common = dict(
        output_subdir=cfg.output_subdir,
        pattern=cfg.pattern,
        columns=cfg.columns,
        avg=AveragingCfg(scale_x=cfg.scale_x, apply_scale_after_mean=True),
        group=GroupingCfg(step=cfg.group_step, bins=cfg.group_bins, round_output_temp=cfg.round_output_temp),
    )

    # Primero, limpia (si quieres también los .dat fuentes, descomenta abajo)
    # clean_folder(sample_dir, cfg.pattern)
    # clean_folder(ref_dir, cfg.pattern)

    # Limpia outputs previos
    clean_folder(sample_dir / cfg.output_subdir, cfg.pattern)
    clean_folder(ref_dir / cfg.output_subdir, cfg.pattern)

    # Procesa referencia
    ref_out = convert_dats(ProcessCfg(input_dir=ref_dir, **proc_common))
    # Procesa muestra
    samp_out = convert_dats(ProcessCfg(input_dir=sample_dir, **proc_common))

    return (ref_out, samp_out)

# -----------------------------------------------------------------------------
# Ejemplo de uso (ajústalo a tu proyecto)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # Constante de ejemplo:
    c = 3e8
    ruta_actual = Path.cwd()

    pair_cfg = PairProcessCfg(
        base_dir=ruta_actual,
        project_rel=Path("EuZn2P2/src"),
        sample_id=6,
        sample_fmt="sample{n}_ang",
        ref_fmt="reference{n}",
        output_subdir="carpeta1",
        pattern="*.dat",
        columns=ColumnsCfg(x="pos", y="X", sep=None, delim_whitespace=True),
        scale_x=2.0/c,              # <- Tu factor 2/c
        group_step=None,            # Usa None para agrupar por temperatura exacta
        group_bins=4,               # O usa bins=4 como tu 'rang' anterior
        round_output_temp=2
    )

    ref_salidas, samp_salidas = process_sample_and_ref(pair_cfg)
    print("REF:", [p.name for p in ref_salidas])
    print("SAMP:", [p.name for p in samp_salidas])
