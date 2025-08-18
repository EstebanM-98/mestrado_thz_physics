# euzn2p2_p3m1_viz.py
# Requisitos: pymatgen, vtk  (pip install pymatgen vtk)

from pymatgen.core import Lattice, Structure
from pymatgen.vis.structure_vtk import StructureVis
from pymatgen.io.ase import AseAtomsAdaptor
import nglview as nv



# --- Parámetros experimentales (P-3m1, No. 164) ---
a, c = 4.08497, 7.00190  # Å (T ~ 213 K)
latt = Lattice.hexagonal(a, c)

# Wyckoff:
# Eu: 1a (0, 0, 0)
# Zn: 2d (2/3, 1/3, z_Zn)
#  P: 2d (1/3, 2/3, z_P)
z_Zn = 0.36947
z_P  = 0.26915

species = ["Eu", "Zn", "Zn", "P", "P"]
frac_coords = [
    [0.0,   0.0,   0.0     ],   # Eu 1a
    [2/3,   1/3,   z_Zn    ],   # Zn 2d
    [1/3,   2/3,   1 - z_Zn],   # mate por simetría
    [1/3,   2/3,   z_P     ],   # P  2d
    [2/3,   1/3,   1 - z_P ]    # mate por simetría
]

struct = Structure(latt, species, frac_coords)
atoms = AseAtomsAdaptor.get_atoms(struct)
nv.show_ase(atoms)







