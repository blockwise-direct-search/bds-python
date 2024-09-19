"""
PyCUTEst example: minimize 2D Rosenbrock function using BDS method.

Jaroslav Fowkes and Lindon Roberts, 2022.
"""

import pdb
import pycutest
from bds_python import bds as bds

p = pycutest.import_problem('ROSENBR')

print("Rosenbrock function in %gD" % p.n)
init_value = p.obj(p.x0)
init_point = p.x0
print(init_value)
pdb.set_trace()

bds(p.obj, p.x0)
