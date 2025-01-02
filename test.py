import numpy as np
from itertools import product
from AuxiliarFunctions import *


N = 16
EDCAaccessCategory = 'BE'

tau, EB, p = SimpleDCF_modelWithBEB(N, EDCAaccessCategory)
print(tau, EB, p)
