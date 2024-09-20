import torch

import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__)) 
os.chdir(DIR+"/../")
sys.path.append(DIR+"/../")

from begin.algorithms.bare.links import *
from begin.scenarios.links import LCScenarioLoader
from begin.utils import GCNLink




results = benchmark.run(epoch_per_task = 20)