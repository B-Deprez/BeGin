import torch

import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__)) 
os.chdir(DIR+"/../")
sys.path.append(DIR+"/../")

from begin.algorithms.bare.nodes import *
from begin.scenarios.nodes import NCScenarioLoader
from begin.utils import GCNNode

scenario = NCScenarioLoader(dataset_name='elliptic', num_tasks=49, metric='rocauc', save_path='data', incr_type='time', task_shuffle=1)
model = GCNNode(scenario.num_feats, scenario.num_classes, 256, dropout=0.0)
benchmark = NCTimeILBareTrainer(model = model,
                                scenario = scenario,
                                optimizer_fn = lambda x: torch.optim.Adam(x, lr=1e-3, weight_decay=0),
                                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1),
                                device = torch.device('cpu'),
                                scheduler_fn = lambda x: torch.optim.lr_scheduler.ReduceLROnPlateau(x, mode='min', patience=20, min_lr=1e-3 * 0.001 * 2., verbose=False),
                                benchmark = True, seed = 42, full_mode = True)
results = benchmark.run(epoch_per_task = 20)