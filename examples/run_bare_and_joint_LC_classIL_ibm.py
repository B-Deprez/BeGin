import torch

import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__)) 
os.chdir(DIR+"/../")
sys.path.append(DIR+"/../")

from begin.algorithms.bare.links import *
from begin.scenarios.links import LCScenarioLoader
from begin.utils import GCNLink


scenario = LCScenarioLoader(
    dataset_name="ibm", 
    num_tasks=9,
    metric="f1micro",
    incr_type="class", 
    task_shuffle=False, 
    save_path="data"
)

model = GCNLink(
    scenario.num_feats, 
    scenario.num_classes, 
    256, 
    dropout=0.5
)

benchmark = LCClassILBareTrainer(
    model = model,
    scenario = scenario,
    optimizer_fn=lambda x: torch.optim.Adam(x, lr=1e-3, weight_decay=0),
    loss_fn=torch.nn.CrossEntropyLoss(ignore_index=-1),
    device=torch.device('cpu'),
    scheduler_fn=lambda x: torch.optim.lr_scheduler.ReduceLROnPlateau(x, mode='min', patience=20, min_lr=1e-3 * 0.001 * 2., verbose=False),
    benchmark=True,
    seed=1997, 
    full_mode=True
)

results = benchmark.run(epoch_per_task = 11)