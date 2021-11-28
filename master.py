# Copyright (c) 2020 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from argparse import ArgumentParser
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import torch
import random
import numpy as np
from poet_distributed.es import initialize_master_fiber
from poet_distributed.poet_algo import MultiESOptimizer
from poet_distributed.poet_ppo_algo import MutliPPOOptimizer


def seed_everything(seed=0, harsh=False):
    """
    Seeds all important random functions
    Args:
        seed (int, optional): seed value. Defaults to 0.
        harsh (bool, optional): torch backend deterministic. Defaults to False.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    if harsh:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def run_main(args):

    initialize_master_fiber()

    #set master_seed
    np.random.seed(args.master_seed)

    # optimizer_zoo = MultiESOptimizer(args=args)
    optimizer_zoo = MutliPPOOptimizer(args=args)

    optimizer_zoo.optimize(iterations=args.n_iterations,
                       propose_with_adam=args.propose_with_adam,
                       reset_optimizer=True,
                       checkpointing=args.checkpointing,
                       steps_before_transfer=args.steps_before_transfer)

def main():
    parser = ArgumentParser()
    parser.add_argument('log_file')
    parser.add_argument('--model_dir', required=True, type=str, help='path to model directory')
    parser.add_argument('--init', default='random')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--lr_decay', type=float, default=0.9999)
    parser.add_argument('--lr_limit', type=float, default=0.001)
    parser.add_argument('--noise_std', type=float, default=0.1)
    parser.add_argument('--noise_decay', type=float, default=0.999)
    parser.add_argument('--noise_limit', type=float, default=0.01)
    parser.add_argument('--l2_coeff', type=float, default=0.01)
    parser.add_argument('--batches_per_chunk', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--eval_batch_size', type=int, default=1)
    parser.add_argument('--eval_batches_per_step', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=64)
    parser.add_argument('--n_iterations', type=int, default=200)
    parser.add_argument('--steps_before_transfer', type=int, default=25)
    parser.add_argument('--master_seed', type=int, default=111)
    parser.add_argument('--mc_lower', type=int, default=25)
    parser.add_argument('--mc_upper', type=int, default=300)
    parser.add_argument('--repro_threshold', type=int, default=150)
    parser.add_argument('--max_num_envs', type=int, default=100)
    parser.add_argument('--normalize_grads_by_noise_std', action='store_true', default=False)
    parser.add_argument('--propose_with_adam', action='store_true', default=False)
    parser.add_argument('--checkpointing', action='store_true', default=False)
    parser.add_argument('--adjust_interval', type=int, default=1)
    parser.add_argument('--returns_normalization', default='normal')
    parser.add_argument('--stochastic', action='store_true', default=False)
    parser.add_argument('--envs', nargs='+')
    parser.add_argument('--start_from', default=None)  # Json file to start from
    parser.add_argument('--max_num_morphs', type=int, default=10)
    parser.add_argument('--morph_evolve_interval', type=int, default=4)
    parser.add_argument('--init_num_morphs', type=int, default=2)
    parser.add_argument('--decay_lr', action="store_true", help="decay PPO lr linearly?")
    parser.add_argument('--lr_end_factor', type=float, default=0.1)
    parser.add_argument('--lr_decay_iters', type=int, default=200)

    args = parser.parse_args()
    logger.info(args)
    # seed_everything(seed=args.master_seed)
    run_main(args)

if __name__ == "__main__":
    main()
