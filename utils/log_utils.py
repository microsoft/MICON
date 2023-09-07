import os
import sys
import tqdm
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from matplotlib.lines import Line2D

class Averager:
    def __init__(self, weight: float = 1):
        self.weight = weight
        self.reset()

    def reset(self):
        self.total = defaultdict(float)
        self.counter = defaultdict(float)

    def update(self, stats):
        for key, value in stats.items():
            self.total[key] = self.total[key] * self.weight + value * self.weight
            self.counter[key] = self.counter[key] * self.weight + self.weight

    def average(self):
        averaged_stats = {
            key: tot / self.counter[key] for key, tot in self.total.items()
        }
        
        return averaged_stats


class Logger:
    def __init__(self):
        self.logger = logging.getLogger("Main")
    
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            stream=DummyTqdmFile(sys.stderr)
        )
        
        self.logger.info(f'Working directory is {os.getcwd()}')
        
    def log_stats(self, stats, step, prefix=''):
        # if self.neptune_logger is not None:
        #     for k, v in stats.items():
        #         self.neptune_logger[f'{prefix}{k}'].log(v, step=step)

        msg_start = f'[{prefix[:-1]}] Step {step}' + ' | '
        dict_msg = ' | '.join([f'{k.capitalize()} --> {v:.3f}' for k, v in stats.items()]) + ' | '

        msg = msg_start + dict_msg

        self.log_message(msg)

    def log_message(self, msg):
        self.logger.info(msg)
        
class DummyTqdmFile(object):
    """ Dummy file-like that will write to tqdm
    https://github.com/tqdm/tqdm/issues/313
    """
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.tqdm.write(x, file=self.file, end='')

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def plot_grad_flow_v2(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(".".join(n.split(".")[:-1]))
            ave_grads.append(p.grad.cpu().abs().mean())
            max_grads.append(p.grad.cpu().abs().max())
    plt.figure(figsize=(36,24))
    plt.tight_layout()
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.8, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical", fontsize=14)
    plt.yticks(np.arange(0, 0.002, 0.0005), fontsize=20)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = 0, top=0.003) # zoom in on the lower gradient regions
    plt.xlabel("Layers", fontsize=14)
    plt.ylabel("Gradient", fontsize=26)
    plt.title("Gradient flow", fontsize=14)
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4)], ['max-gradient', 'mean-gradient'])
    plt.savefig("gradient_check.png", bbox_inches='tight')