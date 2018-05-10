import math
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils import clip_grad_norm

class Optim(object):

    def set_parameters(self, params):
        self.params = list(params)  # careful: params may be a generator
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            self.optimizer = optim.Adam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, method, lr, max_grad_norm, lr_decay=1, start_decay_at=None, max_decay_times=2):
        self.last_score = None
        self.decay_times = 0
        self.max_decay_times = max_decay_times
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False

    def step(self):
        # Compute gradients norm.
        if self.max_grad_norm:
            clip_grad_norm(self.params, self.max_grad_norm)
        self.optimizer.step()

    # decay learning rate if val perf does not improve or we hit the start_decay_at limit
    def updateLearningRate(self, score, epoch):
        if self.start_decay_at is not None and epoch >= self.start_decay_at:
            self.start_decay = True
        #decay_flag = False
        #if self.last_score is not None and score < self.last_score:
        #    self.decay_times += 1
        #    if self.decay_times >= self.max_decay_times:
        #        decay_flag = True
        #else:
        #    self.decay_times = 0

        if self.start_decay:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)

        self.last_score = score
        self.optimizer.param_groups[0]['lr'] = self.lr
