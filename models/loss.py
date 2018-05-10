'''
 @Author: Shuming Ma
 @mail:   shumingma@pku.edu.cn
 @homepage : shumingma.com
'''
import torch
import torch.nn as nn
import models
import data.dict as dict
from torch.autograd import Variable
from torch.nn.parameter import Parameter

def label_criterion(use_cuda):
    crit = nn.CrossEntropyLoss(size_average=True)
    if use_cuda:
        crit.cuda()
    return crit

def criterion(tgt_vocab_size, use_cuda):
    weight = torch.ones(tgt_vocab_size)
    weight[dict.PAD] = 0
    crit = nn.CrossEntropyLoss(weight, size_average=False)
    if use_cuda:
        crit.cuda()
    return crit

def nll_criterion(tgt_vocab_size, use_cuda):
    weight = torch.ones(tgt_vocab_size)
    weight[dict.PAD] = 0
    crit = nn.NLLLoss(weight, size_average=False)
    if use_cuda:
        crit.cuda()
    return crit

class copy_criterion(nn.Module):
    def __init__(self, use_cuda):
        super(copy_criterion, self).__init__()
        self.crit = nn.NLLLoss(size_average=False)
        self.use_cuda = use_cuda

    def forward(self, outputs, targets):
        batch_size, vocab_size = outputs.size()
        weight = torch.ones(vocab_size)
        weight[dict.PAD] = 0
        weight = Variable(weight.unsqueeze(0))
        if self.use_cuda:
            weight = weight.cuda()
        return self.crit(outputs*weight, targets)


class label_embedding(nn.Module):
    def __init__(self, hidden_size, vocab_size, use_cuda):
        super(label_embedding, self).__init__()
        self.use_cuda = use_cuda
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.embedding = torch.nn.Embedding(vocab_size, vocab_size)
        self.embedding.weight = Parameter(torch.eye(vocab_size))

    def forward(self, outputs, y):
        return self.linear(outputs.detach()), self.embedding(y)

class label_embedding_compress(nn.Module):
    def __init__(self, hidden_size, vocab_size, use_cuda):
        super(label_embedding_compress, self).__init__()
        self.use_cuda = use_cuda
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.emb_linear = nn.Linear(hidden_size, vocab_size)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, outputs, y):
        return self.linear(outputs.detach()), self.relu(self.emb_linear(self.embedding(y)))

class criterion_emb(nn.Module):
    def __init__(self, hidden_size, tgt_vocab_size, use_cuda):
        super(criterion_emb, self).__init__()
        self.use_cuda = use_cuda
        self.weight = torch.ones(tgt_vocab_size)
        self.xent_logits_target = nn.CrossEntropyLoss(
            self.weight, size_average=False)
        if use_cuda:
            self.xent_logits_target.cuda()
        self.tau_inverse = 0.5
        self.max_prob = 0.9
        self.mse_strength = 1.0
        self.pretrain = 20000
        self.emb = label_embedding_compress(hidden_size, tgt_vocab_size, use_cuda)

    def cross_entropy_with_logits(self, logits, target_prob):
        prob = torch.nn.functional.log_softmax(logits)
        loss = -torch.sum(prob * target_prob)
        return loss

    def mse_target_with_max_prob(self, prob, target):
        max_prob = torch.gather(prob, -1, target.view(-1, 1))
        loss = torch.sum(torch.abs(max_prob - self.max_prob))
        return loss

    def forward(self, outputs, logits, target, updates):

        y2, emb = self.emb(Variable(outputs.data), target)
        emb_prob = torch.nn.functional.softmax(emb)

        # y1 -> one_hot
        xent_logits_target = self.xent_logits_target(logits, target)
        loss = xent_logits_target

        # y2 -> one_hot
        xent_out_target = self.xent_logits_target(y2, target)
        eloss = xent_out_target

        # y2 -> p2
        out_prob = torch.nn.functional.softmax(y2)
        out_low_prob = torch.nn.functional.softmax(y2 * self.tau_inverse).detach()
        mask = torch.eq(out_low_prob.max(1)[1], target).float().detach()

        # emb -> p2
        xent_emb_out_low = self.cross_entropy_with_logits(emb, out_low_prob * mask.view(-1, 1))
        eloss += xent_emb_out_low

        # y2[gold] -> 0.9
        mse_emb = self.mse_target_with_max_prob(out_prob, target)
        eloss += self.mse_strength * mse_emb

        if updates < self.pretrain:
            return loss + eloss

        # y1 -> emb
        xent_logits_emb = self.cross_entropy_with_logits(
            logits, emb_prob.detach())
        eloss += xent_logits_emb

        return loss + eloss


def memory_efficiency_cross_entropy_loss(hidden_outputs, decoder, targets, criterion, config):
    outputs = Variable(hidden_outputs.data, requires_grad=True, volatile=False)
    num_total, num_correct, loss = 0, 0, 0

    outputs_split = torch.split(outputs, config.max_generator_batches)
    targets_split = torch.split(targets, config.max_generator_batches)
    for i, (out_t, targ_t) in enumerate(zip(outputs_split, targets_split)):
        out_t = out_t.view(-1, out_t.size(2))
        scores_t = decoder.compute_score(out_t)
        loss_t = criterion(scores_t, targ_t.view(-1))
        pred_t = scores_t.max(1)[1]
        num_correct_t = pred_t.data.eq(targ_t.data).masked_select(targ_t.ne(dict.PAD).data).sum()
        num_total_t = targ_t.ne(dict.PAD).data.sum()
        num_correct += num_correct_t
        num_total += num_total_t
        loss += loss_t.data[0]
        loss_t.div(num_total_t).backward()

    grad_output = outputs.grad.data
    hidden_outputs.backward(grad_output)

    return loss, num_total, num_correct, config.tgt_vocab, config.tgt_vocab

def copy_cross_entropy_loss(hidden_outputs, decoder, targets, criterion, config, sim_score=0):
    loss, pred = [], []
    for id, output in enumerate(hidden_outputs):
        l = criterion(output, targets[id])

        if (l.data != l.data).any():
            print(output)
            print(targets[id])
            raise(ValueError)

        p = output.max(1)[1]
        loss.append(l)
        pred.append(p)
    loss = torch.sum(torch.stack(loss))
    pred = torch.stack(pred)

    num_correct = pred.data.eq(targets.data).masked_select(targets.ne(dict.PAD).data).sum()
    num_total = targets.ne(dict.PAD).data.sum()
    loss = loss / num_total

    return loss, num_total, num_correct

def cross_entropy_loss(hidden_outputs, decoder, targets, criterion, config, sim_score=0, compute_score=True):
    outputs = hidden_outputs.view(-1, hidden_outputs.size(2))
    if compute_score:
        scores = decoder.compute_score(outputs)
    else:
        scores = outputs
    loss = criterion(scores, targets.view(-1)) + sim_score
    pred = scores.max(1)[1]
    num_correct = pred.data.eq(targets.data).masked_select(targets.ne(dict.PAD).data).sum()
    num_total = targets.ne(dict.PAD).data.sum()
    #loss.div(num_total).backward()
    #loss = loss.data[0]
    loss = loss / num_total

    return loss, num_total, num_correct


def prior_knowledge_loss(hidden_outputs, decoder, targets, criterion, config, updates):
    outputs = hidden_outputs.view(-1, hidden_outputs.size(2))
    scores = decoder.compute_score(outputs)
    loss = criterion(outputs, scores, targets.view(-1), updates)
    pred = scores.max(1)[1]
    num_correct = pred.data.eq(targets.data).masked_select(targets.ne(dict.PAD).data).sum()
    num_total = targets.ne(dict.PAD).data.sum()
    #loss.div(num_total).backward()
    #loss = loss.data[0]
    loss = loss / num_total

    return loss, num_total, num_correct