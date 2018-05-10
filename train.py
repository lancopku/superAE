'''
 @Author: Shuming Ma
 @mail:   shumingma@pku.edu.cn
 @homepage : shumingma.com
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.utils.serialization import load_lua
import models
import data.dataloader as dataloader
import data.utils as utils
import data.dict as dict
from optims import Optim
#from predict import eval

import os
import argparse
import time
import math
import collections
import codecs

#config
parser = argparse.ArgumentParser(description='train.py')
parser.add_argument('-config', default='default.yaml', type=str,
                    help="config file")
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")
parser.add_argument('-restore', default='', type=str,
                    help="restore checkpoint")
parser.add_argument('-seed', type=int, default=1234,
                    help="Random seed")
parser.add_argument('-model', default='seq2seq', type=str,
                    help="Model selection")
parser.add_argument('-score', default='', type=str,
                    help="score_fn")
parser.add_argument('-pretrain', action='store_true',
                    help="load pretrain embedding")
parser.add_argument('-single_pass', action='store_true',
                    help="train or not")
parser.add_argument('-limit', type=int, default=0,
                    help="data limit")
parser.add_argument('-log', default='', type=str,
                    help="log directory")
parser.add_argument('-unk', action='store_true',
                    help="replace unk")
parser.add_argument('-reduce', action='store_true',
                    help="reduce redundancy")
parser.add_argument('-group', action='store_true',
                    help="group evaluation")
parser.add_argument('-loss', default='', type=str,
                    help="loss function")
parser.add_argument('-weight', type=float, default=0.0,
                    help="weight")
parser.add_argument('-update', type=int, default=0,
                    help="pretrain updates")

opt = parser.parse_args()
config = utils.read_config(opt.config)
torch.manual_seed(opt.seed)

#checkpoint
if opt.restore:
    print('loading checkpoint...\n')
    checkpoints = torch.load(opt.restore)
    config = checkpoints['config']

#cuda
use_cuda = torch.cuda.is_available() and len(opt.gpus)>0
if use_cuda:
    torch.cuda.set_device(opt.gpus[0])
    torch.cuda.manual_seed(opt.seed)
    #print(opt.gpus)
    #os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(opt.gpus)
    #cudnn.benchmark = True

#data
print('loading data...\n')
start_time = time.time()
datas = torch.load(config.data)
print('loading time cost: %.3f' % (time.time()-start_time))

trainset, validset = datas['train'], datas['valid']
src_vocab, tgt_vocab = datas['dicts']['src'], datas['dicts']['tgt']

if not (hasattr(config, 'src_vocab') or hasattr(config, 'tgt_vocab')):
    config.src_vocab = src_vocab.size()
    config.tgt_vocab = tgt_vocab.size()

if opt.limit > 0:
    trainset.src = trainset.src[:opt.limit]
    validset = trainset

if hasattr(config, 'eval_batch_size'):
    eval_batch_size = config.eval_batch_size
else:
    eval_batch_size = config.batch_size

if 'copy' in opt.score:
    padding = dataloader.pg_padding
else:
    padding = dataloader.padding
trainloader = dataloader.get_loader(trainset, batch_size=config.batch_size, shuffle=True, num_workers=0, padding=padding)
validloader = dataloader.get_loader(validset, batch_size=eval_batch_size, shuffle=False, num_workers=0, padding=padding)

if opt.pretrain:
    pretrain_embed = torch.load(config.emb_file)
else:
    pretrain_embed = None
#model
print('building model...\n')
model = getattr(models, opt.model)(config, config.src_vocab, config.tgt_vocab, use_cuda,
                       w2v=pretrain_embed, score_fn=opt.score, weight=opt.weight, pretrain_updates=opt.update,
                                   extend_vocab_size=tgt_vocab.size()-config.tgt_vocab, device_ids=opt.gpus)


if opt.restore:
    model.load_state_dict(checkpoints['model'])
if use_cuda:
    model.cuda()
if len(opt.gpus) > 1:
    model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
    model_module = model.module
else:
    model_module = model

#optimizer
if opt.restore:
    optim = checkpoints['optim']
else:
    optim = Optim(config.optim, config.learning_rate, config.max_grad_norm,
                  lr_decay=config.learning_rate_decay,start_decay_at=config.start_decay_at)
optim.set_parameters(model.parameters())

param_count = 0
for param in model.parameters():
    param_count += param.view(-1).size()[0]

#log
if not os.path.exists(config.log):
    os.mkdir(config.log)
if opt.log == '':
    log_path = config.log + str(int(time.time() * 1000)) + '/'
else:
    log_path = config.log + opt.log + '/'
if not os.path.exists(log_path):
    os.mkdir(log_path)
logging = utils.logging(log_path+'log.txt')
for k, v in config.items():
    logging("%s:\t%s\n" % (str(k), str(v)))
logging("\n")
logging(repr(model)+"\n\n")

logging('total number of parameters: %d\n\n' % param_count)
logging('score function is %s\n\n' % opt.score)

#checkpoint
if opt.restore:
    updates = checkpoints['updates']
else:
    updates = 0
total_loss, start_time = 0, time.time()
report_total, report_correct = 0, 0
scores = [[] for metric in config.metric]
scores = collections.OrderedDict(zip(config.metric, scores))

#train
def train(epoch):
    model.train()

    if opt.model == 'gated':
        model.current_epoch = epoch

    global updates, total_loss, start_time, report_correct, report_total, report_tot_vocab, report_vocab

    for batch in trainloader:

        model.zero_grad()

        src, src_len, tgt, tgt_len = batch['src'], batch['src_len'], batch['tgt'], batch['tgt_len']

        if 'num_oovs' in batch.keys():
            num_oovs = batch['num_oovs']
            #print(num_oovs)
        else:
            num_oovs = 0

        loss, num_total, num_correct = model.train_model(src, src_len, tgt, tgt_len, opt.loss, updates, optim, num_oovs=num_oovs)

        total_loss += loss
        report_correct += num_correct
        report_total += num_total

        #optim.step()
        utils.progress_bar(updates, config.eval_interval)
        updates += 1

        if updates % config.eval_interval == 0:
            logging("epoch: %3d, ppl: %6.3f, time: %6.3f, updates: %8d, accuracy: %2.2f\n"
                    % (epoch, math.exp(total_loss / report_total), time.time()-start_time, updates,
                       report_correct * 100.0 / report_total))
            print('evaluating after %d updates...\r' % updates)
            score = eval(epoch)
            for metric in config.metric:
                scores[metric].append(score[metric])
                if score[metric] >= max(scores[metric]):
                    save_model(log_path+'best_'+metric+'_checkpoint.pt')
                    if metric == 'bleu':
                        with codecs.open(log_path+'best_'+metric+'_prediction.txt','w','utf-8') as f:
                            f.write(codecs.open(log_path+'candidate.txt','r','utf-8').read())
            model.train()
            total_loss, start_time = 0, time.time()
            report_correct, report_total = 0, 0
            report_vocab, report_tot_vocab = 0, 0

        if updates % config.save_interval == 0:
            save_model(log_path+'checkpoint.pt')

    optim.updateLearningRate(score=0, epoch=epoch)


#evaluate
def eval(epoch):
    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    count, total_count = 0, len(validset)

    for batch in validloader:
        raw_src, src, src_len, raw_tgt, tgt, tgt_len = \
            batch['raw_src'], batch['src'], batch['src_len'], batch['raw_tgt'], batch['tgt'], batch['tgt_len']

        if 'num_oovs' in batch.keys():
            num_oovs = batch['num_oovs']
            oovs = batch['oovs']
        else:
            num_oovs = 0
            oovs = None

        if config.beam_size == 1:
            samples, alignment = model.sample(src, src_len, num_oovs=num_oovs)
        else:
            samples, alignment = model.beam_sample(src, src_len, beam_size=config.beam_size)

        if oovs is not None:
            candidate += [tgt_vocab.convertToLabels(s, dict.EOS, oovs=oov) for s, oov in zip(samples, oovs)]
        else:
            candidate += [tgt_vocab.convertToLabels(s, dict.EOS) for s in samples]
        source += raw_src
        reference += raw_tgt
        alignments += [align for align in alignment]

        count += len(raw_src)
        utils.progress_bar(count, total_count)

    if opt.unk:
    ###replace unk
        cands = []
        for s, c, align in zip(source, candidate, alignments):
            cand = []
            for word, idx in zip(c, align):
                if word == dict.UNK_WORD and idx < len(s):
                    try:
                        cand.append(s[idx])
                    except:
                        cand.append(word)
                        print("%d %d\n" % (len(s), idx))
                else:
                    cand.append(word)

            cands.append(cand)
        candidate = cands

    score = {}

    if hasattr(config,'convert'):
        candidate = utils.convert_to_char(candidate)
        reference = utils.convert_to_char(reference)

    if 'bleu' in config.metric:
        result = utils.eval_bleu(reference, candidate, log_path, config)
        score['bleu'] = float(result.split()[2][:-1])
        logging(result)

    if 'rouge' in config.metric:
        result = utils.eval_rouge(reference, candidate, log_path)
        try:
            score['rouge'] = result['F_measure'][0]
            logging("F_measure: %s Recall: %s Precision: %s\n"
                    % (str(result['F_measure']), str(result['recall']), str(result['precision'])))
            #optim.updateLearningRate(score=score['rouge'], epoch=epoch)
        except:
            logging("Failed to compute rouge score.\n")
            score['rouge'] = 0.0

    if 'multi_rouge' in config.metric:
        result = utils.eval_multi_rouge(reference, candidate, log_path)
        try:
            score['multi_rouge'] = result['F_measure'][0]
            logging("F_measure: %s Recall: %s Precision: %s\n"
                    % (str(result['F_measure']), str(result['recall']), str(result['precision'])))
        except:
            logging("Failed to compute rouge score.\n")
            score['multi_rouge'] = 0.0

    if 'SARI' in config.metric:
        result = utils.eval_SARI(source, reference, candidate, log_path, config)
        logging("SARI score is: %.2f\n" % result)
        score['SARI'] = result

    return score


def save_model(path):
    global updates
    #model_state_dict = model_module.state_dict() if len(opt.gpus) > 1 else model.state_dict()
    model_state_dict = model.state_dict()
    checkpoints = {
        'model': model_state_dict,
        'config': config,
        'optim': optim,
        'updates': updates}
    torch.save(checkpoints, path)


def main():
    for i in range(1, config.epoch+1):
        if not opt.single_pass:
            train(i)
        else:
            eval(i)
            return
    for metric in config.metric:
        logging("Best %s score: %.2f\n" % (metric, max(scores[metric])))
        with open(log_path+metric+'.txt', 'w') as f:
            for i, score in enumerate(scores[metric]):
                f.write(str(i)+','+str(score)+'\n')

if __name__ == '__main__':
    main()
