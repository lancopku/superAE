import torch
import torch.nn as nn
import torch.utils.data
import models
import data.dataloader as dataloader
import data.utils as utils
import data.dict as dict
from optims import Optim
from data.dataloader import dataset

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
parser.add_argument('-restore', required=True,
                    help="restore checkpoint")
parser.add_argument('-seed', type=int, default=1234,
                    help="Random seed")
parser.add_argument('-model', default='seq2seq', type=str,
                    help="Model selection")
parser.add_argument('-score', default='', type=str,
                    help="score_fn")
parser.add_argument('-pretrain', action='store_true',
                    help="load pretrain embedding")
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
parser.add_argument('-beam_size', type=int, default=0,
                    help="beam search size")
parser.add_argument('-trun_src', type=int, default=0,
                    help="Maximum source sequence length")
parser.add_argument('-trun_tgt', type=int, default=0,
                    help="Maximum target sequence length")
parser.add_argument('-test_src', default='', type=str,
                    help="Path to the test source data")
parser.add_argument('-test_tgt', default='', type=str,
                     help="Path to the test target data")

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
    #cudnn.benchmark = True


def makeData(srcFile, tgtFile, srcDicts, tgtDicts, sort=False):
    src, tgt = [], []
    raw_src, raw_tgt = [], []
    sizes = []
    count, ignored = 0, 0

    print('Processing %s & %s ...' % (srcFile, tgtFile))
    srcF = open(srcFile, encoding='utf8')
    tgtF = open(tgtFile, encoding='utf8')

    while True:
        sline = srcF.readline()
        tline = tgtF.readline()

        # normal end of file
        if sline == "" and tline == "":
            break

        # source or target does not have same number of lines
        if sline == "" or tline == "":
            print('WARNING: source and target do not have the same number of sentences')
            break

        sline = sline.strip()
        tline = tline.strip()

        # source and/or target are empty
        if sline == "" or tline == "":
            print('WARNING: ignoring an empty line ('+str(count+1)+')')
            ignored += 1
            continue

        sline = sline.lower()
        tline = tline.lower()

        srcWords = sline.split()
        tgtWords = tline.split()

        if opt.trun_src > 0:
            srcWords = srcWords[:opt.trun_src]
        if opt.trun_tgt > 0:
            tgtWords = tgtWords[:opt.trun_tgt]

        srcWords = [word+" " for word in srcWords]
        tgtWords = [word+" " for word in tgtWords]

        src += [srcDicts.convertToIdx(srcWords,
                                      dict.UNK_WORD)]
        tgt += [tgtDicts.convertToIdx(tgtWords,
                                      dict.UNK_WORD,
                                      dict.BOS_WORD,
                                      dict.EOS_WORD)]
        raw_src += [srcWords]
        raw_tgt += [tgtWords]
        sizes += [len(srcWords)]

        count += 1

        if count % 1000 == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()

    print('Prepared %d sentences (%d ignored due to length == 0)' %
          (len(src), ignored))

    return dataset(src, tgt, raw_src, raw_tgt)


#data
print('loading data...\n')
start_time = time.time()
datas = torch.load(config.data)
print('loading time cost: %.3f' % (time.time()-start_time))

src_vocab, tgt_vocab = datas['dicts']['src'], datas['dicts']['tgt']
config.src_vocab = src_vocab.size()
config.tgt_vocab = tgt_vocab.size()

if not opt.test_src:
    testset = datas['valid']
else:
    testset = makeData(opt.test_src, opt.test_tgt, src_vocab, tgt_vocab)

testloader = dataloader.get_loader(testset, batch_size=config.batch_size, shuffle=False, num_workers=2, padding=dataloader.padding)

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
report_vocab, report_tot_vocab = 0, 0
scores = [[] for metric in config.metric]
scores = collections.OrderedDict(zip(config.metric, scores))

if opt.beam_size > 0:
    beam_size = opt.beam_size
else:
    beam_size = config.beam_size

#evaluate
def eval(epoch):
    model.eval()
    reference, candidate, source, alignments = [], [], [], []
    count, total_count = 0, len(testset)

    for batch in testloader:
        raw_src, src, src_len, raw_tgt, tgt, tgt_len = \
            batch['raw_src'], batch['src'], batch['src_len'], batch['raw_tgt'], batch['tgt'], batch['tgt_len']

        if 'num_oovs' in batch.keys():
            num_oovs = batch['num_oovs']
            oovs = batch['oovs']
        else:
            num_oovs = 0
            oovs = None

        if beam_size == 1:
            samples, alignment = model.sample(src, src_len, num_oovs=num_oovs)
        else:
            samples, alignment = model.beam_sample(src, src_len, beam_size=beam_size, num_oovs=num_oovs)

        candidate += [tgt_vocab.convertToLabels(s, dict.EOS) for s in samples]
        source += raw_src
        reference += raw_tgt
        alignments += [align for align in alignment]

        count += len(raw_src)
        utils.progress_bar(count, total_count)

    #if opt.unk:
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
        if opt.reduce:
            phrase_set = {}
            mask = [1 for _ in range(len(cand))]
            for id in range(1,len(cand)):
                phrase = cand[id-1]+" "+cand[id]
                if phrase in phrase_set.keys():
                    mask[id-1] = 0
                    mask[id] = 0
                else:
                    phrase_set[phrase] = True
            cand = [word for word, m in zip(cand, mask) if m == 1]
        cands.append(cand)
    candidate = cands

    if opt.group:
        lengths = [90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]
        group_cand, group_ref = collections.OrderedDict(), collections.OrderedDict()
        for length in lengths:
            group_cand[length] = []
            group_ref[length] = []
        total_length = []
        for s, c, r in zip(source, candidate, reference):
            length = len(s)
            total_length.append(length)
            for l in lengths:
                if length <= l:
                    group_ref[l].append(r)
                    group_cand[l].append(c)
                    break
        print("min length %d, max length %d" % (min(total_length), max(total_length)))
        if 'rouge' in config.metric:
            for l in lengths:
                print("length %d, count %d" % (l, len(group_cand[l])))
                result = utils.eval_rouge(group_ref[l], group_cand[l], log_path)
                try:
                    logging("length: %d F_measure: %s Recall: %s Precision: %s\n\n"
                            % (l, str(result['F_measure']), str(result['recall']), str(result['precision'])))
                except:
                    logging("Failed to compute rouge score.\n")


    score = {}

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



if __name__ == '__main__':
    eval(0)
