'''
 @Author: Shuming Ma
 @mail:   shumingma@pku.edu.cn
 @homepage : shumingma.com
'''
import argparse
import torch
import numpy
import data.dict as dict
from data.dataloader import dataset

parser = argparse.ArgumentParser(description='preprocess.py')

##
## **Preprocess Options**
##

parser.add_argument('-config',    help="Read options from this file")

parser.add_argument('-train_src', required=True,
                    help="Path to the training source data")
parser.add_argument('-train_tgt', required=True,
                    help="Path to the training target data")
parser.add_argument('-valid_src', required=True,
                    help="Path to the validation source data")
parser.add_argument('-valid_tgt', required=True,
                     help="Path to the validation target data")

parser.add_argument('-save_data', required=True,
                    help="Output file for the prepared data")

parser.add_argument('-src_vocab_size', type=int, default=50000,
                    help="Size of the source vocabulary")
parser.add_argument('-tgt_vocab_size', type=int, default=50000,
                    help="Size of the target vocabulary")
parser.add_argument('-src_vocab',
                    help="Path to an existing source vocabulary")
parser.add_argument('-tgt_vocab',
                    help="Path to an existing target vocabulary")


parser.add_argument('-src_length', type=int, default=0,
                    help="Maximum source sequence length")
parser.add_argument('-tgt_length', type=int, default=0,
                    help="Maximum target sequence length")
parser.add_argument('-trun_src', type=int, default=0,
                    help="Maximum source sequence length")
parser.add_argument('-trun_tgt', type=int, default=0,
                    help="Maximum target sequence length")

parser.add_argument('-src_suf', default='src',
                    help="the suffix of the source filename")
parser.add_argument('-tgt_suf', default='tgt',
                    help="the suffix of the target filename")

parser.add_argument('-shuffle',    type=int, default=0,
                    help="Shuffle data")
parser.add_argument('-seed',       type=int, default=3435,
                    help="Random seed")

parser.add_argument('-lower', action='store_true', help='lowercase data')
parser.add_argument('-src_char', action='store_true', help='character based encoding')
parser.add_argument('-tgt_char', action='store_true', help='character based decoding')
parser.add_argument('-share', action='store_true', help='share the vocabulary between source and target')

parser.add_argument('-report_every', type=int, default=100000,
                    help="Report status every this many sentences")

opt = parser.parse_args()

torch.manual_seed(opt.seed)

def makeVocabulary(filename, size):
    vocab = dict.Dict([dict.PAD_WORD, dict.UNK_WORD,
                       dict.BOS_WORD, dict.EOS_WORD], lower=opt.lower)

    if type(filename) == str:
        filename = [filename]

    for _filename in filename:
        if opt.src_suf in _filename:
            max_tokens = opt.trun_src
            max_lengths = opt.src_length
            char = opt.src_char
            print(_filename, ' max tokens: ', max_tokens)
            print(_filename, ' max lengths: ', max_lengths)
        elif opt.tgt_suf in _filename:
            max_tokens = opt.trun_tgt
            max_lengths = opt.tgt_length
            char = opt.tgt_char
            print(_filename, ' max tokens: ', max_tokens)
            print(_filename, ' max lengths: ', max_lengths)
        with open(_filename, encoding='utf8') as f:
            for sent in f.readlines():
                if char:
                    tokens = list(sent.strip())
                else:
                    tokens = sent.strip().split()
                if max_lengths > 0 and len(tokens) > max_lengths:
                    continue
                if max_tokens > 0:
                    tokens = tokens[:max_tokens]
                for word in tokens:
                    vocab.add(word + " ")


    originalSize = vocab.size()
    if size == 0:
        size = originalSize
    vocab = vocab.prune(size)
    print('Created dictionary of size %d (pruned from %d)' %
          (vocab.size(), originalSize))

    return vocab


def initVocabulary(name, dataFile, vocabFile, vocabSize):

    vocab = None
    if vocabFile is not None:
        # If given, load existing word dictionary.
        print('Reading ' + name + ' vocabulary from \'' + vocabFile + '\'...')
        vocab = dict.Dict()
        vocab.loadFile(vocabFile)
        print('Loaded ' + str(vocab.size()) + ' ' + name + ' words')

    if vocab is None:
        # If a dictionary is still missing, generate it.
        print('Building ' + name + ' vocabulary...')
        genWordVocab = makeVocabulary(dataFile, vocabSize)

        vocab = genWordVocab

    return vocab


def saveVocabulary(name, vocab, file):
    print('Saving ' + name + ' vocabulary to \'' + file + '\'...')
    vocab.writeFile(file)


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

        if opt.lower:
            sline = sline.lower()
            tline = tline.lower()

        srcWords = sline.split() if not opt.src_char else list(sline)
        tgtWords = tline.split() if not opt.tgt_char else list(tline)


        if (opt.src_length == 0 or len(srcWords) <= opt.src_length) and \
                (opt.tgt_length == 0 or len(tgtWords) <= opt.tgt_length):

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
        else:
            ignored += 1

        count += 1

        if count % opt.report_every == 0:
            print('... %d sentences prepared' % count)

    srcF.close()
    tgtF.close()

    if opt.shuffle == 1:
        print('... shuffling sentences')
        perm = torch.randperm(len(src))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        sizes = [sizes[idx] for idx in perm]
        raw_src = [raw_src[idx] for idx in perm]
        raw_tgt = [raw_tgt[idx] for idx in perm]

    if sort:
        print('... sorting sentences by size')
        _, perm = torch.sort(torch.Tensor(sizes))
        src = [src[idx] for idx in perm]
        tgt = [tgt[idx] for idx in perm]
        raw_src = [raw_src[idx] for idx in perm]
        raw_tgt = [raw_tgt[idx] for idx in perm]

    print('Prepared %d sentences (%d ignored due to length == 0 or > %d)' %
          (len(src), ignored, opt.src_length))

    return dataset(src, tgt, raw_src, raw_tgt)


def main():

    dicts = {}
    if opt.share:
        assert opt.src_vocab_size == opt.tgt_vocab_size
        print('share the vocabulary between source and target')
        dicts['src'] = initVocabulary('source and target',
                                      [opt.train_src, opt.train_tgt],
                                      opt.src_vocab,
                                      opt.src_vocab_size)
        dicts['tgt'] = dicts['src']
    else:
        dicts['src'] = initVocabulary('source', opt.train_src, opt.src_vocab,
                                      opt.src_vocab_size)
        dicts['tgt'] = initVocabulary('target', opt.train_tgt, opt.tgt_vocab,
                                      opt.tgt_vocab_size)

    print('Preparing training ...')
    train = makeData(opt.train_src, opt.train_tgt, dicts['src'], dicts['tgt'])

    print('Preparing validation ...')
    valid = makeData(opt.valid_src, opt.valid_tgt, dicts['src'], dicts['tgt'])

    if opt.src_vocab is None:
        saveVocabulary('source', dicts['src'], opt.save_data + '.src.dict')
    if opt.tgt_vocab is None:
        saveVocabulary('target', dicts['tgt'], opt.save_data + '.tgt.dict')


    print('Saving data to \'' + opt.save_data + '.train.pt\'...')
    save_data = {'dicts': dicts,
                 'train': train,
                 'valid': valid}
    torch.save(save_data, opt.save_data + '.train.pt')


if __name__ == "__main__":
    main()