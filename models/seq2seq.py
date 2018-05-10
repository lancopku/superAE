'''
 @Author: Shuming Ma
 @mail:   shumingma@pku.edu.cn
 @homepage : shumingma.com
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
import data.dict as dict
import models


class seq2seq(nn.Module):

    def __init__(self, config, src_vocab_size, tgt_vocab_size, use_cuda,
                 w2v=None, score_fn=None, weight=0.0, pretrain_updates=0, extend_vocab_size=0, device_ids=None):
        super(seq2seq, self).__init__()
        if w2v is not None:
            src_embedding = w2v['src_emb']
            tgt_embedding = w2v['tgt_emb']
        else:
            src_embedding = None
            tgt_embedding = None

        if 'copy' in score_fn:
            build_encoder = models.copy_rnn_encoder
            build_decoder = models.copy_rnn_decoder
        else:
            build_encoder = models.rnn_encoder
            build_decoder = models.rnn_decoder

        self.encoder = build_encoder(config, src_vocab_size, embedding=src_embedding)
        if config.shared_vocab == False:
            self.decoder = build_decoder(config, tgt_vocab_size, embedding=tgt_embedding, score_fn=score_fn, extend_vocab_size=extend_vocab_size)
        else:
            self.decoder = build_decoder(config, tgt_vocab_size, embedding=self.encoder.embedding, score_fn=score_fn, extend_vocab_size=extend_vocab_size)
        #if len(device_ids) > 0:
        #    self.encoder = nn.DataParallel(self.encoder, device_ids=device_ids, dim=1)
        #    self.decoder = nn.DataParallel(self.decoder, device_ids=device_ids, dim=1)
        self.use_cuda = use_cuda
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.config = config
        self.weight = weight
        if 'emb' in score_fn:
            self.criterion = models.criterion_emb(config.hidden_size, tgt_vocab_size, use_cuda)
        elif 'copy' in score_fn:
            self.criterion = models.copy_criterion(use_cuda)
        else:
            self.criterion = models.criterion(tgt_vocab_size, use_cuda)
        self.log_softmax = nn.LogSoftmax()

    def compute_loss(self, hidden_outputs, targets, loss_fn, updates):
        if loss_fn == 'memory':
            return models.memory_efficiency_cross_entropy_loss(hidden_outputs, self.decoder, targets, self.criterion, self.config)
        elif 'emb' in loss_fn:
            return models.prior_knowledge_loss(hidden_outputs, self.decoder, targets, self.criterion, self.config, updates)
        elif loss_fn == 'copy':
            return models.copy_cross_entropy_loss(hidden_outputs, self.decoder, targets, self.criterion, self.config)
        else:
            return models.cross_entropy_loss(hidden_outputs, self.decoder, targets, self.criterion, self.config)

    def forward(self, src, src_len, tgt, tgt_len, num_oovs):
        lengths, indices = torch.sort(src_len.squeeze(0), dim=0, descending=True)
        src = torch.index_select(src, dim=1, index=indices)
        tgt = torch.index_select(tgt, dim=1, index=indices)

        contexts, state = self.encoder(src, lengths.data.tolist())
        outputs, final_state = self.decoder(tgt[:-1], state, contexts.transpose(0, 1), src=src, num_oovs=num_oovs)
        return outputs, tgt[1:]

    def train_model(self, src, src_len, tgt, tgt_len, loss_fn, updates, optim, num_oovs=0):

        src = Variable(src)
        tgt = Variable(tgt)
        src_len = Variable(src_len).unsqueeze(0)
        tgt_len = Variable(tgt_len).unsqueeze(0)
        if self.use_cuda:
            src = src.cuda()
            tgt = tgt.cuda()
            src_len = src_len.cuda()
            tgt_len = tgt_len.cuda()

        outputs, targets = self(src, src_len, tgt, tgt_len, num_oovs)
        loss, num_total, num_correct = self.compute_loss(outputs, targets, loss_fn, updates)
        loss.backward()
        optim.step()

        return loss.data[0], num_total, num_correct

    def sample(self, src, src_len, num_oovs=0):

        if self.use_cuda:
            src = src.cuda()
            src_len = src_len.cuda()

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, ind = torch.sort(indices)
        src = Variable(torch.index_select(src, dim=1, index=indices), volatile=True)
        bos = Variable(torch.ones(src.size(1)).long().fill_(dict.BOS), volatile=True)

        if self.use_cuda:
            bos = bos.cuda()

        contexts, state = self.encoder(src, lengths.tolist())
        sample_ids, final_outputs = self.decoder.sample([bos], state, contexts.transpose(0, 1),src=src, num_oovs=num_oovs)
        _, attns_weight = final_outputs
        alignments = attns_weight.max(2)[1]
        sample_ids = torch.index_select(sample_ids.data, dim=1, index=ind)
        alignments = torch.index_select(alignments.data, dim=1, index=ind)
        #targets = tgt[1:]

        return sample_ids.t(), alignments.t()


    def beam_sample(self, src, src_len, beam_size = 1, num_oovs=0):

        #beam_size = self.config.beam_size
        batch_size = src.size(1)

        # (1) Run the encoder on the src. Done!!!!
        if self.use_cuda:
            src = src.cuda()
            src_len = src_len.cuda()

        lengths, indices = torch.sort(src_len, dim=0, descending=True)
        _, ind = torch.sort(indices)
        src = Variable(torch.index_select(src, dim=1, index=indices), volatile=True)
        contexts, encState = self.encoder(src, lengths.tolist())

        #  (1b) Initialize for the decoder.
        def var(a):
            return Variable(a, volatile=True)

        def rvar(a):
            return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        # Repeat everything beam_size times.
        contexts = rvar(contexts.data).transpose(0, 1)
        decState = (rvar(encState[0].data), rvar(encState[1].data))
        #decState.repeat_beam_size_times(beam_size)
        beam = [models.Beam(beam_size, n_best=1,
                          cuda=self.use_cuda)
                for __ in range(batch_size)]
        self.decoder.attention.init_context(contexts)
        # (2) run the decoder to generate sentences, using beam search.

        for i in range(self.config.max_tgt_len):

            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.
            inp = var(torch.stack([b.getCurrentState() for b in beam])
                      .t().contiguous().view(-1))

            # Run one step.
            output, decState, attn = self.decoder.sample_one(inp, decState, contexts)
            # decOut: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            output = unbottle(self.log_softmax(output))
            attn = unbottle(attn)
                # beam x tgt_vocab

            # (c) Advance each beam.
            # update state
            for j, b in enumerate(beam):
                b.advance(output.data[:, j], attn.data[:, j])
                b.beam_update(decState, j)

        # (3) Package everything up.
        allHyps, allScores, allAttn = [], [], []

        for j in ind:
            b = beam[j]
            n_best = 1
            scores, ks = b.sortFinished(minimum=n_best)
            hyps, attn = [], []
            for i, (times, k) in enumerate(ks[:n_best]):
                hyp, att = b.getHyp(times, k)
                hyps.append(hyp)
                attn.append(att.max(1)[1])
            allHyps.append(hyps[0])
            allScores.append(scores[0])
            allAttn.append(attn[0])

        #print(allHyps)
        #print(allAttn)
        return allHyps, allAttn