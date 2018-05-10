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


class ae(nn.Module):

    def __init__(self, config, src_vocab_size, tgt_vocab_size, use_cuda,
                 w2v=None, score_fn=None, weight=0.0, pretrain_updates=0, extend_vocab_size=0, device_ids=None):
        super(ae, self).__init__()
        if w2v is not None:
            src_embedding = w2v['src_emb']
            tgt_embedding = w2v['tgt_emb']
        else:
            src_embedding = None
            tgt_embedding = None

        self.encoder_s2s = models.gated_rnn_encoder(config, src_vocab_size, embedding=src_embedding)

        if config.shared_vocab == False:
            self.decoder = models.rnn_decoder(config, tgt_vocab_size, embedding=tgt_embedding, score_fn=score_fn)
        else:
            self.decoder = models.rnn_decoder(config, tgt_vocab_size, embedding=self.encoder_s2s.embedding, score_fn=score_fn)

        self.encoder_ae = models.rnn_encoder(config, src_vocab_size, embedding=self.decoder.embedding)

        self.use_cuda = use_cuda
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.config = config
        self.weight = weight
        self.pretrain_updates = pretrain_updates
        if 'emb' in score_fn:
            self.criterion = models.criterion_emb(config.hidden_size, tgt_vocab_size, use_cuda)
        else:
            self.criterion = models.criterion(tgt_vocab_size, use_cuda)
        self.log_softmax = nn.LogSoftmax()
        if score_fn.startswith('dis'):
            self.discriminator = nn.Linear(config.num_layers*config.hidden_size*2, 1)
            self.sigmoid = nn.Sigmoid()
        if score_fn.endswith('map'):
            self.h_map = nn.Linear(config.hidden_size, config.hidden_size)
            self.c_map = nn.Linear(config.hidden_size, config.hidden_size)
        self.score_fn = score_fn

    def compute_loss(self, hidden_outputs, targets, loss_fn, updates):
        if loss_fn == 'memory':
            return models.memory_efficiency_cross_entropy_loss(hidden_outputs, self.decoder, targets, self.criterion, self.config)
        elif 'emb' in loss_fn:
            return models.prior_knowledge_loss(hidden_outputs, self.decoder, targets, self.criterion, self.config, updates)
        else:
            return models.cross_entropy_loss(hidden_outputs, self.decoder, targets, self.criterion, self.config)

    def forward(self, src, src_len, tgt, tgt_len, use_s2s=True):
        lengths, indices = torch.sort(src_len.squeeze(0), dim=0, descending=True)
        src = torch.index_select(src, dim=1, index=indices)
        tgt = torch.index_select(tgt, dim=1, index=indices)
        if use_s2s:
            contexts, state = self.encoder_s2s(src, lengths.data.tolist())
        else:
            contexts, state = self.encoder_ae(src, lengths.data.tolist())

        outputs, final_state = self.decoder(tgt[:-1], state, contexts.transpose(0, 1), use_attention=use_s2s)

        return outputs, tgt[1:], state


    def encode_representation(self, src, src_len, tgt, tgt_len, use_s2s=True):
        lengths, indices = torch.sort(src_len.squeeze(0), dim=0, descending=True)
        src = torch.index_select(src, dim=1, index=indices)
        tgt = torch.index_select(tgt, dim=1, index=indices)
        if use_s2s:
            contexts, state = self.encoder_s2s(src, lengths.data.tolist())
        else:
            contexts, state = self.encoder_ae(src, lengths.data.tolist())
        return state


    def s2s_loss(self, src, src_len, tgt, tgt_len, loss_fn, updates):
        outputs, targets, state_src = self(src, src_len, tgt, tgt_len)
        loss_s2s, num_total, num_correct = self.compute_loss(outputs, targets, loss_fn, updates)
        return loss_s2s, state_src, num_total, num_correct

    def ae_loss(self, tgt, tgt_len, loss_fn, updates):
        outputs, targets, state_tgt = self(tgt[1:-1], tgt_len - 2, tgt, tgt_len, use_s2s=False)
        loss_ae, _, _ = self.compute_loss(outputs, targets, loss_fn, updates)
        return loss_ae, state_tgt


    def train_model(self, src, src_len, tgt, tgt_len, loss_fn, updates, optim, num_oovs=0):

        #if updates > self.pretrain_updates and self.pretrain_updates > 0:
        #    tgt, _ = self.sample(src, src_len)
        #    tgt_len = [list(t).index(dict.EOS) if dict.EOS in list(t) else len(list(t)) for t in tgt]
        #    tgt_len = [l+2 if l > 0 else 3 for l in tgt_len]

        #    tgt_pad = torch.zeros(len(tgt), max(tgt_len)).long()
        #    for i, s in enumerate(tgt):
        #        tgt_pad[i, 0] = dict.BOS
        #        end = tgt_len[i]-2
        #        tgt_pad[i, 1:end+1] = s[:end]
        #        tgt_pad[i, end+1] = dict.EOS

            #print(tgt, tgt_len, tgt_pad)

        #    tgt_len = torch.Tensor(tgt_len)
        #    tgt = tgt_pad.t()

            #bos = torch.ones(1, src.size(1)).long().fill_(dict.BOS)
            #eos = torch.ones(1, src.size(1)).long().fill_(dict.EOS)
            #print(bos, eos, tgt)
            #tgt = torch.cat((bos, tgt, eos), dim=0)

        src = Variable(src)
        tgt = Variable(tgt)
        src_len = Variable(src_len).unsqueeze(0)
        tgt_len = Variable(tgt_len).unsqueeze(0)
        if self.use_cuda:
            src = src.cuda()
            tgt = tgt.cuda()
            src_len = src_len.cuda()
            tgt_len = tgt_len.cuda()

        loss = 0

        loss_s2s, state_src, num_total, num_correct = self.s2s_loss(src, src_len, tgt, tgt_len, loss_fn, updates)
        loss += loss_s2s

        if 'step' in loss_fn:
            loss_s2s.backward(retain_graph=True)

        loss_ae, state_tgt = self.ae_loss(tgt, tgt_len, loss_fn, updates)
        loss += loss_ae

        if 'step' in loss_fn:
            loss_ae.backward(retain_graph=True)
            #state_src = self.encode_representation(src, src_len, tgt, tgt_len)
            #state_tgt = self.encode_representation(tgt[1:-1], tgt_len - 2, tgt, tgt_len, use_s2s=False)

        h_src, c_src = state_src
        h_tgt, c_tgt = state_tgt

        if self.score_fn.endswith('map'):
            h_src_detach = Variable(h_src.data)
            c_src_detach = Variable(c_src.data)
            h_tgt_detach = self.h_map(Variable(h_tgt.data))
            c_tgt_detach = self.c_map(Variable(c_tgt.data))
            h_tgt = self.h_map(h_tgt)
            c_tgt = self.c_map(c_tgt)
        else:
            h_src_detach = Variable(h_src.data)
            c_src_detach = Variable(c_src.data)
            h_tgt_detach = Variable(h_tgt.data)
            c_tgt_detach = Variable(c_tgt.data)

        loss_supervise = 0
        if self.weight > 0:
            if 'l1' in self.score_fn:
                loss_reg = torch.mean(torch.abs(h_src - h_tgt)) + torch.mean(torch.abs(c_src - c_tgt))
            elif 'cos' in self.score_fn:
                loss_reg = torch.mean(torch.mul(h_src, h_tgt)/torch.norm(h_src)/torch.norm(h_tgt)) + \
                           torch.mean(torch.mul(c_src, c_tgt)/torch.norm(c_src)/torch.norm(c_tgt))
            else:
                loss_reg = torch.mean(torch.pow(h_src-h_tgt, 2)) + torch.mean(torch.pow(c_src-c_tgt, 2))
            loss_supervise += self.weight * loss_reg

        if self.score_fn.startswith('dis'):
            batch_size = h_src.size(1)
            src_hidden_vector = torch.cat([h_src_detach, c_src_detach], dim=-1).transpose(0, 1).contiguous().view(batch_size, -1)
            tgt_hidden_vector = torch.cat([h_tgt_detach, c_tgt_detach], dim=-1).transpose(0, 1).contiguous().view(batch_size, -1)

            if updates < self.pretrain_updates or (updates >= self.pretrain_updates and updates % 5 == 0):
                fake_label = self.sigmoid(self.discriminator(Variable(src_hidden_vector.data)))
                true_label = self.sigmoid(self.discriminator(Variable(tgt_hidden_vector.data)))
                dis_loss = -torch.log(true_label) - torch.log(1-fake_label)
                loss_supervise += self.weight * torch.mean(dis_loss) / self.config.hidden_size

            if updates >= self.pretrain_updates:
                #print(src_hidden_vector.size())
                #print(self.discriminator.weight.t().data.size())
                src_label = self.sigmoid(torch.matmul(src_hidden_vector, Variable(self.discriminator.weight.t().data)))
                tgt_label = self.sigmoid(torch.matmul(tgt_hidden_vector, Variable(self.discriminator.weight.t().data)))
                gen_loss = -torch.log(src_label) - torch.log(1-tgt_label)
                loss_supervise += self.weight * torch.mean(gen_loss) / self.config.hidden_size

        loss += loss_supervise
        if 'step' in loss_fn:
            loss_supervise.backward()
        else:
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

        contexts, state = self.encoder_s2s(src, lengths.tolist())
        sample_ids, final_outputs = self.decoder.sample([bos], state, contexts.transpose(0, 1))
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
        contexts, encState = self.encoder_s2s(src, lengths.tolist())

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