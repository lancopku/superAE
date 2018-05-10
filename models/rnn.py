'''
 @Author: Shuming Ma
 @mail:   shumingma@pku.edu.cn
 @homepage : shumingma.com
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import data.dict as dict
import models

class StackedGRU(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0 = hidden
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, h_0[i])
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        return input, h_1


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, hidden_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)



class encoder_decoder(nn.Module):

    def __init__(self, encoder, decoder, use_attention, compute_score=False):
        super(encoder_decoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.use_attention = use_attention
        self.compute_score = compute_score

    def forward(self, src, src_len, tgt, tgt_len):
        lengths, indices = torch.sort(src_len.squeeze(0), dim=0, descending=True)
        src = torch.index_select(src, dim=1, index=indices)
        tgt = torch.index_select(tgt, dim=1, index=indices)

        contexts, state = self.encoder(src, lengths.data.tolist())
        outputs, final_state = self.decoder(tgt[:-1], state, contexts.transpose(0, 1),
                                            use_attention=self.use_attention, compute_score=self.compute_score)
        return outputs, tgt[1:], state



class rnn_encoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None):
        super(rnn_encoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        #self.hidden_size = config.hidden_size // 2 if config.bidirectional else config.hidden_size
        if hasattr(config, 'enc_num_layers'):
            num_layers = config.enc_num_layers
        else:
            num_layers = config.num_layers
        self.hidden_size = config.hidden_size

        if hasattr(config,'gru'):
            self.rnn = nn.GRU(input_size=config.emb_size, hidden_size=config.hidden_size,
                           num_layers=num_layers, dropout=config.dropout, bidirectional=config.bidirectional)
        else:
            self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                           num_layers=num_layers, dropout=config.dropout, bidirectional=config.bidirectional)

        self.config = config


    def forward(self, input, lengths):
        embs = pack(self.embedding(input), lengths)
        outputs, state = self.rnn(embs)
        outputs = unpack(outputs)[0]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        #if self.config.bidirectional:
        #    h, c = state
        #    h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        #    c = torch.cat([c[0:c.size(0):2], c[1:c.size(0):2]], 2)
        #    state = (h, c)
        return outputs, state


class gated_rnn_encoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None):
        super(gated_rnn_encoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        if hasattr(config, 'enc_num_layers'):
            num_layers = config.enc_num_layers
        else:
            num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                           num_layers=num_layers, dropout=config.dropout, bidirectional=config.bidirectional)
        self.config = config
        self.tanh_gated = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.Tanh())
        self.sig_gated = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.Sigmoid())

    def forward(self, input, lengths):
        embs = pack(self.embedding(input), lengths)
        outputs, state = self.rnn(embs)
        outputs = unpack(outputs)[0]
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        outputs = self.tanh_gated(outputs/0.9) * outputs
        outputs = self.sig_gated(outputs) * outputs

        return outputs, state


class rnn_decoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None, score_fn=None, extend_vocab_size=0):
        super(rnn_decoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        if hasattr(config,'gru'):
            self.rnn = StackedGRU(input_size=config.emb_size, hidden_size=config.hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout)
        else:
            self.rnn = StackedLSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout)

        self.score_fn = score_fn
        if self.score_fn.startswith('general'):
            self.linear = nn.Linear(config.hidden_size, config.emb_size)
        elif score_fn.startswith('concat'):
            self.linear_query = nn.Linear(config.hidden_size, config.hidden_size)
            self.linear_weight = nn.Linear(config.emb_size, config.hidden_size)
            self.linear_v = nn.Linear(config.hidden_size, 1)
        elif not self.score_fn.startswith('dot'):
            self.linear = nn.Linear(config.hidden_size, vocab_size)

        if self.score_fn.startswith('copy'):
            self.gen_linear = nn.Sequential(nn.Linear(config.hidden_size, 1), nn.Sigmoid())

        if hasattr(config, 'att_act'):
            activation = config.att_act
            print('use attention activation %s' % activation)
        else:
            activation = None

        if activation == 'bahd':
            self.attention = models.bahdanau_attention(config.hidden_size, config.emb_size, activation)
        else:
            self.attention = models.luong_attention(config.hidden_size, activation)
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.config = config
        self.extend_vocab_size = extend_vocab_size

    def forward(self, inputs, init_state, contexts, use_attention=True, compute_score=False, src=None, num_oovs=0):
        embs = self.embedding(inputs)
        outputs, state, attns = [], init_state, []
        self.attention.init_context(contexts)
        for emb in embs.split(1):
            x = emb.squeeze(0)
            output, state = self.rnn(x, state)
            if use_attention:
                output, attn_weights = self.attention(output, x, contexts)
                attns.append(attn_weights)
                if compute_score:
                    output = self.compute_score(output, src, attn_weights)
            output = self.dropout(output)
            outputs += [output]
        if not compute_score:
            outputs = torch.stack(outputs)
        #outputs = self.linear(outputs.view(-1, self.hidden_size))
        return outputs, state

    def compute_score(self, hiddens, src=None, attns=None):
        if self.score_fn.startswith('general'):
            if self.score_fn.endswith('not'):
                scores = torch.matmul(self.linear(hiddens), Variable(self.embedding.weight.t().data))
            else:
                scores = torch.matmul(self.linear(hiddens), self.embedding.weight.t())
        elif self.score_fn.startswith('concat'):
            if self.score_fn.endswith('not'):
                scores = self.linear_v(torch.tanh(self.linear_query(hiddens).unsqueeze(1) + \
                                      self.linear_weight(Variable(self.embedding.weight.data)).unsqueeze(0))).squeeze(2)
            else:
                scores = self.linear_v(torch.tanh(self.linear_query(hiddens).unsqueeze(1) + \
                                      self.linear_weight(self.embedding.weight).unsqueeze(0))).squeeze(2)
        elif self.score_fn.startswith('dot'):
            if self.score_fn.endswith('not'):
                scores = torch.matmul(hiddens, Variable(self.embedding.weight.t().data))
            elif self.score_fn.endswith('non'):
                scores = torch.matmul(hiddens, self.embedding.weight.t())
            else:
                scores = torch.matmul(hiddens, self.embedding.weight.t())
        elif self.score_fn.startswith('copy'):
            batch_size, hidden_size = hiddens.size(0), hiddens.size(1)
            vocab_dists = torch.nn.functional.softmax(self.linear(hiddens))
            extend_dists = Variable(torch.zeros(batch_size, self.extend_vocab_size)).cuda()
            # print(vocab_dists.size())
            # print(extend_dists.size())
            vocab_dists = torch.cat([vocab_dists, extend_dists], dim=-1)

            copy_dists = Variable(torch.zeros(vocab_dists.size())).cuda().scatter_(1, src.t(), attns)

            p_gen = self.gen_linear(hiddens)
            scores = vocab_dists * p_gen + copy_dists * (1-p_gen)
            #scores = vocab_dists + copy_dists
        else:
            scores = self.linear(hiddens)
        return scores

    def sample(self, input, init_state, contexts, src=None, num_oovs=0):
        #emb = self.embedding(input)
        inputs, outputs, sample_ids, state = [], [], [], init_state
        attns = []
        inputs += input
        max_time_step = self.config.max_tgt_len

        self.attention.init_context(contexts)
        for i in range(max_time_step):
            output, state, attn_weights = self.sample_one(inputs[i], state, contexts, src=None)
            predicted = output.max(1)[1]
            inputs += [predicted]
            sample_ids += [predicted]
            outputs += [output]
            attns += [attn_weights]

        sample_ids = torch.stack(sample_ids)
        attns = torch.stack(attns)
        return sample_ids, (outputs, attns)

    def sample_one(self, input, state, contexts, src=None, num_oovs=0):
        emb = self.embedding(input)
        output, state = self.rnn(emb, state)
        hidden, attn_weigths = self.attention(output, emb, contexts)
        output = self.compute_score(hidden, src=src)

        return output, state, attn_weigths



class copy_rnn_encoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None):
        super(copy_rnn_encoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        self.hidden_size = config.hidden_size // 2 if config.bidirectional else config.hidden_size

        self.rnn = nn.LSTM(input_size=config.emb_size, hidden_size=self.hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout, bidirectional=config.bidirectional)
        self.vocab_size=vocab_size
        self.config = config

    def forward(self, input, lengths):

        mask = torch.ge(input, self.vocab_size).float()
        input_in_vocab = Variable(torch.zeros(input.size()).fill_(dict.UNK)).cuda().long()
        input_in_vocab = mask.long() * input_in_vocab + (1 - mask).long() * input

        embs = pack(self.embedding(input_in_vocab), lengths)
        outputs, state = self.rnn(embs)
        outputs = unpack(outputs)[0]
        if self.config.bidirectional:
            h, c = state
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            c = torch.cat([c[0:c.size(0):2], c[1:c.size(0):2]], 2)
            state = (h, c)
        return outputs, state



class copy_rnn_decoder(nn.Module):

    def __init__(self, config, vocab_size, embedding=None, score_fn=None, extend_vocab_size=0):
        super(copy_rnn_decoder, self).__init__()
        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, config.emb_size)
        self.rnn = StackedLSTM(input_size=config.emb_size, hidden_size=config.hidden_size,
                           num_layers=config.num_layers, dropout=config.dropout)

        self.score_fn = score_fn

        self.linear = nn.Linear(config.hidden_size, vocab_size)
        self.gen_linear = nn.Sequential(nn.Linear(config.hidden_size*3+config.emb_size, 1), nn.Sigmoid())

        if hasattr(config, 'att_act'):
            activation = config.att_act
            print('use attention activation %s' % activation)
        else:
            activation = None

        self.attention = models.luong_attention(config.hidden_size, activation)
        self.hidden_size = config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.config = config
        self.extend_vocab_size = extend_vocab_size
        self.vocab_size = vocab_size

    def forward(self, inputs, init_state, contexts, src, num_oovs, use_attention=True):

        mask = torch.ge(inputs, self.vocab_size).float()
        input_in_vocab = Variable(torch.zeros(inputs.size()).fill_(dict.UNK)).cuda().long()
        input_in_vocab = mask.long() * input_in_vocab + (1 - mask).long() * inputs

        embs = self.embedding(input_in_vocab)
        outputs, state, attns = [], init_state, []
        self.attention.init_context(contexts)
        for emb in embs.split(1):
            x = emb.squeeze(0)
            output, state = self.rnn(x, state)
            if use_attention:
                output, attn_weights = self.attention(output, contexts)
                attns.append(attn_weights)
                output = self.compute_score(output, num_oovs, src, attn_weights, states=state, x=x)
            output = self.dropout(output)
            outputs += [output]

        return outputs, state

    def compute_score(self, hiddens, num_oovs, src, attns, states, x):

        batch_size, hidden_size = hiddens.size(0), hiddens.size(1)
        vocab_dists = torch.nn.functional.softmax(self.linear(hiddens))
        extend_dists = Variable(torch.zeros(batch_size, num_oovs)).cuda()
        # print(vocab_dists.size())
        # print(extend_dists.size())
        vocab_dists = torch.cat([vocab_dists, extend_dists], dim=-1)

        copy_dists = Variable(torch.zeros(vocab_dists.size())).cuda().scatter_(1, src.t(), attns)

        #print(hiddens.size(), states[0].size(), states[1].size(),x.size())
        p_gen = self.gen_linear(torch.cat([hiddens,states[0][-1],states[1][-1],x],-1))
        #p_gen = self.gen_linear(hiddens)

        #print(p_gen.size())
        #print(vocab_dists.size())
        #print(copy_dists.size())
        scores = vocab_dists * p_gen + copy_dists * (1-p_gen)

        return torch.log(scores+1e-10)

    def sample(self, input, init_state, contexts, src, num_oovs):
        #emb = self.embedding(input)
        inputs, outputs, sample_ids, state = [], [], [], init_state
        attns = []
        inputs += input
        max_time_step = self.config.max_tgt_len

        self.attention.init_context(contexts)
        for i in range(max_time_step):
            output, state, attn_weights = self.sample_one(inputs[i], state, contexts, src=src, num_oovs=num_oovs)
            predicted = output.max(1)[1]
            inputs += [predicted]
            sample_ids += [predicted]
            outputs += [output]
            attns += [attn_weights]

        sample_ids = torch.stack(sample_ids)
        attns = torch.stack(attns)
        return sample_ids, (outputs, attns)

    def sample_one(self, input, state, contexts, src, num_oovs):

        mask = torch.ge(input, self.vocab_size).float()
        input_in_vocab = Variable(torch.zeros(input.size()).fill_(dict.UNK)).cuda().long()
        input_in_vocab = mask.long() * input_in_vocab + (1 - mask).long() * input

        emb = self.embedding(input_in_vocab)
        output, state = self.rnn(emb, state)
        hidden, attn_weigths = self.attention(output, contexts)
        output = self.compute_score(hidden, num_oovs=num_oovs, src=src, attns=attn_weigths, states=state, x=emb)

        return output, state, attn_weigths
