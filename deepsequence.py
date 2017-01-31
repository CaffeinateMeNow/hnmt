"""
Deeply stacked sequences for bnas.
Can be both teacher-forced with scan, and single-stepped in beam search.
"""

from collections import namedtuple
import theano
from theano import tensor as T

from bnas import init
from bnas.model import Model, LSTM
from bnas.fun import function
from bnas.utils import expand_to_batch

Recurrence = namedtuple('Recurrence',
    ['variable', 'init', 'dropout'])

class Unit(Model):
    """Base class for recurrent units"""
    def __init__(self, name):
        super().__init__(name)
        # recurrent inputs/outputs
        self.recurrences = []
        # non-sequence inputs
        self.non_sequences = []

    def add_recurrence(self, var, init=None, dropout=0):
        self.recurrences.append(Recurrence(var, init, dropout))

    def add_non_sequence(self, var):
        self.non_sequences.append(var)

    @property
    def n_rec(self):
        return len(self.recurrences)

    @property
    def n_nonseq(self):
        return len(self.non_sequences)

    # subclasses should define step(out, unit_recs, unit_nonseqs) -> unit_recs

class DeepSequence(Model):
    """Recurrent sequence with one or more units"""
    def __init__(self, name, units, backwards=False, offset=0):
        super().__init__(name)
        self.units = units if units is not None else []
        self.backwards = backwards
        self.offset = offset
        self._step_fun = None

    def __call__(self, inputs, inputs_mask,
                 nontrainable_recurrent_inits, non_sequences=None):
        # combine trainable and nontrainable inits
        all_inits = []
        batch_size = inputs.shape[1]
        for rec in self.recurrences:
            if rec.init is None:
                # nontrainable init is passed in as argument
                all_inits.append(nontrainable_recurrent_inits.pop(0))
            else:
                # trainable inits must be expanded to batch size
                all_inits.append(
                    expand_to_batch(rec.init, batch_size))
            # FIXME: make dropout masks here
        seqs_in = [{'input': inputs, 'taps': [self.offset]},
                   {'input': inputs_mask, 'taps': [self.offset]}]
        # FIXME: add extra sequences, if needed
        non_sequences = self.non_sequences
        for unit in self.units:
            non_sequences.extend(unit.parameters_list())
            # FIXME: add dropout masks to nonseqs. Interleaved?
        seqs, _ = theano.scan(
                fn=self.step,
                go_backwards=self.backwards,
                sequences=seqs_in,
                outputs_info=all_inits,
                non_sequences=non_sequences)
        if self.backwards:
            return tuple(seq[::-1] for seq in seqs)
        else:
            return seqs

    def step(self, inputs, inputs_mask, *args):
        args_tail = list(args)
        grouped_rec = []
        grouped_nonseq = []
        recurrents_out = []
        out = inputs
        # group recurrents and non-sequences by unit
        # FIXME: separate leading extra sequences
        for unit in self.units:
            unit_rec, args_tail = \
                args_tail[:unit.n_rec], args_tail[unit.n_rec:]
            grouped_rec.append(unit_rec)
        for unit in self.units:
            unit_nonseq, args_tail = \
                args_tail[:unit.n_nonseq], args_tail[unit.n_nonseq:]
            grouped_nonseq.append(unit_rec)
        recurrents_in = [rec for unit_rec in grouped_rec for rec in unit_rec]
        # apply the units
        for (unit, unit_rec, unit_nonseq)  in zip(
                self.units, grouped_rec, grouped_nonseq):
            unit_recs_out = unit.step(out, unit_rec, unit_nonseq)
            # first recurrent output becomes new input
            out = unit_recs_out[0]
            recurrents_out.extend(unit_recs_out)
        # apply inputs mask to all recurrents
        inputs_mask_bcast = inputs_mask.dimshuffle(0, 'x')
        recurrents_out = [
            T.switch(inputs_mask_bcast, rec_out, rec_in)
            for (rec_out, rec_in) in zip(recurrents_out, recurrents_in)]
        # use recurrents_out[this.final_out_idx]
        return recurrents_out

    def step_fun(self):
        if self._step_fun is None:
            all_inputs = [T.matrix('inputs')]
            for unit in self.units:
                all_inputs.extend((
                    rec.var for rec in unit.recurrences))
            self._step_fun = function(
                all_inputs,
                self.step(*all_inputs),
                name='{}_step_fun'.format(self.name))
        return self._step_fun

    @property
    def final_out_idx(self):
        return -self.units[-1].n_rec

    @property
    def recurrences(self):
        return [rec for unit in self.units for rec in unit.recurrences]

    @property
    def non_sequences(self):
        return [nonseq for unit in self.units for nonseq in unit.non_sequences]


class LSTMUnit(Unit):
    def __init__(self, name, *args,
                 dropout=0, trainable_initial=False, **kwargs):
        super().__init__(name)
        self.add(LSTM('gate', *args, **kwargs))
        if trainable_initial:
            self.param('h_0', (self.gate.state_dims,),
                       init_f=init.Gaussian(fan_in=self.gate.state_dims))
            self.param('c_0', (self.gate.state_dims,),
                       init_f=init.Gaussian(fan_in=self.gate.state_dims))
            h_0 = self._h_0
            c_0 = self._c_0
        else:
            h_0 = None
            c_0 = None
        self.add_recurrence(T.matrix('h_tm1'), init=h_0, dropout=dropout)
        self.add_recurrence(T.matrix('c_tm1'), init=c_0, dropout=0)
        if self.gate.use_attention:
            # attention output
            self.add_recurrence(None, init=None, dropout=0)
            self.add_non_sequence(T.tensor3('attended'))
            self.add_non_sequence(T.tensor3('attended_dot_u'))
            self.add_non_sequence(T.matrix('attention_mask'))

    def step(self, out, unit_recs, unit_nonseqs):
        unit_recs = self.gate(out, *(unit_recs + unit_nonseqs))
        return unit_recs


class ResidualUnit(Unit):
    """Wraps another Unit"""
    pass
