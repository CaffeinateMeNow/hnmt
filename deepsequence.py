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
NonSequence = namedtuple('NonSequence',
    ['variable', 'func', 'idx'])
OutputOnly = object()

class Unit(Model):
    """Base class for recurrent units"""
    def __init__(self, name):
        super().__init__(name)
        # recurrent inputs/outputs
        self._recurrences = []
        # non-sequence inputs
        self._non_sequences = []

    def add_recurrence(self, var, init=None, dropout=0):
        """params:
            var -- Theano variable
            init -- 1) None if nontrainable init will be passed as input
                    2) parameter if a trainable init
                 or 3) OutputOnly if no init is needed
            dropout -- float in interval [0,1)
        """
        self._recurrences.append(Recurrence(var, init, dropout))

    def add_non_sequence(self, var, func=None, idx=None):
        """params:
            var -- Theano variable
            func -- 1) None if passed in as input
                 or 2) function(var) -> var
                       if precomputed from another input
            idx -- index of input non-sequences to give
                   as argument to func
        """
        if func is not None:
            assert idx is not None
        self._non_sequences.append(NonSequence(var, func, idx))

    @property
    def recurrences(self):
        return tuple(self._recurrences)

    @property
    def non_sequences(self):
        return tuple(self._non_sequences)

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
                 nontrainable_recurrent_inits=None, non_sequences=None):
        # combine trainable and nontrainable inits
        inits_in = []
        batch_size = inputs.shape[1]
        for rec in self.recurrences:
            if rec.init is None:
                # nontrainable init is passed in as argument
                inits_in.append(nontrainable_recurrent_inits.pop(0))
            elif rec.init == OutputOnly:
                # no init needed
                inits_in.append(None)
            else:
                # trainable inits must be expanded to batch size
                inits_in.append(
                    expand_to_batch(rec.init, batch_size))
            # FIXME: make dropout masks here
        seqs_in = [{'input': inputs, 'taps': [self.offset]},
                   {'input': inputs_mask, 'taps': [self.offset]}]
        # FIXME: add extra sequences, if needed
        non_sequences_in = self.make_nonsequences(non_sequences)
        seqs, _ = theano.scan(
                fn=self.step,
                go_backwards=self.backwards,
                sequences=seqs_in,
                outputs_info=inits_in,
                non_sequences=non_sequences_in)
        if self.backwards:
            seqs = tuple(seq[::-1] for seq in seqs)
        ## group outputs in a useful way
        # main output of final unit
        final_out = seqs[self.final_out_idx]
        # true recurrent states (inputs for next iteration)
        states = []
        # OutputOnly are not fed into next iteration
        outputs = []
        for (rec, seq) in zip(self.recurrences, seqs):
            if rec.init == OutputOnly:
                outputs.append(seq)
            else:
                states.append(seq)
        return final_out, states, outputs

    def make_nonsequences(self, non_sequences):
        non_sequences_in = []
        if non_sequences is None:
            non_sequences = []
        for unit in self.units:
            # previous units already popped off
            unit_nonseq = list(non_sequences)
            for ns in unit.non_sequences:
                if ns.func is not None:
                    non_sequences_in.append(ns.func(unit_nonseq[ns.idx]))
                else:
                    non_sequences_in.append(non_sequences.pop(0))
        for unit in self.units:
            non_sequences_in.extend(unit.parameters_list())
            # FIXME: add dropout masks to nonseqs. Interleaved?
        return non_sequences_in

    def step(self, inputs, inputs_mask, *args):
        args_tail = list(args)
        grouped_rec = []
        grouped_nonseq = []
        recurrents_in = []
        recurrents_out = []
        out = inputs
        # group recurrents and non-sequences by unit
        # FIXME: separate leading extra sequences
        for unit in self.units:
            unit_rec = []
            for rec in unit.recurrences:
                if rec.init == OutputOnly:
                    # output only: scan has removed the None
                    recurrents_in.append(OutputOnly)
                else:
                    rec_in = args_tail.pop(0)
                    unit_rec.append(rec_in)
                    recurrents_in.append(rec_in)
            grouped_rec.append(unit_rec)
        for unit in self.units:
            unit_nonseq, args_tail = \
                args_tail[:unit.n_nonseq], args_tail[unit.n_nonseq:]
            grouped_nonseq.append(unit_nonseq)
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
            if rec_in is not OutputOnly else rec_out
            for (rec_out, rec_in) in zip(recurrents_out, recurrents_in)]
        # you probably only care about recurrents_out[this.final_out_idx]
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
            self.add_recurrence(
                T.matrix('attention'), init=OutputOnly, dropout=0)

            self.add_non_sequence(T.tensor3('attended'))
            # precomputed from attended
            self.add_non_sequence(T.tensor3('attended_dot_u'),
                func=self.gate.attention_u, idx=0)
            self.add_non_sequence(T.matrix('attention_mask'))

    def step(self, out, unit_recs, unit_nonseqs):
        unit_recs = self.gate(out, *(unit_recs + unit_nonseqs))
        return unit_recs


class ResidualUnit(Unit):
    """Wraps another Unit"""
    def __init__(self, wrapped, var=None):
        super().__init__('residual_using_{}'.format(wrapped.name))
        self.wrapped = wrapped
        var = var if var is not None else T.matrix('residual')
        self.residual = Recurrence(var, OutputOnly, dropout=0)

    def step(self, out, unit_recs, unit_nonseqs):
        unit_recs = self.wrapped.step(out, unit_recs, unit_nonseqs)
        out += unit_recs[0]     # add residual
        return (out,) + unit_recs

    @property
    def recurrences(self):
        return (self.residual,) + self.wrapped.recurrences

    @property
    def non_sequences(self):
        return self.wrapped.non_sequences
