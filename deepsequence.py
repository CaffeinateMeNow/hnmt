"""
Deeply stacked sequences for bnas.
Can be both teacher-forced with scan, and single-stepped in beam search.
"""

from bnas.model import Model

class Unit(Model):
    """Base class for recurrent units"""
    def __init__(self, name):
        super().__init__(name)
        # recurrent inputs/outputs
        self.nontrainable_recurrents = []
        self.trainable_recurrents = []
        # non-sequence inputs
        self.non_sequences = []

    def n_rec(self):
        return (len(self.nontrainable_recurrents)
            + len(self.trainable_recurrents))

    def n_nonseq(self):
        return len(self.non_sequences)

    # subclasses should define step

class DeepSequence(Model):
    """Recurrent sequence with one or more units"""
    def __init__(self, name, units):
        super().__init__(name)
        self.units = units if units is not None else []
        self._step_fun = None

    def __call__(self, inputs, inputs_mask,
                 nontrainable_recurrent_inits, non_sequences=None):
        # FIXME: combine expanded trainable inits with nontrainable inits
        pass

    def step(self, inputs, inputs_mask, recurrents_in, non_sequences=None):
        if non_sequences is None:
            non_sequences = []
        recurrents_tail = list(recurrents_in)
        recurrents_out = []
        out = inputs
        for unit in self.units:
            # group recurrents and non-sequences by unit
            unit_rec, recurrents_tail = \
                recurrents_tail[:unit.n_rec], recurrents_tail[unit.n_rec:]
            unit_nonseq, non_sequences = \
                non_sequences[:unit.n_nonseq], non_sequences[unit.n_nonseq:]
            unit_recs_out = unit.step(out, unit_rec, unit_nonseq)
            # first recurrent output becomes new input
            out = unit_recs_out[0]
            recurrents_out.extend(unit_recs_out)
        if len(recurrents_tail) != 0:
            raise Exception(
                '{} unused recurrent inits'.format(len(recurrents_tail))
        if len(non_sequences) != 0:
            raise Exception(
                '{} unused non-sequences'.format(len(non_sequences))
        if len(recurrents_out) != len(recurrents_in):
            raise Exception(
                '{} recurrent inits != {} recurrents out'.format(
                    len(recurrents_out), len(recurrents_in)))
        # apply inputs mask to all recurrents
        inputs_mask_bcast = inputs_mask.dimshuffle(0, 'x')
        recurrents_out = [
            T.switch(inputs_mask_bcast, rec_out, rec_in)
            for (rec_out, rec_in) in zip(recurrents_out, recurrents_in)]
        # use recurrents_out[this.final_out_idx()]
        return recurrents_out

    def step_fun(self):
        if self._step_fun is None:
            all_inputs = [T.matrix('inputs')]
            for unit in self.units:
                all_inputs.extend(unit.nontrainable_recurrents) # FIXME: all recurrents, but in what order?
            self._step_fun = function(
                all_inputs,
                self.step(*all_inputs)
                name='{}_step_fun'.format(self.name))
        return self._step_fun

    def final_out_idx(self):
        return -self.units[-1].n_rec


class LSTMUnit(Unit):
    pass

class ResidualUnit(Unit):
    """Wraps another unit"""
    pass
