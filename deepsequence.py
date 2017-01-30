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

    # subclasses should define step

class DeepSequence(Model):
    """Recurrent sequence with one or more units"""
    def __init__(self, name, units):
        super().__init__(name)
        self.units = units if units is not None else []

    def __call__(self, inputs, inputs_mask,
                 nontrainable_recurrent_inits, non_sequences=None):
        pass

    def step(self, inputs, inputs_mask, recurrents_in, non_sequences=None):
        pass

    def step_fun(self):
        pass


class LSTMUnit(Unit):
    pass
