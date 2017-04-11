"""Search algorithms for recurrent networks."""

from collections import namedtuple

import itertools
import numpy as np
import theano

Hypothesis = namedtuple(
    'Hypothesis',
    ['sentence',    # index of sentence in minibatch
     'score',       # raw score
     'norm_score',  # score adjusted by penalties
     'history',     # sequence up to last symbol
     'last_sym',    # last symbol
     'states',      # RNN state
     'coverage',    # accumulated coverage
     'unks',        # states at UNK symbols
     'aux'])        # states for aux task

def by_sentence(beams):
    return itertools.groupby(
        sorted(beams,
               key=lambda hyp: (hyp.sentence, -hyp.norm_score, -hyp.score)),
        lambda hyp: hyp.sentence)

def beam_with_coverage(
        step,
        states0,
        batch_size,
        start_symbol,
        stop_symbol,
        unk_symbol,
        max_length,
        inputs_mask,
        beam_size=8,
        min_length=0,
        alpha=0.2,
        beta=0.2,
        len_smooth=5.0,
        prune_margin=3.0,
        keep_unk_states=True,
        keep_aux_states=False):
    """Beam search algorithm.

    See the documentation for :meth:`greedy()`.
    The additional arguments are FIXME
    prune_margin is misleadingly named beamsize in Wu et al 2016

    Returns
    -------
    outputs : numpy.ndarray(int64)
        Array of shape ``(n_beams, length, batch_size)`` with the output
        sequences. `n_beams` is less than or equal to `beam_size`.
    outputs_mask : numpy.ndarray(theano.config.floatX)
        Array of shape ``(n_beams, length, batch_size)``, containing the
        mask for `outputs`.
    scores : numpy.ndarray(float64)
        Array of shape ``(n_beams, batch_size)``.
        Log-probability of the sequences in `outputs`.
    """

    beams = [Hypothesis(i, 0., -1e30, (), start_symbol,
                        [[s[i, :] for s in ms] for ms in states0],
                        1e-30, (), ())
             for i in range(batch_size)]

    for i in range(max_length-2):
        # build step inputs
        active = [hyp for hyp in beams if hyp.last_sym != stop_symbol]
        completed = [hyp for hyp in beams if hyp.last_sym == stop_symbol]
        if len(active) == 0:
            return by_sentence(beams)

        states = []
        prev_syms = np.zeros((1, len(active)), dtype=np.int64)
        mask = np.ones((len(active),), dtype=theano.config.floatX)
        sent_indices = np.zeros((len(active),), dtype=np.int64)
        for (j, hyp) in enumerate(active):
            states.append(hyp.states)
            prev_syms[0, j] = hyp.last_sym
            sent_indices[j] = hyp.sentence
        # for each state of each model, concatenate hypotheses
        states = [[np.array(x) for x in zip(*y)]
                  for y in zip(*states)]

        # predict
        all_states, all_dists, attention, all_unks = step(
            i, states, prev_syms, mask, sent_indices)
        if i <= min_length:
            all_dists[:, stop_symbol] = 1e-30
        all_dists = np.log(all_dists)
        n_symbols = all_dists.shape[-1]

        # preprune symbols
        # using beam_size+1, because score of stop_symbol
        # may still become worse
        best_symbols = np.argsort(all_dists, axis=1)[:, -(beam_size+1):]

        # extend active hypotheses
        extended = []
        for (j, hyp) in enumerate(active):
            history = hyp.history + (hyp.last_sym,)
            for symbol in best_symbols[j, :]:
                score = hyp.score + all_dists[j, symbol]
                norm_score = -1e30
                # attention: (batch, source_pos)
                if attention is not None:
                    coverage = hyp.coverage + attention[j, :]
                else:
                    coverage = None
                if symbol == stop_symbol:
                    # length penalty
                    # (history contains start symbol but not stop symbol)
                    if alpha > 0:
                        lp = (((len_smooth + len(history) - 1.) ** alpha)
                            / ((len_smooth + 1.) ** alpha))
                    else:
                        lp = 1
                    # coverage penalty
                    # apply mask: adding 1 to masked elements removes penalty
                    if beta > 0 and coverage is not None:
                        coverage += (1. - inputs_mask[:, hyp.sentence])
                        cp = beta * np.sum(np.log(
                            np.minimum(coverage, np.ones_like(coverage))))
                    else:
                        cp = 0
                    norm_score = (score / lp) + cp
                new_states = [[s[j, :] for s in ms] for ms in all_states]
                if keep_unk_states and symbol == unk_symbol:
                    new_unks = tuple(unk[j, :] for unk in all_unks)
                    unks = hyp.unks + (new_unks,)
                else:
                    unks = hyp.unks
                if keep_aux_states:
                    # only for monitoring: doesn't do ensemble
                    # FIXME: assumes word-decoder is single-layer
                    h = new_states[0][0]
                    h_breve = all_unks[0][j, :]
                    new_aux = np.concatenate([h, h_breve], axis=-1).astype(
                        dtype=theano.config.floatX)
                    aux = hyp.aux + (new_aux,)
                else:
                    aux = hyp.aux
                extended.append(
                    Hypothesis(hyp.sentence,
                               score,
                               norm_score,
                               history,
                               symbol,
                               new_states,
                               coverage,
                               unks,
                               aux))

        # prune hypotheses
        beams = []
        for (_, group) in by_sentence(completed + extended):
            group = list(group)
            best_normalized = max(hyp.norm_score for hyp in group)
            group = [hyp for hyp in group
                     if hyp.last_sym != stop_symbol
                        or hyp.norm_score > best_normalized - prune_margin]
            beams.extend(sorted(group, key=lambda hyp: -hyp.score)[:beam_size])
    return by_sentence(beams)
