"""Search algorithms for recurrent networks."""

from collections import namedtuple

import itertools
import heapq
import numpy as np
import theano

# heap of completed hypotheses sorted by norm_score
Hypothesis = namedtuple(
    'Hypothesis',
    ['norm_score',  # final score adjusted by penalties
     'sentence',    # index of sentence in minibatch
     'score',       # raw score
     'prune_score', # score used for pruning
     'history',     # sequence up to last symbol
     'last_sym',    # last symbol
     'states',      # RNN state
     'coverage',    # accumulated coverage
     'unks',        # states at UNK symbols
     'aux'])        # states for aux task

def by_sentence(beams):
    return itertools.groupby(
        sorted(beams,
               key=lambda hyp: hyp.sentence),
        lambda hyp: hyp.sentence)

def sort_out(completed):
    return [sorted(sentence,
                   key=lambda hyp: (-hyp.norm_score, -hyp.score))
            for sentence in completed]

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
        alpha=0.01,
        beta=0.4,
        gamma=0.0,
        len_smooth=5.0,
        prune_mult=1.0,
        n_best=None,
        keep_unk_states=True,
        keep_aux_states=False):
    """Beam search algorithm.

    See the documentation for :meth:`greedy()`.
    The additional arguments are FIXME

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

    # coverage for masked-out timesteps is initialized to 1, to remove penalty
    if inputs_mask is not None:
        coverages = [1e-30 + 1. - inputs_mask[:, i]
                     for i in range(batch_size)]
    else:
        coverages = [None] * batch_size
    beams = [Hypothesis(-1e30, i, 0., 0., (), start_symbol,
                        [[s[i, :] for s in ms] for ms in states0],
                        coverages[i], (), ())
             for i in range(batch_size)]

    if n_best is None:
        n_best = batch_size
    completed = [list() for i in range(batch_size)]
    best_normalized = [-1e30] * batch_size 
    for i in range(max_length-2):
        # build step inputs
        active = []
        for hyp in beams:
            if hyp.last_sym != stop_symbol:
                active.append(hyp)
            else:
                heapq.heappush(completed[hyp.sentence], hyp)
                best_normalized[hyp.sentence] = max(
                    hyp.norm_score, best_normalized[hyp.sentence])
                if len(completed[hyp.sentence]) > n_best:
                    # prunes smallest (i.e. worst) completed hyp
                    heapq.heappop(completed[hyp.sentence])
        if len(active) == 0:
            return sort_out(completed), i

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
                # overattending penalty
                if gamma > 0 and coverage is not None:
                    oap = gamma * -max(0, np.max(coverage) - 1.)
                else:
                    oap = 0
                if symbol == stop_symbol:
                    # length penalty
                    # (history contains start symbol but not stop symbol)
                    if alpha > 0:
                        lp = (((len_smooth + len(history) - 1.) ** alpha)
                            / ((len_smooth + 1.) ** alpha))
                    else:
                        lp = 1
                    # coverage penalty
                    if beta > 0 and coverage is not None:
                        cp = beta * np.sum(np.log(
                            np.minimum(coverage, np.ones_like(coverage))))
                    else:
                        cp = 0
                    norm_score = (score / lp) + cp + oap
                # both score and overattending penalty are monotonically worsening
                prune_score = score + oap
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
                    Hypothesis(norm_score,
                               hyp.sentence,
                               score,
                               prune_score,
                               history,
                               symbol,
                               new_states,
                               coverage,
                               unks,
                               aux))

        # prune active hypotheses
        # this heuristic can prune out winning hypotheses,
        # as length penalty and coverage penalty can keep on improving
        def keep(hyp, best_normalized):
            return hyp.prune_score > (best_normalized * prune_mult)
        beams = []
        for (sent, group) in by_sentence(extended):
            group = list(group)
            group = [hyp for hyp in group if keep(hyp, best_normalized[sent])]
            beams.extend(sorted(group, key=lambda hyp: -hyp.score)[:beam_size])
    # force-terminate actives, if needed
    for sent in range(batch_size):
        if len(completed[sent]) == 0:
            completed[sent] = [hyp for hyp in beams if hyp.sentence == sent]
    return sort_out(completed), max_length - 1
