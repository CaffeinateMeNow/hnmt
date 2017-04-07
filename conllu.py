"""Text processing.

The :class:`TextEncoder` class is the main feature of this module, the helper
functions were used in earlier examples and should be phased out.
"""

from collections import Counter, namedtuple

import numpy as np
import theano

MiniBatch = namedtuple('MiniBatch',
    ['src',             # (tokens, mask, chars, charmask)
     'tgt',             # (tokens, mask, chars, charmask)
     'aux',             # (aux0, aux1, ...) or None
    ])
# each of the fields contains a sequence with one element per token
# FIXME: split morph tags into multiple fields? lang-specific
#   Finnish: Number, Case, Person, Mood, Tense, Misc=(Card/Ord/Post/Prep/Foreign/Abbr/AbbrNum)
# FIXME: more options: is_compound? 
Conllu = namedtuple('Conllu',
    ['surface', 'lemma', 'upos', 'morph', 'head', 'deplbl'])
Aux = namedtuple('Conllu',
    ['surface', 'logf', 'lemma', 'upos', 'morph', 'head', 'deplbl'])
        
def conllu_helper(split):
    surface = []
    for line in split:
        surface.append(line[1])
    columns = list(zip(*split))
    lemmas = columns[2]
    upos = columns[3]
    morphs = columns[5]     # as a single, pipe-separated string
    heads = []           
    for i in columns[6]:
        i = int(i) 
        if i > 0:
            # lemma at i (CoNLL-U uses 1-based indexing)
            head = columns[2][i-1]
            # only last part of compound
            head = head.split('#')[-1]
            heads.append(head)
        else:
            heads.append('#root')
    deplbl = columns[7]
    return Conllu(surface, lemmas, upos, morphs, heads, deplbl)
            
def read_conllu(lines):
    raw = []    
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            # empty lines indicate sentence breaks
            yield conllu_helper(raw)
            raw = []
            continue
        if line[0] == '#':
            # comments start with hash
            continue
        raw.append(line.split('\t'))

def pad_aux(encoded_sequences, length,
            pad_right=True, dtype=np.int32):
    """
    arguments:
        encoded_sequences -- a list of Aux tuples
    """
    if not encoded_sequences:
        # An empty matrix would mess up things, so create a dummy 1x1
        # matrix with an empty mask in case the sequence list is empty.
        m = np.zeros((1 if max_length is None else max_length, 1),
                        dtype=dtype)
        return m

    out = Aux(*[
        np.zeros((length, len(encoded_sequences)), dtype=dtype)
        for _ in encoded_sequences._fields])

    for i,tpl in enumerate(encoded_sequences):
        for j,field in enumerate(tpl):
            if pad_right:
                out[j][:len(encoded),i] = encoded
            else:
                out[j]m[-len(encoded):,i] = encoded
    return out


class LogFreqEncoder(object):
    def __init__(self,
                 vocab=None,
                 sequences=None,
                 use_boundaries=True):
        self.use_boundaries = use_boundaries

        if vocab is not None:
            self.vocab = vocab
        else:
            if sequences is not None:
                self.vocab = Counter(x for xs in sequences for x in xs)

    def __str__(self):
        return 'LogFreqEncoder(%d)' % len(self)

    def __repr__(self):
        return str(self)

    def __getitem__(self, x):
        return int(np.log(self.vocab[token] + 1))

    def __len__(self):
        return len(self.vocab)

    def encode_sequence(self, sequence, max_length=None, unknowns=None):
        start = (0,) if self.use_boundaries else ()
        stop = (0,) if self.use_boundaries else ()
        encoded = tuple(self[symbol] for symbol in sequence)
        if max_length is None \
        or len(encoded)+len(start)+len(stop) <= max_length:
            return start + encoded + stop
        else:
            return start + encoded[:max_length-(len(start)+len(stop))] + stop

    def pad_sequences(self, sequences,
                      max_length=None, pad_right=True, dtype=np.int32):
        if not sequences:
            # An empty matrix would mess up things, so create a dummy 1x1
            # matrix with an empty mask in case the sequence list is empty.
            m = np.zeros((1 if max_length is None else max_length, 1),
                         dtype=dtype)
            mask = np.zeros_like(m, dtype=np.bool)
            return m, mask
        encoded_sequences = [
                self.encode_sequence(sequence, max_length)
                for sequence in sequences]
        length = max(map(len, encoded_sequences))
        length = length if max_length is None else min(length, max_length)

        m = np.zeros((length, len(sequences)), dtype=dtype)
        mask = np.zeros_like(m, dtype=np.bool)

        for i,encoded in enumerate(encoded_sequences):
            if pad_right:
                m[:len(encoded),i] = encoded
                mask[:len(encoded),i] = 1
            else:
                m[-len(encoded):,i] = encoded
                mask[-len(encoded):,i] = 1

        return m, mask
