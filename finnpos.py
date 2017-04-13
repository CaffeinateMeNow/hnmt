from collections import Counter, namedtuple
from text import Encoded

import numpy as np
import theano

MiniBatch = namedtuple('MiniBatch',
    ['src',             # (tokens, mask, chars, charmask)
     'tgt',             # (tokens, mask, chars, charmask)
     'aux',             # (aux0, aux1, ...) or None
    ])
# each of the fields contains a sequence with one element per token
# the main 'sequence' is the sequence form
# FIXME: split morph tags into multiple fields? lang-specific
#   Finnish: Number, Case, Person, Mood, Tense
Finnpos = namedtuple('Finnpos',
    ['sequence', 'lemma', 'pos', 'num', 'case', 'pers', 'mood', 'tense'])
Aux = namedtuple('Aux',
    ['sequence', 'logf', 'lemma', 'pos', 'num', 'case', 'pers', 'mood', 'tense'])

# julistan    _   julistaa    [POS=VERB]|[VOICE=ACT]|[MOOD=INDV]|[TENSE=PRESENT]|[PERS=SG1]   _   
        
def finnpos_helper(split):
    columns = list(zip(*split))
    sequence = columns[0]
    lemmas = columns[2]
    combo_morphs = columns[3]     # as a single, pipe-separated string
    pos = []
    num = []
    case = []
    pers = []
    mood = []
    tense = []
    for morphs in combo_morphs:
        morphs = morphs.replace('[', '').replace(']', '')
        morphs = morphs.split('|')
        morphs = dict(pair.split('=') for pair in morphs
                      if '=' in pair)
        pos.append(morphs.get('POS', 'UNKNOWN'))    # should already be UNKNOWN
        num.append(morphs.get('NUM', '<NONE>'))
        case.append(morphs.get('CASE', '<NONE>'))
        pers.append(morphs.get('PERS', '<NONE>'))
        mood.append(morphs.get('MOOD', '<NONE>'))
        tense.append(morphs.get('TENSE', '<NONE>'))
    return Finnpos(sequence, lemmas, pos, num, case, pers, mood, tense)
            
def read_finnpos(lines):
    raw = []    
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            # empty lines indicate sentence breaks
            yield finnpos_helper(raw)
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
    n_batch = max(1, len(encoded_sequences))
    out = Aux(*[
        np.zeros((length, n_batch), dtype=dtype)
        for _ in Aux._fields])

    if not encoded_sequences:
        # An empty matrix would mess up things, so create a dummy 1x1
        # matrix with an empty mask in case the sequence list is empty.
        return out

    for i,tpl in enumerate(encoded_sequences):
        for j,encoded in enumerate(tpl):
            encoded = encoded.sequence      # throw away empty unknowns
            if pad_right:
                out[j][:len(encoded),i] = encoded
            else:
                out[j][-len(encoded):,i] = encoded
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
        mostfreq, _ = self.vocab.most_common(1)[0]
        self.max_val = self[mostfreq]

    def __str__(self):
        return 'LogFreqEncoder(%d)' % len(self)

    def __repr__(self):
        return str(self)

    def __getitem__(self, token):
        return int(np.log(self.vocab[token] + 1))

    def __len__(self):
        return self.max_val + 1

    def encode_sequence(self, sequence, max_length=None, dtype=np.int32):
        start = (0,) if self.use_boundaries else ()
        stop = (0,) if self.use_boundaries else ()
        encoded = tuple(self[symbol] for symbol in sequence)
        if max_length is None \
        or len(encoded)+len(start)+len(stop) <= max_length:
            out = start + encoded + stop
        else:
            out = start + encoded[:max_length-(len(start)+len(stop))] + stop
        return Encoded(np.asarray(out, dtype=dtype), None)

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

    def decode_sentence(self, encoded, no_boundary=False):
        seq = encoded.sequence
        if self.use_boundaries and not no_boundary:
            seq = seq[1:]
        #return [str(np.exp(x) - 1) for x in seq]
        return [str(x) for x in seq]

