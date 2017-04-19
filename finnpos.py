from collections import Counter, namedtuple
from text import Encoded
from utils import *

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
    ['surface', 'lemma', 'pos', 'num', 'case', 'pers', 'mood', 'tense'])
Aux = namedtuple('Aux',
    ['surface', 'logf', 'lemma', 'pos', 'num', 'case', 'pers', 'mood', 'tense'])

# julistan    _   julistaa    [POS=VERB]|[VOICE=ACT]|[MOOD=INDV]|[TENSE=PRESENT]|[PERS=SG1]   _   
        
def finnpos_helper(split):
    columns = list(zip(*split))
    surface = columns[0]
    lemmas = columns[2]
    pos = columns[3]
    num = columns[4]
    case = columns[5]
    pers = columns[6]
    mood = columns[7]
    tense = columns[8]
    return Finnpos(surface, lemmas, pos, num, case, pers, mood, tense)
            
def finnpos_reader(filename):
    def reader():
        raw = []    
        for line in open_file(filename):
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
    return reader

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
            # vocab should be a Counter or similar
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


class FinnposEncoder(object):
    def __init__(self,
                 max_vocab=None,
                 max_lemma_vocab=None,
                 min_char_count=0,
                 overlap=0):
        self.max_vocab = max_vocab
        self.max_lemma_vocab = max_lemma_vocab
        self.overlap = overlap
        char_encoder = TextEncoder(min_count=min_char_count)
        self.subencoders = {
            'surface': TwoThresholdTextEncoder(
                max_vocab=max_vocab,
                overlap=overlap,
                sub_encoder=char_encoder)
            'logf': LogFreqEncoder(),
            'lemma': TextEncoder(max_vocab=args.lemma_vocabulary),
            'pos': TextEncoder(),
            'num': TextEncoder(),
            'case': TextEncoder(),
            'pers': TextEncoder(),
            'mood': TextEncoder(),
            'tense': TextEncoder()}

    def count(self, tpl):
        self.subencoders['surface'].count(tpl.surface)
        self.subencoders['logf'].count(tpl.lemma)
        self.subencoders['lemma'].count(tpl.lemma)
        self.subencoders['pos'].count(tpl.pos)
        self.subencoders['num'].count(tpl.num)
        self.subencoders['case'].count(tpl.case)
        self.subencoders['pers'].count(tpl.pers)
        self.subencoders['mood'].count(tpl.mood)
        self.subencoders['tense'].count(tpl.tense)

    def done(self):
        for subenc in self.subencoders.values():
            subenc.done()

    def fields(self):
        ('surface', len(self.subencoders['surface'])),
        ('logf',    len(self.subencoders['logf'])),
        ('lemma',   len(self.subencoders['lemma'])),
        ('pos',     len(self.subencoders['pos'])),
        ('num',     len(self.subencoders['num'])),
        ('case',    len(self.subencoders['case'])),
        ('pers',    len(self.subencoders['pers'])),
        ('mood',    len(self.subencoders['mood'])),
        ('tense',   len(self.subencoders['tense'])),

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

    def __repr__(self):
        return str(self)

    def __getitem__(self, token):
        raise Exception('__getitem__ on compound encoder FinnposEncoder')

    def __len__(self):
        # length of main vocabulary
        return len(self.subencoders['surface'])

# FIXME: WIP
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
