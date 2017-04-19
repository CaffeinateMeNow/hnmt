from collections import Counter, namedtuple
from text import Encoded, TextEncoder, TwoThresholdTextEncoder 
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


class LogFreqEncoder(object):
    def __init__(self,
                 vocab=None,
                 use_boundaries=True):
        self.use_boundaries = use_boundaries

        if vocab is not None:
            # vocab should be a Counter or similar
            self.vocab = vocab
        else:
            self.vocab = Counter()

    def count(self, sequence):
        for lemma in sequence:
            self.vocab[lemma] += 1

    def done(self):
        # note that vocabulary is intentionally not truncated
        # this keeps actual frequencies of rare lemmas
        # but the softmax doesn't get any bigger
        mostfreq, _ = self.vocab.most_common(1)[0]
        self.max_val = self[mostfreq]

    def fields(self):
        return ('logf', len(self))

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

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
                sub_encoder=char_encoder),
            'logf': LogFreqEncoder(),
            'lemma': TextEncoder(max_vocab=self.max_lemma_vocab),
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
        return (
            ('surface', len(self.subencoders['surface'])),
            ('logf',    len(self.subencoders['logf'])),
            ('lemma',   len(self.subencoders['lemma'])),
            ('pos',     len(self.subencoders['pos'])),
            ('num',     len(self.subencoders['num'])),
            ('case',    len(self.subencoders['case'])),
            ('pers',    len(self.subencoders['pers'])),
            ('mood',    len(self.subencoders['mood'])),
            ('tense',   len(self.subencoders['tense'])),
            )

    def __str__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

    def __repr__(self):
        return str(self)

    def __getitem__(self, token):
        raise Exception('__getitem__ on compound encoder FinnposEncoder')

    def __len__(self):
        # length of main vocabulary
        return len(self.subencoders['surface'])

    def encode_sequence(self, fields, max_length=None, dtype=np.int32):
        surf = self.subencoders['surface'].encode_sequence(fields.surface,
             max_length=max_length, dtype=dtype)
        logf = self.subencoders['logf'].encode_sequence(fields.lemma,
             max_length=max_length, dtype=dtype)
        lemma = self.subencoders['lemma'].encode_sequence(fields.lemma,
             max_length=max_length, dtype=dtype)
        pos = self.subencoders['pos'].encode_sequence(fields.pos,
             max_length=max_length, dtype=dtype)
        num = self.subencoders['num'].encode_sequence(fields.num,
             max_length=max_length, dtype=dtype)
        case = self.subencoders['case'].encode_sequence(fields.case,
             max_length=max_length, dtype=dtype)
        pers = self.subencoders['pers'].encode_sequence(fields.pers,
             max_length=max_length, dtype=dtype)
        mood = self.subencoders['mood'].encode_sequence(fields.mood,
             max_length=max_length, dtype=dtype)
        tense = self.subencoders['tense'].encode_sequence(fields.tense,
             max_length=max_length, dtype=dtype)
        return Aux(surf, logf, lemma, pos, num, case, pers, mood, tense)

    def pad_sequences(self, encoded_sequences,
                      max_length=None, pad_right=True, dtype=np.int32):
        m, mask, char, char_mask = self.subencoders['surface'].pad_sequences(
            [x.surface for x in encoded_sequences],
            max_length=max_length, pad_right=pad_right, dtype=dtype)
        length = m.shape[0]
        n_batch = m.shape[1]
        out = Aux(*[
            np.zeros((length, n_batch), dtype=dtype)
            for _ in Aux._fields])

        for i,tpl in enumerate(encoded_sequences):
            for j,encoded in enumerate(tpl):
                encoded = encoded.sequence      # throw away empty unknowns
                if pad_right:
                    out[j][:len(encoded),i] = encoded
                else:
                    out[j][-len(encoded):,i] = encoded
        return m, mask, char, char_mask, out

    def decode_sentence(self, encoded):
        return (
            self.subencoders['surface'].decode_sentence(encoded.surface),
            self.subencoders['logf'].decode_sentence(encoded.lemma, no_boundary=True),
            self.subencoders['lemma'].decode_sentence(encoded.lemma),
            self.subencoders['pos'].decode_sentence(encoded.pos),
            self.subencoders['num'].decode_sentence(encoded.num),
            self.subencoders['case'].decode_sentence(encoded.case),
            self.subencoders['pers'].decode_sentence(encoded.pers),
            self.subencoders['mood'].decode_sentence(encoded.mood),
            self.subencoders['tense'].decode_sentence(encoded.tense),
        )
