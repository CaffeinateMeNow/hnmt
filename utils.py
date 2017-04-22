import collections
import itertools
import gzip

try:
    from nltk import word_tokenize, wordpunct_tokenize
except ImportError:
    print('Cannot load nltk tokenizer')

def safe_zip(*iterables):
    iters = [iter(x) for x in iterables]
    sentinel = object()
    for (j, tpl) in enumerate(itertools.zip_longest(*iterables, fillvalue=sentinel)):
        for (i, val) in enumerate(tpl):
            if val is sentinel:
                raise ValueError('Column {} was too short. '
                    'Row {} (and later) missing.'.format(i, j))
        yield tpl

def open_file(filename):
    if filename.endswith('.gz'):
        def open_func(fname):
            return gzip.open(fname, 'rt', encoding='utf-8')
    else:
        def open_func(fname):
            return open(fname, 'r', encoding='utf-8')
    with open_func(filename) as fobj:
        for line in fobj:
            yield line

# data with no aux fields
Surface = collections.namedtuple('Surface', ['surface'])

def tokenize(filename, tokenizer, lower):
    def reader():
        def process(line):
            if lower: line = line.lower()
            if tokenizer == 'char': return line.strip()
            elif tokenizer == 'space': return line.split()
            return word_tokenize(line)
        return [Surface(process(x)) for x in open_file(filename)]
    return reader
