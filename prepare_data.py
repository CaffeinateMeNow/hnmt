import collections
import itertools
import numpy as np
import random

LineLengths = collections.namedtuple('LineLengths',
    ['idx', 'src_len', 'tgt_len'])
LineStatistics = collections.namedtuple('LineStatistics',
    ['idx', 'shard', 'src_len', 'tgt_len', 'n_unk', 'group'])

class ShardedData(object):
    def __init__(self,
                 src_lines,
                 tgt_lines,
                 src_encoders,
                 tgt_encoders,
                 src_max_len=600,
                 tgt_max_len=600,
                 min_lines_per_group=128,
                 max_lines_per_shard=1000000,
                 min_saved_padding=2048):
        # callables, yielding tokenized lines
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
        # list of functions: [type] -> encoder
        self.src_encoders = src_encoders
        self.tgt_encoders = tgt_encoders
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        self.min_lines_per_group = min_lines_per_group
        self.max_lines_per_shard = max_lines_per_shard
        self.min_saved_padding = min_saved_padding
        # first LineLengths, later LineStatistics
        self.line_stats = []
        self.src_token_counts = [collections.counter()
                                 for _ in self.src_encoders]
        self.tgt_token_counts = [collections.counter()
                                 for _ in self.tgt_encoders]
        self.n_shards = None
        self.shard_indices = None

    def collect_statistics(self):
        # first pass
        for (i, (src, tgt)) in enumerate(safe_zip(self.src_lines(),
                                                  self.tgt_lines())):
            src_len = len(src.surface)
            tgt_len = len(tgt.surface)
            # filter out too long lines
            if src_len > self.src_max_len:
                continue
            if tgt_len > self.tgt_max_len:
                continue
            # total line count => shard sizes/num
            # length distribution => thresholds for padding groups
            self.line_stats.append(LineLengths(i, src_len, tgt_len))
            # token counts => vocabulary index (encoder)
            for (field, counter) in zip(src, self.src_token_counts):
                counter.update(field)
            for (field, counter) in zip(tgt, self.tgt_token_counts):
                counter.update(field)
        # preassign sentences to shards by random draw without replacement
        self.n_shards = int(np.ceil(len(self.line_stats) / self.max_lines_per_shard))
        lines_per_shard = int(np.ceil(len(self.line_stats) / self.n_shards))
        self.shard_indices = [j for i in range(self.n_shards) for j in [i] * lines_per_shard]
        random.shuffle(self.shard_indices)
        #- choose thresholds for padding groups
        #    - sort lengths
        #    - calculate cumulative padding waste function
        #    - select threshold to maximize saving
        #        - mid * (len[end] - len[mid])
        #    - split src/tgt alternatingly, while enough samples and big enough saving

    def encode(self):
        #- second pass
        #    - one pass per shard
        #        - choose padding group by lengths
        #        - encode, pad and concatenate
        #        - also track number of unks

    def prepare_data(self):
        self.collect_statistics()
        self.encode()


def safe_zip(*iterables):
    iters = [iter(x) for x in iterables]
    sentinel = object()
    for (j, tpl) in enumerate(itertools.zip_longest(*iterables, fillvalue=sentinel)):
        for (i, val) in enumerate(tpl):
            if val is sentinel:
                raise ValueError('Column {} was too short. '
                    'Row {} (and later) missing.'.format(i, j))
        yield tpl
