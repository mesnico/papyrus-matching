import pandas as pd

from pos_real import PositiveRealMatchDataset
from pos_synth import PositiveSyntheticMatchDataset
from neg import NegativeMatchDataset
from neg_synth import NegativeSyntheticMatchDataset

data = []
root = 'data/unified'

for split in ['train', 'val']:

    with open(f'{root}/{split}.txt', 'r') as f:
        ids = f.read().splitlines()

    dset = PositiveRealMatchDataset(root, ids)
    data.append({'kind': 'real_pos', 'pad': None, 'split': split, 'samples': len(dset)})

    for pad in (20,): # (0, 5, 10, 15):
        dset = PositiveSyntheticMatchDataset(root, ids, pad=pad)
        data.append({'kind': 'synth_pos', 'pad': pad, 'split': split, 'samples': len(dset)})

        dset = NegativeMatchDataset(root, ids, pad=pad)
        data.append({'kind': 'neg', 'pad': pad, 'split': split, 'samples': len(dset)})

        dset = NegativeSyntheticMatchDataset(root, ids, pad=pad)
        data.append({'kind': 'synth_neg', 'pad': pad, 'split': split, 'samples': len(dset)})

df = pd.DataFrame(data)
df.to_csv('dataset_stats.csv', index=False)

# Print summary, root and kind on columns, pad on rows
# include pad=None for real_pos as well, repeating the value for all pads
summary = df.pivot_table(index='pad', columns=['split', 'kind'], values='samples', aggfunc='sum', fill_value=0)
summary['train', 'real_pos'] = df.query('(kind == "real_pos") & (split == "train")')['samples'].values[0]
summary['val', 'real_pos'] = df.query('(kind == "real_pos") & (split == "val")')['samples'].values[0]
summary = summary.sort_index(axis=1, level=[0,1])
print(summary)
