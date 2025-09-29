import pandas as pd

from pos_real import PositiveRealMatchDataset
from pos_synth import PositiveSyntheticMatchDataset
from neg import NegativeMatchDataset

data = []
for root in ['data/organized', 'data/organized_test']:
    dset = PositiveRealMatchDataset(root)
    data.append({'kind': 'real_pos', 'pad': None, 'root': root, 'samples': len(dset)})

    for pad in (0, 5, 10, 15):
        dset = PositiveSyntheticMatchDataset(root, pad=pad)
        data.append({'kind': 'synth_pos', 'pad': pad, 'root': root, 'samples': len(dset)})

        dset = NegativeMatchDataset(root, pad=pad)
        data.append({'kind': 'neg', 'pad': pad, 'root': root, 'samples': len(dset)})

df = pd.DataFrame(data)
df.to_csv('dataset_stats.csv', index=False)

# Print summary, root and kind on columns, pad on rows
# include pad=None for real_pos as well, repeating the value for all pads
summary = df.pivot_table(index='pad', columns=['root', 'kind'], values='samples', aggfunc='sum', fill_value=0)
summary['data/organized', 'real_pos'] = df.query('(kind == "real_pos") & (root == "data/organized")')['samples'].values[0]
summary['data/organized_test', 'real_pos'] = df.query('(kind == "real_pos") & (root == "data/organized_test")')['samples'].values[0]
summary = summary.sort_index(axis=1, level=[0,1])
print(summary)
