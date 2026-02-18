from pathlib import Path

def make_splits(data_dirs: list[Path], multi_fragment_train_percentage=0.7):
    '''
    Create train and validation splits starting from the png images found in the data directories.
    In particular, it creates two text files, train.txt and val.txt, containing the relative paths.
    In train, there will be a certain percentage of multi-fragment images, as specified by multi_fragment_train_percentage. The others are put in val.
    A multi-fragment image is defined as an image whose filename contains a '+'.
    All the other images are considered single-fragment images and will be part of the training set.
    Args:
        data_dirs (list[Path]): List of paths to the data directories.
        multi_fragment_train_percentage (float): Percentage of multi-fragment images to include in the training set.
    '''

    train_lines = []
    val_lines = []
    for data_dir in data_dirs:
        rgba_dir = data_dir / 'rgba'
        image_files = sorted(p for p in rgba_dir.glob('*.png'))

        multi_fragment_images = [p for p in image_files if '+' in p.stem]
        single_fragment_images = [p for p in image_files if '+' not in p.stem]

        print(f'Total of {len(single_fragment_images)} single-fragment images found in {data_dir}.')
        print(f'Total of {len(multi_fragment_images)} multi-fragment images found in {data_dir}.')

        num_multi_fragment_train = int(len(multi_fragment_images) * multi_fragment_train_percentage)
        print(f'Assigning {num_multi_fragment_train} multi-fragment images to the training set and {len(multi_fragment_images) - num_multi_fragment_train} to the validation set.')
        train_multi_fragment = multi_fragment_images[:num_multi_fragment_train]
        val_multi_fragment = multi_fragment_images[num_multi_fragment_train:]

        train_lines.extend(str(p.stem) + '\n' for p in train_multi_fragment)
        val_lines.extend(str(p.stem) + '\n' for p in val_multi_fragment)
        train_lines.extend(str(p.stem) + '\n' for p in single_fragment_images)

    with open('data/train.txt', 'w') as f:
        f.writelines(train_lines)
    with open('data/val.txt', 'w') as f:
        f.writelines(val_lines)

if __name__ == '__main__':
    data_dirs = [
        Path('data/unified'),
    ]
    make_splits(data_dirs, multi_fragment_train_percentage=0.65)