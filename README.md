# Papyrus Fragment Matching with Deep Learning

Papyrus Fragment Matching is an experimental tool for finding matches between fragments of ancient papyrus. It produces files that can be imported into [JoinPap](https://github.com/cnr-isti-vclab/JoinPap) for visualizing the most promising fragment matchings.

This software is devoped within the PRIN PNRR 2022 project [Reconstructing Fragmentary Papyri through Human-Machine Interaction](https://www.joinpap.unifi.it/),
a joint effort between the Istituto Papirologico "Girolamo Vitelli" in Florence, and the Istituto di Scienza e Tecnologie dell'Informazione "A. Faedo" of the National Research Council (ISTI-CNR), Pisa.

## Requirements

- Python 3.10
- **[Highly Recommended]** A device with CUDA GPUs to accelerate the computation and at least 24 CPU cores and 64GB of RAM

## Installation

We suggest to employ virtual environments to contain all the dependencies:

```cmd
python -m venv .venv
source .venv/bin/activate
```

Then, the tools can be installed as follows:
```cmd
pip install git+https://github.com/mesnico/papyrus-matching.git
```

## Usage

### 1. Data preparation

The tool assumes fragments are contained in `.png` file format into a folder, and each fragment comes with the recto and verso images, following the convention: `filename.png` for recto and `filename_back.png` for verso.

You can find an example in folder [TODO]. 

> Be aware that we assume the fragments are properly scanned, masked, and recto/verso aligned consistently before using this software. 

> Note that jpg is not a suitable format because alpha channel (used for properly cropping the fragments) cannot be stored.

If the recto/verso are not properly cropped (they have transparent margins that extend beyond the area of the fragment), you can use the following script to automatically crop them:

```
crop_fragments my_fragments
```

where `my_fragments` is an example name of the directory where the images are stored.
This command will create a new folder in the same directory called `my_fragments_cropped`.

### 2. Analyzing all fragment pairs
This is a very computationally intensive task. We have, for each pair, to analyze all the possible translations of fragment A into fragment B to understand where and if they can match at certain positions. This is done for both recto and verso sides.

```
compute my_fragments_cropped --output_dir results
```

### 3. Merge the analysis of recto and verso
The final step is to merge the results obtained from recto and verso sides:

```
postprocess results/my_fragments_cropped
```

### 4. Importing into JoinPap
In JoinPap:
- load the fragments stored in `my_fragments` folder
- open the AI tool interface (robot icon)
- browse to `results/my_fragments_cropped` directory and select it.
- wait for the results to appear

## Training

This repository also contains the code for training a custom version of the matching model.

> The instructions on this functionality will be released soon