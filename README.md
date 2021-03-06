# NeuralMachineTranslation

## About

This is a repository for the final project in 
[HSE Deep Learning in NLP Course](https://github.com/BobaZooba/HSE-Deep-Learning-in-NLP-Course). 
The goal of the project is to develop a machine translation system that comes close in quality to an existing system. 
The project was mainly focused on translating from Russian to Ukrainian, however it is language-agnostic and, in theory,
should work with any language pair. This particular pair was chosen mainly due to the availability of a reference BLEU
score obtained by a system using modern architecture, and my personal ability to manually gauge the quality of 
translations, as well as the possibility of getting some linguistic insight on the intricacies of neural
machine translation.

A report (in Russian) is available [here](report.md).


## How to Reproduce

**Note**: to reproduce the results locally you'll need a Unix-like system and a CUDA-compatible GPU. Alternatively, you can utilize
[Google Colaboratory](https://colab.research.google.com/) (as in an example provided 
[here](colab_train.ipynb)).


In any case, first you'll need to use the following commands:

```
git clone https://github.com/slowwavesleep/NeuralMachineTranslation.git
```

```
cd NeuralMachineTranslation
```

Then install dependencies.

```
pip install -r requirements.txt
```

Or you can use `colab_helper.sh` to install dependencies and get training data.

```
bash colab_helper.sh
```

### Training a Model

To train the main model of this project run this:

```
python train.py main_config.yml
```

Yaml config file contains all the necessary parameters to train the model, as well as the paths specifying where to get
the data and where to save results and models.

### Testing a Model

To evaluate the results of a model run the following command:

```
python test.py results/main/translations.txt
```

Where the only argument specifies the location of a file to evaluate.

## Results

The translations done by models in this project can be found [here](/results).

BLEU score evaluations are presented below.

|Model| Num. of examples|BLEU|
|:-------------|:----------:|-----------:|
|Baseline|100000|2.1|
|Main|100000|32.94|
|Main|800000|49.55|

## Data

Data for training and evaluation is taken from
[Tatoeba Challenge](https://github.com/Helsinki-NLP/Tatoeba-Challenge/blob/master/Data.md).
Model results for this data are available
[here](https://github.com/Helsinki-NLP/Tatoeba-Challenge/blob/master/results/tatoeba-results-all.md).
More specifically, the [rus-ukr pair](https://object.pouta.csc.fi/Tatoeba-Challenge/rus-ukr.tar).

The models may very well be used with other language pairs from other resources. There are a few tools
that can help with that.

### Converting Tab Delimited Data

If you have data in tab delimited format, you can use `prepare_anki.py` to convert into two separate files
assuming that the first and second elements in each line correspond to source and target sentences respectively.


### Splitting Arbitrary Data

If you have aligned sentences stored in two files, you can use `split_data.py` to conveniently split your data into
train, dev, and test parts, which then can be used to train a new model, provided you specify correct file paths in yaml
configuration file.
