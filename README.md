# NeuralMachineTranslation

## About

This is a repository for final project in [HSE Deep Learning in NLP Course](https://github.com/BobaZooba/HSE-Deep-Learning-in-NLP-Course). 
The goal of the project is to develop a machine translation system that comes close in quality to an existing system. The project was
mainly focused on translating from Russian to Ukrainian, however it is language-agnostic and, in theory, should work with any language pair.
This particular pair was chosen mainly due to availability of a reference result obtained by a system using modern architecture and my
personal ability to manually gauge the quality of translations, as well as the possibility of getting some linguistic insight on the
intricacies of neural machine translation.

A report (in Russian) is available [here](https://github.com/slowwavesleep/NeuralMachineTranslation/blob/master/report.md).


## How to reproduce

**Note**: to reproduce the results locally you'll need a Unix-like system and a CUDA-compatible GPU. Alternatively, you can utilize
[Google Colaboratory](https://colab.research.google.com/) (an example is provided 
[here](https://github.com/slowwavesleep/NeuralMachineTranslation/blob/master/colab_train.ipynb)).


In any case, first you'll nedd to use the following commands:

```
git clone https://github.com/slowwavesleep/NeuralMachineTranslation.git
```

```
cd NeuralMachineTranslation
```

### Training a model

To train the main model of this project run this:

```
python train.py main_config.yml
```

Yaml config file contains all the necessary parameters to train the model, as well as the paths specifing where to get
the data and where to save results and models.

### Testing a model

To evaluate the results of a model run the following command:

```
python test.py results/main/translations.txt
```

Where the only argument specifies the location of a file to evaluate.

## Data

Data for training and evaluation is taken from [Tatoeba Challenge](https://github.com/Helsinki-NLP/Tatoeba-Challenge/blob/master/Data.md).
Model results for this data are available [here](https://github.com/Helsinki-NLP/Tatoeba-Challenge/blob/master/results/tatoeba-results-all.md).

## Split arbitrary data

If you have aligned sentences stored in two big files, you can use `split_data.py` to conveniently split your data into train, dev, test, which
then cand be used to train a new model, provided you specify correct file paths in yaml configuration file.
