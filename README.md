# GRad-CAM enhAnced Convolution neural nEtwork (GRACE)

The repository contains code refered to the work:

_Francesco Paolo Caforio, Giuseppina Andresini, Annalisa Appice, Gennaro Vessio, and Donato Malerba_

[Leveraging Grad-CAM to Improve the Accuracy of Network Intrusion Detection Systems](#) 

Discovery Science – 24th International Conference, DS 2021,  Proceedings. [Lecture Notes in Computer Science](https://dblp.org/db/series/lncs/index.html) (to appear, 2021)

Please cite our work if you find it useful for your research and work.
```
Updating
```

## Code requirements

The code relies on the following **python3.6+** libs.

Packages need are:
* [Tensorflow 1.13](https://www.tensorflow.org/) 
* [Keras 2.3](https://github.com/keras-team/keras) 
* [Scikit-learn](https://scikit-learn.org/stable/)
* [Pandas 0.23.4](https://pandas.pydata.org/)
* [Numpy 1.15.4](https://www.numpy.org/)
* [Matplotlib 2.2](https://matplotlib.org/)

## How to use
Repository contains scripts of all experiments included in the paper:
* __GRACE.ipynb__ : script to run GRACE
* __Grad_CAM.ipynb__ : script to run GRAD-CAM (construction of the heatmap dataset) 
* __CNN+Grad_CAM+NN.ipynb__ : script to run configuration CNN+Grad_CAM+NN 

## Data & models
The datasets used for experiments and the deep learning models computed in the experiments described in Caforio et al. (2021) are accessible from [__DATASETS__](https://drive.google.com/drive/folders/1CacXYvmK5iHJ94rH-ZbDw0tnsuUF0d7X)
