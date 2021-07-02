# Neural-Additive-Models
Pytorch implementation of neural additive model

## Description
<p align="justify">
This repository contains Pytorch implementation of neural additive models (NAMs) as previously described in [Agarwal et al., 2020] along its demonstration on COMPAS datasets obtained from <code>responsibly.ai</code> (https://docs.responsibly.ai/index.html) [Barocas et al., 2017].
</p>

## Running
To run the demo, you may first want to create conda environment and to install all dependencies:
```S
conda env create -f environment.yml
```

You can then activate the environment:
```S
conda activate PyNam
```

Run the demo:
```
python demo.py
```

<p align="justify">
This executes a routine where NAM model is repeatedly initialized, trained and subsequently tested using randomly selected subsets. Obtained results include visualization of the receiver oparating characteristic curve (ROC) and visualization of the shape functions learned by individual NAMs. 
</p>

![compas_roc](results/compas_roc.png?raw=true "COMPAS - ROC curve")

<p align="justify">
Shape functions describe how individual features affect prediction of the model. Thin lines represent different shape functions from the individual NAMs, to show their agreement. The thick lines represent aggregated shape function. Histograms in the background shows distribution of values of the individual features.
</p>

![compas_roc](results/compas_shapes.png?raw=true "COMPAS - shape functions")


## References
  * Agarwal, Rishabh, et al. "Neural additive models: Interpretable machine learning with neural nets." 2020.
  * Barocas, Solon, et al. "Fairness in machine learning." 2017.