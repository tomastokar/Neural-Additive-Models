# Neural-Additive-Models
Pytorch implementation of neural additive model

## Description
<p align="justify">
This repository contains Pytorch implementation of neural additive models (NAMs) as previously described in [Agarwal et al., 2020] along its demonstration across four different datasets obtained from <code>responsibly.ai</code> (https://docs.responsibly.ai/index.html)
</p>

These datasets include:
  * COMPAS
  * German Credit Dataset
  * Adults Dataset
  * FICO 

## Requirements

  * python >= 3.7
  * pytorch >= 1.5


To run the demos, you may first want to create conda environment and to install all dependencies:
```S
conda env create -f environment.yml
```

You can then activate the environment:
```S
conda activate PyNam
```

## COMPAS
COMPAS demo can be run by:

```
python compas_demo.py
```

![compas_roc](results/roc_curve.png?raw=true "COMPAS - ROC curve")

## TODO
    - German Credit Dataset
    - Adults Dataset
    - FICO 


## References
  * Agarwal, Rishabh, et al. "Neural additive models: Interpretable machine learning with neural nets." 2020.