# DVI

# ADDED readme notes - 07/03/2023
Scripts used for main experiments:
- scripts/toy_data_regression.py 
  --> Runs toy regression on dummy task from Wu19. Saves images in NEWpics (ATTENTION: it overwites the one already created)
      Aims at replicating what was done by the autors (saved in pics), but on heteroskedastic case. 
      There were some issues in computing the variance for the heteroskedastic case. The DVI has been fixed, while the fix for MCVI was de-prioritised since we will use Blundell15 as benchmark
- TODO scripts/toy_data_regression_old.py 
  --> Runs toy regression on dummy task from Blundell15 
- scripts/fc_variance.py 
  --> Runs classification on MNIST using 3 fc layers network

Main params to set (inside each script):
- epochs

Note: If the code does not recognize the folder as source and fails to import functions, add the markovalexander/DVI-master folder to PYTHONPATH e.g. running:
```
export PYTHONPATH=<base_repo>/GitHub/bnn-uncertainty/markovalexander/DVI-master
```

Note: More details on code and findings can be found on Google Drive, in 'MLMI4 Advanced ML/Code2019.docx' (wip file, might turn to PDF once done)

# OLD README From the authors of the repo

Pytorch implementation of https://arxiv.org/pdf/1810.03958.pdf with additional experiments:

- classification task
- conv nets
- variational dropout
- variance networks
