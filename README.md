# DEUCE
This code is for the Dataset Enrichment through Discovery of Underrepresented Conditions and Environments (DEDUCE) pipeline, which was published as "Improving AI Test and Evaluation by Filling Semantic Coverage Gaps with Targeted Generative Augmentation" at the SPIE DCS 2026 conference.

To replicate paper results, we include a link to models and datasets located here: FILL IN WHEN PUBLISHED. Each pipeline run relies heavily on information provided in the config files in the codebase, which must be changed based on the location of the models and dataset paths.

BDD100k dataset download is [here](https://bair.berkeley.edu/blog/2018/05/30/bdd/), CCT dataset download is [here](https://beerys.github.io/CaltechCameraTraps/)

Note that the augmentation module is not included in this codebase, though our data includes the weights used for the CycleGAN-Turbo model

Replicating paper results (note that the augmentation module is not included in this codebase, though our data includes the weights used for the CycleGAN-Turbo model):
- scripts/pipeline_evaluations/full_pipeline.py: runs through the semantic identification and importance
- scripts/pipeline_evaluations/embedding_evaluation_offlabels.py: runs through the embedding evaluation with labels given by the pipeline
- scripts/pipeline_evaluations/embedding_evaluation_hardcode.py: runs through the embedding evaluation with labels given by the pipeline
- scripts/te_evaluations: runs through the test and evaluation of both datasets given locations of the dataset and synthetic data folders

