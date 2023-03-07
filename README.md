# Entities Annotation from Electronic Health Records

## Project Description
TER Code

## How to Install Dependencies
To install dependencies, run:

```
pip install -r requirements.txt
```
Note that to be able to correctly install pytesseract you should follow [this guide](https://pypi.org/project/pytesseract/), but processed text is already available in [test.txt](data/test.txt). In addition, to use SpaCy "transformer" component, you need to have a compatible version of CUDA library installed. Please, refer to SpaCy installation [guide](https://spacy.io/usage/embeddings-transformers#transformers) for more information.
Moreover, to perform hyperparameter tuning, it is needed a WanB account. To set up a WandB environment, refer to [here](https://docs.wandb.ai/guides/integrations/spacy).

## Project structure
[cross_validation.py](cross_validation.py) and [cross_validation_alt.py](cross_validation_alt.py) run a cross-validation process followed by an evaluation process respectively on pipelines able to label DRUG and SYM and DRUG, ADR and NLD.<br/>
[metrics.py](metrics.py) contains a list of functions to evaluate the model. To test a pipeline it is possible to launch it changing the path in the code, otherwise it is possible to use:
```
 python -m spacy benchmark accuracy model_path test_data_path --gpu-id 0
```
[data](data) contains train and test data.<br/>
[cv](cv) and [cv_alt](cv_alt) contain data folds created for the cross-validation process, specifically for models able to label DRUG and SYM and DRUG, ADR and NLD.<br/>
[configs](configs) contains spaCy config files, specifically [config_3.cfg](configs/config_3.cfg) for the pipeline which labels DRUG, ADR and NLD, and [config_4.cfg](configs/config_4.cfg) for the one which labels DRUG and SYM.<br/>
[dict.py](dict.py) generate a pattern file containing patterns for the Entity Ruler component. Not that pattern files are already provided: [pattern.jsonl](pattern.jsonl) contains DRUG patterns, while [pattern_2.jsonl](pattern_2.jsonl) contains DRUG and SYM patterns. <br/>
[sweeps_using_config.py](sweeps_using_config.py) allows merging spaCy config file and WandB yaml file.<br/>
[my_sweep.yml](my_sweep.yml) and [my_sweep_2.yml](my_sweep_2.yml) contain two possible configuration to run hyperparameter tuning using wandb.<br/>
[pdf_extraction.py](pdf_extraction.py) contains code to extract text from pdf file. Extracted text is already available in [data/test.txt](data/test.txt)<br/>
[doc.py](doc.py) contains methods to extract annotations from docx files.
## Additional notes
To correctly run the code, some paths may need to be changed, therefore check them before running a file.




