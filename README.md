[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

# Patent AI Research Project

This repository contains the development of a research project in Artificial Intelligence focused on analyzing and modeling data related to patents. The goal is to create tools, methods, and models that support the extraction, classification, and interpretation of technological information from patent databases.

## Project Structure

Below is the proposed directory structure to organize the research artifacts clearly and productively:

``` 
data/
├── raw/ # Raw (unprocessed) data
├── processed/ # Cleaned and processed data
└── README.md # Description of the datasets

docs/ # Project documentation
├── README.md # Reading recommendations, drafts, etc.
├── meeting-notes/ # Meeting minutes

notebooks/ # Jupyter notebooks for analysis/experiments
├── exploratory/ # Exploratory analyses
├── experiments/ # Formal experiments
└── sandbox/ # Drafts and tests

src/ # Project source code
├── data/ # Data processing scripts
├── features/ # Feature engineering
├── models/ # Model development
└── utils/ # Common utilities

models/ # Trained and serialized models
└── README.md # Model descriptions

.gitignore # Files to be ignored by Git

LICENSE # Project license

README.md # Project overview (this file)

requirements.txt # Python dependencies
```

## How to Start Implementation
Create a virtual environment using the `requirements.txt` file with the command: `python3 -m venv venv`; Activate it with the command `source venv/bin/activate`.

1. Create a virtual environment using the command: `python3 -m venv venv`.
2. Activate the virtual environment using the command: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows).
3. Install the dependencies listed in the `requirements.txt` file using the command: `pip install -r requirements.txt`.
4. Open a new terminal to execute these commands.

When installing a new library, export your environment with the command `pip freeze > requirements.txt`.

## Project Map
[stemming.ipynb](notebooks/experiments/stemming.ipynb): This notebook process the TRIZ and patent bases of text normalization and linguistic preprocessing for patent data using Natural Language Processing (NLP) techniques, generating a standard lemmatized data.

[to_finder.ipynb](notebooks/experiments/to_finder.ipynb): This notebook presents a workflow for extracting and matching "Task" and "Object" elements from patent documents using Natural Language Processing (NLP) techniques.

[translation_patents_inpi_dataset.ipynb](notebooks/experiments/translation_patents_inpi_dataset.ipynb)


[exploring_triz_multilingual.ipynb](notebooks/exploratory/exploring_triz_multilingual.ipynb)

## Contacts

[Carla Bonato Marcolin](http://lattes.cnpq.br/3648130183559806)  
  Email: carla@ufu.br

[Katia Cinara Tregnago Cunha](http://lattes.cnpq.br/4187253937050785)  
  Email: katia.patentes@gmail.com

Marcos Antenor  
  Email: marcos.antenor@ufu.br

[Patrick Luiz de Araújo](https://github.com/PatrickLdA)  
  Email: patrickluizdearaujo@gmail.com
