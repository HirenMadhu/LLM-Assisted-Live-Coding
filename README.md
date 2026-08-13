# LLM-Assisted Live Coding

Repository for CPSC 581: Introduction to Machine Learning

## Process to Replicate Results:

1) Create the required conda environment: `conda env create -f environment.yml`

2) Activate the created environment: `conda activate music`

3) Change path to src/.

4) Generate dataset (optionally modify number of generated code files): `python gen_data`

5) Download [Sonic Pi](https://sonic-pi.net) needed for generating the audio files (for Mac-ARM use version 4.5).

6) Open Sonic Pi so that a running instance is present.

7) Generate embeddings (this dynamically creates audio recordings via [python-sonic](https://pypi.org/project/python-sonic/)): 
`python gen_emb`

8) Run Evaluation: `python impl_main`

## Data

The data generation is based on existing [Sonic Pi](https://sonic-pi.net) code snippets.
By using [Jinja2](https://jinja.palletsprojects.com/en/stable/), we transform these snippets into templates with added randomized parameters shown in [`data.json`](./templates/data.json). 
The final dataset is stored into [sonicpi_embeddings.arrow](./sonicpi_embeddings.arrow/) directory and contains 5400 code-audio embeddings, along with the respective code (.pi) file.
The files used for data generation are listed below:

- [gen_data.py](/src/gen_data.py): For every template in the [templates](/templates) directory, it generates the specified (`num_datasets`) number of randomized templates using keys declared in [`data.json`](./templates/data.json). The generated data are stored in the [datasets](/datasets/) directory.

- [gen_recordings.py](/src/gen_recordings.py): For every dataset in [datasets](/datasets/), it utilizes [python-sonic](https://pypi.org/project/python-sonic/) and a running Sonic Pi instance to generate **wav** recordings (20 secs) and store them in [recordings](/recordings) directory.

- [gen_emb.py](/src/gen_emb.py): For each code file in [datasets](/datasets/) directory, generate an 8 second recording and then use [codeEmbed](src/codeEmbed.py) and [wavEmbed](src/wavEmbed.py) to create the embeddings for code and audio. To save space, store the recordings into a temp/ folder and after creating all the embedings for a specific sample, create the temp/ directory.

- [wavEmbed.py](src/wavEmbed.py): Functions for generating WAV embeddings. Sample each audio file at 16 kHz.

- [codeEmbed.py](src/codeEmbed.py): Functions for generating code embeddings.

## Model Training
For the implementation, we used two independent Multi-Layer Perceptrons. 
To quantify the alignment between learned representations, we utilize two additional similarity metrics:
Canonical Correlation Analysis (CCA) and Centered Kernel Alignment (CKA).
The files used for implementation are listed below:

- [impl_main.py](src/impl_main.py): Main file for model training and metric generation. It parses our dataset, trains the model and prints out the results for CCA and CKA metrics.

- [model.py](src/model.py): Definition of EmbeddingAligner model.

- [utils.py](src/utils.py): Various utility functions.