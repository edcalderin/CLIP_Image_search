# CLIP Image search

It consists on an image searcher based on user input. The applications is able to search for images that match the query. The resulting images will be shown in rank order, with the ranking calculated using the cosine similarity.

**Video:**  
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/owXqDU_kKsc/hqdefault.jpg)](https://youtu.be/owXqDU_kKsc)


## Setup

* Install conda

* Create the environment:  `conda create -n clip_image_search python=3.12`

* Activate environment: `source activate clip_image_search` or  
`conda activate clip_image_search` if the previous one fails.

* Install pip: `conda install pip` This ensures that the requirements will be installed inside the environment directory.

* (Optional) Run `where pip` and verify that pip command comes from the current environment.

* Install requirements: `pip install -r requirements.txt`

## Run the app

1. Generate the embeddings: `python -m src.get_embeddings`

It will create a binary with the embeddings in the root directory called `embeddings.pkl`

2. Start the streamlit application: `python -m streamlit run src/app.py`

3. Feel free to add more images to enrich the application. I only collected a few.

## Lint

Ruff commands:

* ```ruff format .```
* ```ruff check . --fix```
