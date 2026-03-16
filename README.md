# Mega-Tropiques AI Humidity Retrieval
An AI-based system for retrieving atmospheric relative humidity profiles from the ISRO Mega-Tropiques SAPHIR satellite.

This software uses LightGBM machine learning models trained on satellite brightness temperature data to estimate relative humidity across multiple atmospheric layers.

## Features
- Satellite data preprocessing from HDF5 files
- Machine learning training pipeline
- Multi-layer humidity prediction (L2–L5)
- Geospatial visualization of humidity swaths
- Model evaluation and scientific metrics
- Interactive dashboard using Streamlit

## Project Structure
app.py – Streamlit interface for visualization  
src/ – Core machine learning and analysis code  
scripts/ – Development utilities  
models/ – Location for trained models  

## Installation
Clone the repository and install dependencies:
pip install numpy lightgbm h5py scipy plotly streamlit scikit-learn

streamlit run app.py


## Training the Models
To train humidity retrieval models:


python src/train_full.py --mode preprocess
python src/train_full.py --mode train --rh L2
python src/train_full.py --mode train --rh L3
python src/train_full.py --mode train --rh L4
python src/train_full.py --mode train --rh L5


## Data
This software processes satellite data from the ISRO Mega-Tropiques mission.

The dataset is not included in the repository due to size.

## License
MIT License

## Data Requirements
This software processes SAPHIR satellite observations from the **Megha-Tropiques mission**.

The satellite datasets are distributed through **MOSDAC (Meteorological and Oceanographic Satellite Data Archival Centre)** operated by ISRO.

Because the datasets are large, they are **not included in this repository**.

Users must download the required files from MOSDAC:

https://mosdac.gov.in

### Required Files

Two files are required for each orbit:

• **L1A** – Brightness temperature observations from the SAPHIR radiometer  
• **L2A** – Retrieved atmospheric relative humidity profiles

Both files **must correspond to the same observation date and orbit**.

Example filenames:

l1a_XXXXX.h5  
l2a_XXXXX.h5  

Where `XXXXX` represents the orbit identifier.

### Folder Structure

Place the downloaded files inside the `data/` directory:

data/
   l1a_XXXXX.h5  
   l2a_XXXXX.h5  

### Workflow

1. Download matching **L1A and L2A** SAPHIR files from MOSDAC  
2. Place them in the `data/` folder  
3. Run preprocessing

python train_full.py --mode preprocess

4. Train the model

python train_full.py --mode train --rh L2

5. Launch the visualization interface

streamlit run app.py

Note: A MOSDAC account may be required to access SAPHIR datasets.