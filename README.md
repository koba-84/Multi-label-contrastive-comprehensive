# Multi-Label Contrastive Learning: A Comprehensive Study

*Official implementation of the papers:  
[**Exploring Contrastive Learning for Long-Tailed Multi-Label Text Classification**](<https://scholar.google.com/citations?view_op=view_citation&hl=fr&user=W8aMnu8AAAAJ&citation_for_view=W8aMnu8AAAAJ:u5HHmVD_uO8C1>)  
and  
[**Multi-Label Contrastive Learning: A Comprehensive Study**](<https://scholar.google.com/citations?view_op=view_citation&hl=fr&user=W8aMnu8AAAAJ&citation_for_view=W8aMnu8AAAAJ:2osOgNQ5qMEC>)*


## Key Contributions
- Comprehensive analysis of contrastive learning loss in multi-label classification
- Evaluation across datasets with varying label counts and training data volumes
- Performance insights in both computer vision and natural language processing domains
- Development of a novel loss function to improve multi-label classification performance

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/A-REMPLIR
cd ???
```

### 2. Create Virtual Environment
```bash
python3 -m venv env
source env/bin/activate  # On Windows, use `env\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Wandb Setup
To track all the experimenta, our code use Weight and Biases :
1. Create a Weights & Biases account at [wandb.ai](https://wandb.ai/)
2. Get your API key from your wandb profile
3. In `main.py`, set your Wandb credentials:
```python
import os

# Replace with your credentials
os.environ['WANDB_API_KEY'] = 'your_wandb_api_key'
ENTITY_NAME_WANDB = 'your_wandb_username'
```

## Running Experiments

### NLP Experiments
```bash
cd src
python3 main.py --config config/config_nlp.json
```

### Computer Vision Experiments
```bash
cd src
python3 main.py --config config/config_vision.json
```
To run non-contrastive losses experiments :
```bash
cd src
python3 main.py --config config/config_vision_BCE.json
```


## Citation
If you use this work in your research, please cite:
```
@article{audibert2024multilabelcontrastive,
  title={Multi-Label Contrastive Learning: A Comprehensive Study},
  author={Audibert, Alexandre and Gauffre, Aurélien and Amini, Massih-Reza},
  journal={TBD},
  year={2024}
}
```

## License
[Specify your license here, e.g., MIT, Apache 2.0]



**Note:** Configurations and exact reproduction  require specific dataset access.
