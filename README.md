# Multi-Label Contrastive Learning a Comprehensive study

## Requirements
python 

## Installing with the following command
```bash
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
cd src
```
## Setting Up Wandb
In main.py you have to modify the following line to run our code. You have to modify "entity" for wandb and give your WANDB_API_KEY.
```python
############ To Complete ###############
os.environ['WANDB_API_KEY'] = ""
ENTITY_NAME_WANDB = ""
#########################################
```

## Run our Code
In the 'src' directory, execute the following command:
```bash
python3 main.py --config config/config.json
```

