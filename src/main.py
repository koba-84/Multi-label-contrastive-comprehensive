import argparse as arg
from parse_config import ConfigParser
from trainer.trainer import trainer
import logging
from typing import Dict
import itertools
import os
import wandb
print("Library import done")

############ To Complete ###############
os.environ['WANDB_API_KEY'] = ""
ENTITY_NAME_WANDB = ""
#########################################

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["WANDB__SERVICE_WAIT"] = "400"
wandb.login()

def main(config: Dict[str, int]):
    """Main function of our projects

    :param config: first config file
    :type config:  Dict[str: int]
    """
    config = config.config
    logging.basicConfig(filename='error.log',
                        filemode='w',
                        format='%(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    dic_list = {}
    for key, value in config.items():
        if isinstance(value, type([1])):
            dic_list[key] = value
        else:
            dic_list[key] = [value]
    # Iterate on all possible combinaisons
    keys, values = zip(*dic_list.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]
    for configuration in permutations_dicts:
        trainer(config=configuration, entity_name=ENTITY_NAME_WANDB)


if __name__ == '__main__':
    # We use argParse with a config file
    parser = arg.ArgumentParser()
    parser.add_argument("-c", "--config", help="Add a config training")
    args = parser.parse_args()
    if args.config:
        config = ConfigParser(args.config)
    else:
        raise ValueError("Need to degine config")
    # We give the config to our main
    main(config=config)
    
    