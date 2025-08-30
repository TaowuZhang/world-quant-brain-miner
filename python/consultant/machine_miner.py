import machine_lib as ml
from time import sleep
import time
import logging
import json
import os
from itertools import product
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('machine_mining.log'),
        logging.StreamHandler()
    ]
)

class MachineMiner:
    def __init__(self, username: str, password: str):
        self.brain = ml.WorldQuantBrain(username, password)
        self.alpha_bag = []
        self.gold_bag = []
        
    def mine_alphas(self, region="USA", universe="TOP3000"):
        logging.info(f"Starting machine alpha mining for region: {region}, universe: {universe}")
        
        while True:
            try:
                # Get data fields
                logging.info("Fetching data fields...")
                fields_df = self.brain.get_datafields(region=region, universe=universe)
                logging.info(f"Got {len(fields_df)} data fields")
                
                matrix_fields = self.brain.process_datafields(fields_df, "matrix")
                vector_fields = self.brain.process_datafields(fields_df, "vector")
                logging.info(f"Processed {len(matrix_fields)} matrix fields and {len(vector_fields)} vector fields")
                
                # Generate first order alphas
                logging.info("Generating first order alphas...")
                first_order = self.brain.get_first_order(vector_fields + matrix_fields, self.brain.ops_set)
                logging.info(f"Generated {len(first_order)} first order alphas")
                logging.info(f"Sample alphas: {first_order[:3]}")
                
                # Prepare alpha batches
                alpha_list = [(alpha, 0) for alpha in first_order]
                pools = self.brain.load_task_pool(alpha_list, 10, 10)
                logging.info(f"Created {len(pools)} pools with {len(pools[0]) if pools else 0} tasks each")
                
                # Run simulations
                logging.info("Starting simulations...")
                self.brain.multi_simulate(pools, "INDUSTRY", region, universe, 0)
                
                # Process results
                self._process_results()
                
            except Exception as e:
                logging.error(f"Error in mining loop: {str(e)}")
                sleep(600)
                self.brain.login()
                continue

    def _process_results(self):
        # Implementation of _process_results method
        pass

    def save_results(self):
        timestamp = int(time.time())
        results = {
            "timestamp": timestamp,
            "gold_alphas": self.gold_bag
        }
        
        with open(f'machine_results_{timestamp}.json', 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to machine_results_{timestamp}.json")

def main():
    # Read credentials from credential.txt
    try:
        with open('credential.txt', 'r') as f:
            credentials = json.load(f)
        username = credentials[0]
        password = credentials[1]
    except (FileNotFoundError, json.JSONDecodeError, IndexError) as e:
        raise ValueError(f"Error reading credentials from credential.txt: {e}")
    
    if not username or not password:
        raise ValueError("Invalid credentials in credential.txt")
        
    miner = MachineMiner(username, password)
    miner.mine_alphas()

if __name__ == "__main__":
    main() 