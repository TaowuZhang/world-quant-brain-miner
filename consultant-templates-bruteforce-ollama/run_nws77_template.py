#!/usr/bin/env python3
"""
Runner script for nws77 template testing
Based on the experience shared with nws77 dataset sentiment analysis
"""

import sys
import os
import json
import logging
from bruteforce_template_generator import BruteforceTemplateGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run nws77 template testing with dataset-specific filtering"""
    
    # Check if credentials file exists
    credentials_file = "credential.json"
    if not os.path.exists(credentials_file):
        logger.error(f"❌ Credentials file not found: {credentials_file}")
        logger.info("Please create credential.json with your WorldQuant Brain credentials")
        return
    
    logger.info("🚀 Starting nws77 Template Testing")
    logger.info("📊 Based on experience: 174 backtests, 1 submittable Alpha")
    logger.info("🎯 Template: ts_decay_linear(<ts_Statistical_op/>(<dataprocess_op/>(-group_backfill(vec_max(<nws77/>),country,60, std = 4.0)), 90),5, dense = false)")
    
    # Initialize generator with nws77 dataset targeting
    generator = BruteforceTemplateGenerator(
        credentials_path=credentials_file,
        ollama_model="llama3.1",  # Not used for custom templates
        max_concurrent=8,
        target_dataset="nws77"  # Focus on nws77 dataset
    )
    
    logger.info("🎯 Testing nws77 template with variable substitution")
    logger.info("📊 Search space: 29 data fields × 2 dataprocess ops × 3 ts_statistical ops = 174 combinations")
    
    # Run the bruteforce testing with nws77 template
    generator.run_bruteforce(
        custom_template_file="nws77_template.json",
        max_batches=1,  # Only need 1 batch for nws77 template
        resume=False
    )
    
    logger.info("🏁 nws77 template testing completed")

if __name__ == "__main__":
    main()
