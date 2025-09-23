#!/usr/bin/env python3
"""
Test custom templates without Ollama
"""

from bruteforce_template_generator import BruteforceTemplateGenerator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test custom templates without Ollama"""
    try:
        # Initialize generator
        gen = BruteforceTemplateGenerator('credential.json')
        
        # Test custom template loading
        templates = gen.load_custom_templates('custom_templates.json')
        logger.info(f"🎯 Loaded {len(templates)} custom templates")
        
        # Test a single template to make sure it works
        if templates:
            test_template = templates[0]
            logger.info(f"📝 Testing template: {test_template[:60]}...")
            
            # Test with a simple simulation (this should NOT use Ollama)
            result = gen.simulate_template(test_template, "USA", "close", "INDUSTRY", max_retries=1)
            if result.success:
                logger.info(f"✅ Template test successful: Sharpe={result.sharpe:.3f}")
            else:
                logger.info(f"⚠️ Template test failed: {result.error_message}")
        
        logger.info("🏁 Custom template test completed - NO OLLAMA USED!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()
