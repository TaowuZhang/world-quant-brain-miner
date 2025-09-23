#!/usr/bin/env python3
"""
Test script for custom templates with anl/fnd data fields
"""

from bruteforce_template_generator import BruteforceTemplateGenerator
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test custom templates with anl/fnd fields"""
    try:
        # Initialize generator
        gen = BruteforceTemplateGenerator('credential.json')
        
        # Load custom templates
        templates = gen.load_custom_templates('custom_templates.json')
        
        if not templates:
            logger.error("❌ No custom templates loaded")
            return
        
        logger.info(f"🎯 Testing {len(templates)} custom templates")
        
        # Test a few templates to make sure they work
        test_templates = templates[:3]  # Test first 3 templates
        
        for i, template in enumerate(test_templates):
            logger.info(f"📝 Testing template {i+1}: {template[:60]}...")
            
            # Test with a simple simulation
            try:
                result = gen.simulate_template(template, "USA", "close", "INDUSTRY", max_retries=1)
                if result.success:
                    logger.info(f"✅ Template {i+1} successful: Sharpe={result.sharpe:.3f}")
                else:
                    logger.warning(f"⚠️ Template {i+1} failed: {result.error_message}")
            except Exception as e:
                logger.error(f"❌ Template {i+1} error: {e}")
        
        logger.info("🏁 Custom template testing completed")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()
