import asyncio
import json
import logging
import os
import websockets
import httpx
from datetime import datetime
import sys
import time
from typing import Dict, List, Optional

# Add the parent directory to the path to import the alpha generator
sys.path.append('/app/src')

from alpha_generator import AlphaGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AlphaGeneratorAgent:
    def __init__(self):
        self.agent_id = "alpha_generator"
        self.agent_hub_url = os.getenv("AGENT_HUB_URL", "http://agent-hub:8000")
        self.websocket_url = f"ws://agent-hub:8000/ws/{self.agent_id}"
        self.generator = None
        self.websocket = None
        self.running = False
        
    async def start(self):
        """Start the alpha generator agent"""
        logger.info("Starting Alpha Generator Agent...")
        
        # Initialize the alpha generator
        try:
            credentials_path = os.getenv("WORLDQUANT_CREDENTIALS_PATH", "/app/credentials/credential.txt")
            moonshot_api_key = os.getenv("MOONSHOT_API_KEY")
            
            if not moonshot_api_key:
                logger.error("MOONSHOT_API_KEY environment variable not set")
                return
            
            self.generator = AlphaGenerator(credentials_path, moonshot_api_key)
            logger.info("Alpha generator initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing alpha generator: {e}")
            return
        
        # Register with agent hub
        await self.register_with_hub()
        
        # Start WebSocket connection
        await self.connect_websocket()
        
        self.running = True
        
    async def stop(self):
        """Stop the alpha generator agent"""
        logger.info("Stopping Alpha Generator Agent...")
        self.running = False
        
        if self.websocket:
            await self.websocket.close()
    
    async def register_with_hub(self):
        """Register this agent with the agent hub"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.agent_hub_url}/agents",
                    json={
                        "name": "Alpha Generator Agent",
                        "agent_type": "alpha_generator",
                        "capabilities": {
                            "generate_alpha": True,
                            "test_alpha": True,
                            "batch_processing": True
                        },
                        "metadata": {
                            "version": "1.0.0",
                            "description": "Generates and tests alpha factors using WorldQuant Brain"
                        }
                    }
                )
                
                if response.status_code == 200:
                    agent_data = response.json()
                    logger.info(f"Registered with agent hub: {agent_data['id']}")
                else:
                    logger.error(f"Failed to register with agent hub: {response.text}")
                    
        except Exception as e:
            logger.error(f"Error registering with agent hub: {e}")
    
    async def connect_websocket(self):
        """Connect to the agent hub WebSocket"""
        try:
            self.websocket = await websockets.connect(self.websocket_url)
            logger.info("Connected to agent hub WebSocket")
            
            # Send registration message
            await self.websocket.send(json.dumps({
                "type": "agent_register",
                "agent_id": self.agent_id,
                "capabilities": ["generate_alpha", "test_alpha", "batch_processing"],
                "timestamp": datetime.utcnow().isoformat()
            }))
            
            # Start message handling
            await self.handle_messages()
            
        except Exception as e:
            logger.error(f"Error connecting to WebSocket: {e}")
    
    async def handle_messages(self):
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self.process_message(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding message: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed")
        except Exception as e:
            logger.error(f"Error in message handling: {e}")
    
    async def process_message(self, message: dict):
        """Process a received message"""
        message_type = message.get("type")
        
        if message_type == "message":
            await self.handle_agent_message(message)
        elif message_type == "ping":
            await self.send_pong()
        else:
            logger.info(f"Received message: {message_type}")
    
    async def handle_agent_message(self, message: dict):
        """Handle a message from another agent"""
        payload = message.get("payload", {})
        action = payload.get("action")
        
        if action == "generate_alpha":
            await self.generate_alpha(payload.get("parameters", {}))
        elif action == "test_alpha":
            await self.test_alpha(payload.get("expression", ""))
        elif action == "batch_generate":
            await self.batch_generate(payload.get("batch_size", 5))
        else:
            logger.warning(f"Unknown action: {action}")
    
    async def generate_alpha(self, parameters: dict):
        """Generate alpha factors"""
        try:
            logger.info("Generating alpha factors...")
            
            # Get data fields and operators
            data_fields = self.generator.get_data_fields()
            operators = self.generator.get_operators()
            
            # Generate alpha ideas
            alpha_ideas = self.generator.generate_alpha_ideas(data_fields, operators)
            
            if alpha_ideas:
                # Test the first alpha
                result = self.generator.test_alpha(alpha_ideas[0])
                
                # Send result back
                await self.send_result({
                    "action": "generate_alpha",
                    "status": "success",
                    "alpha_expression": alpha_ideas[0],
                    "result": result,
                    "total_generated": len(alpha_ideas)
                })
            else:
                await self.send_result({
                    "action": "generate_alpha",
                    "status": "error",
                    "message": "No alpha ideas generated"
                })
                
        except Exception as e:
            logger.error(f"Error generating alpha: {e}")
            await self.send_result({
                "action": "generate_alpha",
                "status": "error",
                "message": str(e)
            })
    
    async def test_alpha(self, expression: str):
        """Test an alpha expression"""
        try:
            logger.info(f"Testing alpha expression: {expression}")
            
            result = self.generator.test_alpha(expression)
            
            await self.send_result({
                "action": "test_alpha",
                "status": "success",
                "expression": expression,
                "result": result
            })
            
        except Exception as e:
            logger.error(f"Error testing alpha: {e}")
            await self.send_result({
                "action": "test_alpha",
                "status": "error",
                "message": str(e)
            })
    
    async def batch_generate(self, batch_size: int):
        """Generate and test a batch of alpha factors"""
        try:
            logger.info(f"Starting batch generation of {batch_size} alphas...")
            
            # Get data fields and operators
            data_fields = self.generator.get_data_fields()
            operators = self.generator.get_operators()
            
            # Generate alpha ideas
            alpha_ideas = self.generator.generate_alpha_ideas(data_fields, operators)
            
            if alpha_ideas:
                # Test batch
                successful = self.generator.test_alpha_batch(alpha_ideas[:batch_size])
                
                # Get results
                results = self.generator.get_results()
                
                await self.send_result({
                    "action": "batch_generate",
                    "status": "success",
                    "batch_size": batch_size,
                    "successful": successful,
                    "results": results
                })
            else:
                await self.send_result({
                    "action": "batch_generate",
                    "status": "error",
                    "message": "No alpha ideas generated"
                })
                
        except Exception as e:
            logger.error(f"Error in batch generation: {e}")
            await self.send_result({
                "action": "batch_generate",
                "status": "error",
                "message": str(e)
            })
    
    async def send_result(self, result: dict):
        """Send a result back to the agent hub"""
        try:
            if self.websocket:
                message = {
                    "type": "agent_response",
                    "agent_id": self.agent_id,
                    "payload": result,
                    "timestamp": datetime.utcnow().isoformat()
                }
                await self.websocket.send(json.dumps(message))
                logger.info(f"Sent result: {result.get('action')}")
        except Exception as e:
            logger.error(f"Error sending result: {e}")
    
    async def send_pong(self):
        """Send pong response to ping"""
        try:
            if self.websocket:
                await self.websocket.send(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }))
        except Exception as e:
            logger.error(f"Error sending pong: {e}")
    
    async def heartbeat(self):
        """Send periodic heartbeat"""
        while self.running:
            try:
                if self.websocket:
                    await self.websocket.send(json.dumps({
                        "type": "heartbeat",
                        "agent_id": self.agent_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }))
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
                await asyncio.sleep(30)

async def main():
    """Main function"""
    agent = AlphaGeneratorAgent()
    
    try:
        # Start heartbeat task
        heartbeat_task = asyncio.create_task(agent.heartbeat())
        
        # Start the agent
        await agent.start()
        
        # Keep the agent running
        while agent.running:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        await agent.stop()
        if 'heartbeat_task' in locals():
            heartbeat_task.cancel()

if __name__ == "__main__":
    asyncio.run(main()) 