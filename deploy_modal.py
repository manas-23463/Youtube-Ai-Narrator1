#!/usr/bin/env python3
"""
Deployment script for YouTube AI Narrator on Modal
"""

import modal
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def deploy_to_modal():
    """Deploy the YouTube AI Narrator to Modal"""
    
    # Check for required environment variables
    required_vars = ['OPENAI_API_KEY', 'ELEVENLABS_API_KEY']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please set them in your .env file or environment")
        return False
    
    try:
        print("🔐 Setting up Modal secrets...")
        
        # Create secrets for API keys
        secrets = {
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "ELEVENLABS_API_KEY": os.getenv("ELEVENLABS_API_KEY")
        }
        
        # Create Modal secret
        secret = modal.Secret.from_dict(secrets)
        print("✅ Modal secrets created successfully")
        
        # Deploy the API
        print("🚀 Deploying YouTube AI Narrator API...")
        
        # Import and deploy
        from modal_api import app
        
        # Deploy the app
        app.deploy()
        
        print("✅ Deployment successful!")
        print("\n📋 API Endpoints:")
        print("- POST /process_video_api - Process a YouTube video")
        print("- GET /health_check - Health check")
        print("- GET /get_voices - List available voices")
        
        return True
        
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        return False

if __name__ == "__main__":
    deploy_to_modal() 