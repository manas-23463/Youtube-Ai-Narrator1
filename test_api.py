#!/usr/bin/env python3
"""
Test client for YouTube AI Narrator Modal API
"""

import requests
import json
import base64
import os
from typing import Optional

class YouTubeAINarratorClient:
    """Client for the YouTube AI Narrator Modal API"""
    
    def __init__(self, base_url: str):
        """
        Initialize the client
        
        Args:
            base_url: Base URL of the Modal API (e.g., https://your-app.modal.run)
        """
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> dict:
        """Check if the API is healthy"""
        response = requests.get(f"{self.base_url}/health_check")
        return response.json()
    
    def get_voices(self) -> dict:
        """Get available ElevenLabs voices"""
        response = requests.get(f"{self.base_url}/get_voices")
        return response.json()
    
    def process_video(
        self,
        youtube_url: str,
        voice_id: Optional[str] = None,
        whisper_model: str = "base",
        output_filename: Optional[str] = None
    ) -> dict:
        """
        Process a YouTube video with AI narrations
        
        Args:
            youtube_url: YouTube video URL
            voice_id: ElevenLabs voice ID (optional, uses default if not provided)
            whisper_model: Whisper model size
            output_filename: Optional output filename
            
        Returns:
            API response with processing results
        """
        payload = {
            "youtube_url": youtube_url,
            "whisper_model": whisper_model
        }
        
        if voice_id:
            payload["voice_id"] = voice_id
        
        if output_filename:
            payload["output_filename"] = output_filename
        
        response = requests.post(
            f"{self.base_url}/process_video_api",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        return response.json()
    
    def download_video(self, response: dict, output_path: str) -> bool:
        """
        Download the processed video from the API response
        
        Args:
            response: API response containing video data
            output_path: Path to save the video file
            
        Returns:
            True if successful, False otherwise
        """
        if not response.get("success"):
            print(f"❌ Processing failed: {response.get('error')}")
            return False
        
        if not response.get("download_url"):
            print("❌ No download URL in response")
            return False
        
        try:
            # Extract base64 data from data URL
            data_url = response["download_url"]
            if data_url.startswith("data:video/mp4;base64,"):
                base64_data = data_url.split(",", 1)[1]
                video_data = base64.b64decode(base64_data)
                
                # Save to file
                with open(output_path, 'wb') as f:
                    f.write(video_data)
                
                print(f"✅ Video saved to: {output_path}")
                return True
            else:
                print("❌ Invalid data URL format")
                return False
                
        except Exception as e:
            print(f"❌ Error downloading video: {str(e)}")
            return False

def main():
    """Example usage of the API client"""
    
    # Replace with your actual Modal API URL
    # You'll get this URL after deploying with: modal deploy modal_api.py
    api_url = "https://your-app.modal.run"  # Replace with actual URL
    
    client = YouTubeAINarratorClient(api_url)
    
    # Test health check
    print("🏥 Health check:")
    health = client.health_check()
    print(json.dumps(health, indent=2))
    print()
    
    # Get available voices
    print("🎙️ Available voices:")
    voices = client.get_voices()
    print(json.dumps(voices, indent=2))
    print()
    
    # Process a video (replace with actual YouTube URL)
    youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Replace with your video
    
    print(f"🎬 Processing video: {youtube_url}")
    result = client.process_video(
        youtube_url=youtube_url,
        whisper_model="base"  # Use "base" for faster processing
    )
    
    print("📊 Processing result:")
    print(json.dumps(result, indent=2))
    
    # Download the video if successful
    if result.get("success"):
        output_filename = result.get("output_filename", "ai_narrated_video.mp4")
        client.download_video(result, output_filename)

if __name__ == "__main__":
    main() 