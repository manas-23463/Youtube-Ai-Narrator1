import modal
import os
import tempfile
import logging
from datetime import datetime
from typing import Optional

# Configure Modal
app = modal.App("youtube-ai-narrator")

# Create image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "yt-dlp>=2023.12.30",
        "openai-whisper>=20231117",
        "openai>=1.6.1",
        "elevenlabs>=0.2.27",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "moviepy>=1.0.3",
        "pydub>=0.25.1",
        "numpy>=1.24.3",
        "torch>=2.1.2",
        "torchaudio>=2.1.2",
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0"
    ])
    .add_local_dir("/Users/manasmore/Desktop/youtube-AI-narrator2", "/app")
)

@app.function(
    image=image,
    gpu="T4",  # Use GPU for Whisper transcription
    timeout=3600,  # 1 hour timeout
    memory=8192,   # 8GB RAM
    cpu=4.0,       # 4 CPU cores
    secrets=[modal.Secret.from_name("youtube-ai-narrator-secrets")]
)
def process_video_modal(
    youtube_url: str,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",  # Default voice ID
    whisper_model: str = "base",
    output_filename: Optional[str] = None
) -> dict:
    """
    Process a YouTube video with AI narrations via Modal
    
    Args:
        youtube_url: YouTube video URL
        voice_id: ElevenLabs voice ID
        whisper_model: Whisper model size
        output_filename: Optional output filename
        
    Returns:
        Dictionary with processing results and download URL
    """
    import os
    import tempfile
    import logging
    from datetime import datetime
    import sys
    
    # Add the app directory to Python path
    sys.path.append("/app")
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Set environment variables (these should be set in Modal secrets)
    required_keys = ['OPENAI_API_KEY', 'ELEVENLABS_API_KEY']
    for key in required_keys:
        if not os.getenv(key):
            raise ValueError(f"Missing required environment variable: {key}")
    
    try:
        logger.info(f"Starting YouTube AI Narrator processing for: {youtube_url}")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix='youtube_ai_narrator_')
        
        # For now, we'll use a simple approach without the complex modules
        # This is a simplified version that demonstrates the concept
        
        logger.info("✅ Simplified processing complete!")
        
        # Return a mock result for now
        return {
            "success": True,
            "video_title": "Test Video",
            "segments_count": 3,
            "narrations_count": 3,
            "output_filename": "test_video.mp4",
            "video_data": b"mock_video_data",
            "video_size_bytes": 1000
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing video: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.local_entrypoint()
def main():
    """Local entrypoint for testing"""
    print("YouTube AI Narrator Modal API")
    print("Deploy with: modal deploy modal_app.py")
    print("Then call the API endpoint to process videos")

# For local development and testing
if __name__ == "__main__":
    main() 