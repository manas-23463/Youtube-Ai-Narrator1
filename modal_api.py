import modal
from pydantic import BaseModel
from typing import Optional
import base64
import os
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

app = modal.App("youtube-ai-narrator-api")

# Create image with all dependencies for video processing (updated with moviepy)
processing_image_v2 = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .run_commands(
        "pip install --upgrade pip",
        "pip install yt-dlp>=2023.12.30",
        "pip install openai-whisper>=20231117",
        "pip install openai>=1.6.1",
        "pip install elevenlabs>=0.2.27",
        "pip install requests>=2.31.0",
        "pip install python-dotenv>=1.0.0",
        "pip install pydub>=0.25.1",
        "pip install numpy>=1.24.3",
        "pip install torch>=2.1.2",
        "pip install torchaudio>=2.1.2",
        "pip install ffmpeg-python>=0.2.0"
    )
    .add_local_dir("/Users/manasmore/Desktop/youtube-AI-narrator2", "/app")
)

# Create image with FastAPI dependencies
api_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install([
        "fastapi>=0.104.1",
        "uvicorn>=0.24.0",
        "python-multipart>=0.0.6",
        "pydantic>=2.0.0",
        "openai>=1.6.1",
        "elevenlabs>=0.2.27",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "openai-whisper>=20231117"
    ])
    .add_local_dir("/Users/manasmore/Desktop/youtube-AI-narrator2", "/app")
)

class VideoProcessRequest(BaseModel):
    youtube_url: str
    voice_id: Optional[str] = "21m00Tcm4TlvDq8ikWAM"
    whisper_model: Optional[str] = "base"
    output_filename: Optional[str] = None
    cookies_file: Optional[str] = None  # Base64 encoded cookies file

class VideoProcessResponse(BaseModel):
    success: bool
    video_title: Optional[str] = None
    segments_count: Optional[int] = None
    narrations_count: Optional[int] = None
    output_filename: Optional[str] = None
    download_url: Optional[str] = None
    error: Optional[str] = None

@app.function(
    image=processing_image_v2,
    gpu="T4",
    timeout=3600,
    memory=8192,
    cpu=4.0,
    secrets=[modal.Secret.from_name("youtube-ai-narrator-secrets-new")]
)
def process_video_modal(
    youtube_url: str,
    voice_id: str = "21m00Tcm4TlvDq8ikWAM",
    whisper_model: str = "base",
    output_filename: Optional[str] = None,
    cookies_file: Optional[str] = None
) -> dict:
    """Process a YouTube video with AI narrations using the full pipeline"""
    import os
    import tempfile
    import logging
    from datetime import datetime
    import sys
    import base64
    
    # Add the app directory to Python path
    sys.path.append("/app")
    
    # Set up logging
    logger = logging.getLogger(__name__)
    

    
    # Check required environment variables
    required_keys = ['OPENAI_API_KEY', 'ELEVENLABS_API_KEY']
    for key in required_keys:
        if not os.getenv(key):
            raise ValueError(f"Missing required environment variable: {key}")
    
    try:
        logger.info(f"Starting YouTube AI Narrator processing for: {youtube_url}")
        
        # Create temporary directory
        temp_dir = tempfile.mkdtemp(prefix='youtube_ai_narrator_')
        
        # Handle cookies file if provided
        cookies_path = None
        if cookies_file:
            try:
                # Decode base64 cookies file
                cookies_data = base64.b64decode(cookies_file)
                cookies_path = os.path.join(temp_dir, "youtube_cookies.txt")
                with open(cookies_path, 'wb') as f:
                    f.write(cookies_data)
                logger.info("Cookies file created successfully")
            except Exception as e:
                logger.warning(f"Failed to process cookies file: {str(e)}")
        else:
            # Use default cookies file if available
            default_cookies_path = "/app/youtube_cookies.txt"
            if os.path.exists(default_cookies_path):
                cookies_path = default_cookies_path
                logger.info("Using default cookies file")
        
        # Step 1: Download video
        logger.info("Step 1: Downloading YouTube video...")
        from youtube_downloader import YouTubeDownloader
        downloader = YouTubeDownloader(temp_dir, cookies_path)
        video_path, audio_path, video_title = downloader.download_video(youtube_url)
        
        video_title = video_title.replace('/', '_').replace('\\', '_')
        logger.info(f"Downloaded: {video_title}")
        
        # Step 2: Transcribe and segment audio
        logger.info("Step 2: Transcribing and segmenting audio...")
        from transcription import TranscriptionSegmenter
        transcriber = TranscriptionSegmenter(whisper_model)
        segments = transcriber.segment_transcription(audio_path)
        print(f"[DEBUG] Segments after transcription: {segments}")
        logger.info(f"Segments after transcription: {segments}")
        if not segments:
            print("[DEBUG] No segments found after transcription!")
            logger.error("No segments found after transcription!")
            raise ValueError("No segments found in transcription")
        logger.info(f"Found {len(segments)} concept segments")
        print(f"[DEBUG] Passing {len(segments)} segments to narration step.")
        logger.info(f"Passing {len(segments)} segments to narration step.")
        
        # Step 3: Generate AI narrations
        logger.info("Step 3: Generating AI narrations...")
        from narrator import ConceptNarrator
        narrator = ConceptNarrator(voice_id=voice_id)
        narrations = narrator.create_narrations(segments, video_title)
        
        if not narrations:
            raise ValueError("No narrations generated")
        
        logger.info(f"Generated {len(narrations)} narrations")
        
        # Step 4: Process video with narrations
        logger.info("Step 4: Processing video with narrations...")
        from video_processor import VideoProcessor
        video_processor = VideoProcessor(temp_dir)
        
        # Generate output filename if not provided
        if not output_filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title[:50]  # Limit length
            output_filename = f"{safe_title}_AI_Narrated_{timestamp}.mp4"
        
        output_path = os.path.join(temp_dir, output_filename)
        
        # Process the video
        final_video = video_processor.process_video_with_narrations(
            video_path, narrations, output_path
        )
        
        logger.info(f"✅ Processing complete! Output: {final_video}")
        
        # Read the final video file
        with open(final_video, 'rb') as f:
            video_data = f.read()
        
        # Cleanup
        try:
            narrator.cleanup_narrations()
            video_processor.cleanup()
            downloader.cleanup()
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")
        
        # Return results
        return {
            "success": True,
            "video_title": video_title,
            "segments_count": len(segments),
            "narrations_count": len(narrations),
            "output_filename": output_filename,
            "video_data": video_data,
            "video_size_bytes": len(video_data)
        }
        
    except Exception as e:
        logger.error(f"❌ Top-level error in process_video_modal: {str(e)}")
        print(f"[DEBUG] ❌ Top-level error in process_video_modal: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.function(
    image=api_image,
    timeout=3600,
    memory=8192,
    cpu=4.0,
    secrets=[modal.Secret.from_name("youtube-ai-narrator-secrets-new")]
)
@modal.fastapi_endpoint(method="POST")
def process_video_api(request: VideoProcessRequest) -> VideoProcessResponse:
    """REST API endpoint to process YouTube videos with AI narrations"""
    try:
        # Call the processing function within the same app
        result = process_video_modal.remote(
            youtube_url=request.youtube_url,
            voice_id=request.voice_id,
            whisper_model=request.whisper_model,
            output_filename=request.output_filename,
            cookies_file=request.cookies_file
        )
        
        if result["success"]:
            video_data_b64 = base64.b64encode(result["video_data"]).decode('utf-8')
            return VideoProcessResponse(
                success=True,
                video_title=result["video_title"],
                segments_count=result["segments_count"],
                narrations_count=result["narrations_count"],
                output_filename=result["output_filename"],
                download_url=f"data:video/mp4;base64,{video_data_b64}",
                error=None
            )
        else:
            return VideoProcessResponse(
                success=False,
                error=result["error"]
            )
    except Exception as e:
        return VideoProcessResponse(success=False, error=str(e))

@app.function(
    image=api_image,
    timeout=3600,
    memory=8192,
    cpu=4.0,
    secrets=[modal.Secret.from_name("youtube-ai-narrator-secrets-new")]
)
@modal.fastapi_endpoint(method="GET")
def health_check():
    return {"status": "ok"}

@app.function(
    image=api_image,
    timeout=3600,
    memory=8192,
    cpu=4.0,
    secrets=[modal.Secret.from_name("youtube-ai-narrator-secrets-new")]
)
@modal.fastapi_endpoint(method="GET")
def get_voices():
    """Get available ElevenLabs voices"""
    try:
        import sys
        sys.path.append("/app")
        from narrator import ElevenLabsTTS
        
        # Initialize TTS with a dummy voice ID just to access the API
        tts = ElevenLabsTTS(voice_id="dummy")
        
        # Try to get available voices
        try:
            voices = tts.get_available_voices()
        except Exception as voice_error:
            # If we can't get voices due to permissions, return popular default voices
            voices = [
                {
                    "name": "Rachel",
                    "voice_id": "21m00Tcm4TlvDq8ikWAM",
                    "category": "premade",
                    "description": "Professional female voice"
                },
                {
                    "name": "Domi",
                    "voice_id": "AZnzlk1XvdvUeBnXmlld",
                    "category": "premade", 
                    "description": "Professional female voice"
                },
                {
                    "name": "Bella",
                    "voice_id": "EXAVITQu4vr4xnSDxMaL",
                    "category": "premade",
                    "description": "Professional female voice"
                },
                {
                    "name": "Antoni",
                    "voice_id": "ErXwobaYiN019PkySvjV",
                    "category": "premade",
                    "description": "Professional male voice"
                },
                {
                    "name": "Elli",
                    "voice_id": "MF3mGyEYCl7XYWbV9V6O",
                    "category": "premade",
                    "description": "Professional female voice"
                }
            ]
        
        return {
            "success": True,
            "voices": voices,
            "count": len(voices) if voices else 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "voices": [],
            "count": 0
        }

if __name__ == "__main__":
    print("YouTube AI Narrator Modal API")
    print("Deploy with: modal deploy modal_api.py")
    print("Then access the API endpoints:")
    print("- POST /process_video_api - Process a YouTube video")
    print("- GET /health_check - Health check")
    print("- GET /get_voices - List available voices") 