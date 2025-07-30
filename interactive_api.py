"""
Interactive AI Video Player API
Provides real-time transcription, AI summaries, and learning interactions
"""

import os
import json
import base64
import tempfile
import logging
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import modal
from dotenv import load_dotenv

# Import existing modules
from transcription import transcribe_and_segment
from narrator import ElevenLabsTTS
import openai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Interactive AI Video Player API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:8080", "http://127.0.0.1:8080", "file://"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class VideoProcessRequest(BaseModel):
    youtube_url: str
    auto_trigger: bool = True
    trigger_interval: int = 30
    show_avatar: bool = False
    bgm_enabled: bool = False

class VideoSegment(BaseModel):
    start: float
    end: float
    transcript: str
    summary: str
    confidence: float = 0.0
    words: Optional[List[dict]] = None  # Support word-level timestamps

class VideoProcessResponse(BaseModel):
    success: bool
    video_id: str
    video_title: str
    segments: List[VideoSegment]
    total_duration: float
    error: Optional[str] = None

class LearningPoint(BaseModel):
    timestamp: float
    segment_index: int
    summary: str
    transcript: str
    saved_at: str

# Global storage for learning points (in production, use a database)
learning_points = []

# Create Modal app
stub = modal.App("interactive-video-player-api")

# Create image with all dependencies
interactive_image = (
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
        "openai-whisper>=20231117",
        "yt-dlp>=2023.12.30",
        "pydub>=0.25.1",
        "numpy>=1.24.3"
    ])
    .add_local_dir("/Users/manasmore/Desktop/youtube-AI-narrator2", "/app")
)

@stub.function(
    image=interactive_image,
    timeout=3600,
    memory=8192,
    cpu=4.0,
    secrets=[modal.Secret.from_name("youtube-ai-narrator-secrets-new")]
)
@modal.fastapi_endpoint(method="POST")
def process_video_interactive(request: VideoProcessRequest):
    """Process YouTube video for interactive learning experience"""
    try:
        import sys
        sys.path.append("/app")
        
        from youtube_downloader import YouTubeDownloader
        from transcription import transcribe_and_segment
        import yt_dlp
        import tempfile
        import os
        from datetime import datetime
        
        # Extract video ID from URL
        video_id = extract_video_id(request.youtube_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")
        
        # Download video
        downloader = YouTubeDownloader()
        video_info = downloader.get_video_info(request.youtube_url)
        video_title = video_info.get('title', 'Unknown Title')
        
        # Download audio for transcription
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
            audio_path = temp_audio.name
        
        try:
            downloader.download_audio(request.youtube_url, audio_path)
            
            # Transcribe and segment
            segments = transcribe_and_segment(audio_path)
            
            # Generate AI summaries for each segment
            video_segments = []
            for i, segment in enumerate(segments):
                summary = generate_ai_summary(segment.text)
                video_segments.append(VideoSegment(
                    start=segment.start_time,
                    end=segment.end_time,
                    transcript=segment.text,
                    summary=summary,
                    confidence=segment.confidence
                ))
            
            # Calculate total duration
            total_duration = max(seg.end for seg in video_segments) if video_segments else 0
            
            return VideoProcessResponse(
                success=True,
                video_id=video_id,
                video_title=video_title,
                segments=video_segments,
                total_duration=total_duration
            )
            
        finally:
            # Cleanup temporary files
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return VideoProcessResponse(
            success=False,
            video_id="",
            video_title="",
            segments=[],
            total_duration=0,
            error=str(e)
        )

@stub.function(
    image=interactive_image,
    timeout=300,
    memory=4096,
    cpu=2.0,
    secrets=[modal.Secret.from_name("youtube-ai-narrator-secrets-new")]
)
@modal.fastapi_endpoint(method="POST")
def generate_learning_point(segment_index: int, timestamp: float, transcript: str, current_time: float = None):
    """Generate AI learning point for a specific video segment"""
    try:
        import sys
        sys.path.append("/app")
        from datetime import datetime
        
        # If current_time is provided, generate time-specific summary
        if current_time is not None:
            # For now, use the provided transcript but mark it as time-specific
            summary = generate_ai_summary(transcript, is_time_specific=True)
        else:
            # Generate regular summary
            summary = generate_ai_summary(transcript, is_time_specific=False)
        
        # Create learning point
        learning_point = LearningPoint(
            timestamp=timestamp,
            segment_index=segment_index,
            summary=summary,
            transcript=transcript,
            saved_at=datetime.now().isoformat()
        )
        
        # In production, save to database
        # For now, store in memory
        learning_points.append(learning_point.dict())
        
        return {
            "success": True,
            "learning_point": learning_point.dict()
        }
        
    except Exception as e:
        logger.error(f"Error generating learning point: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@stub.function(
    image=interactive_image,
    timeout=300,
    memory=4096,
    cpu=2.0,
    secrets=[modal.Secret.from_name("youtube-ai-narrator-secrets-new")]
)
@modal.fastapi_endpoint(method="GET")
def get_learning_points():
    """Get all saved learning points"""
    return {
        "success": True,
        "learning_points": learning_points
    }

@stub.function(
    image=interactive_image,
    timeout=300,
    memory=4096,
    cpu=2.0,
    secrets=[modal.Secret.from_name("youtube-ai-narrator-secrets-new")]
)
@modal.fastapi_endpoint(method="GET")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "service": "Interactive AI Video Player API"}

# Helper functions
def extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL"""
    import re
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)',
        r'youtube\.com\/watch\?.*v=([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def generate_ai_summary(text: str, is_time_specific: bool = False) -> str:
    """Generate AI summary using OpenAI GPT"""
    try:
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        if is_time_specific:
            system_prompt = "You are an educational AI assistant. Create a concise, engaging explanation of the video content up to this specific point in time. Focus on what has been covered so far and help learners understand the key concepts presented. Keep it under 100 words and make it easy to understand."
        else:
            system_prompt = "You are an educational AI assistant. Create a concise, engaging summary of the given text that helps learners understand the key concepts. Keep it under 100 words and make it easy to understand."
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"Summarize this educational content: {text}"
                }
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Error generating AI summary: {e}")
        return f"Summary: {text[:100]}..." if len(text) > 100 else text

def extract_transcript_up_to_time(segments: List[VideoSegment], current_time: float) -> str:
    """Extract transcript content up to the current time, using word-level timestamps"""
    if not segments:
        return ""
    
    segment = segments[0]  # We now have only one segment with full transcript
    words = getattr(segment, 'words', None)
    
    logger.info(f"Extracting transcript up to {current_time}s. Words available: {words is not None}")
    
    if words:
        # Use word timestamps to extract only words up to current_time
        relevant_words = []
        for word in words:
            if word['start'] <= current_time:
                relevant_words.append(word['text'])
        
        extracted_text = ' '.join(relevant_words)
        logger.info(f"Extracted {len(relevant_words)} words up to {current_time}s. Text: {extracted_text[:100]}...")
        return extracted_text
    else:
        # Fallback: use the full transcript if word timestamps aren't available
        logger.warning("No word timestamps available, using full transcript")
        return segment.transcript

def generate_audio_narration(text: str, voice_id: str = "21m00Tcm4TlvDq8ikWAM") -> Optional[str]:
    """Generate audio narration using ElevenLabs"""
    try:
        tts = ElevenLabsTTS(voice_id=voice_id)
        audio_data = tts.synthesize_speech(text)
        
        # Convert to base64 for web delivery
        audio_b64 = base64.b64encode(audio_data).decode('utf-8')
        return f"data:audio/mpeg;base64,{audio_b64}"
        
    except Exception as e:
        logger.error(f"Error generating audio narration: {e}")
        return None

# Local development endpoints (for testing without Modal)
@app.post("/process-video")
async def process_video_local(request: VideoProcessRequest):
    """Local endpoint for video processing"""
    try:
        logger.info(f"Processing video request: {request.youtube_url}")
        
        # Extract video ID from URL
        video_id = extract_video_id(request.youtube_url)
        if not video_id:
            raise ValueError("Invalid YouTube URL")
        
        logger.info(f"Extracted video ID: {video_id}")
        
        # Import required modules
        from youtube_downloader import YouTubeDownloader
        from transcription import transcribe_and_segment
        import tempfile
        import os
        
        # Download video info and audio
        fixed_output_dir = os.path.join(os.getcwd(), 'outputs', 'tmp')
        os.makedirs(fixed_output_dir, exist_ok=True)
        downloader = YouTubeDownloader(output_dir=fixed_output_dir)
        
        # Download the video to get info
        logger.info("Downloading video to get info...")
        video_path, audio_path, video_title = downloader.download_video(request.youtube_url)
        
        # Log contents of output directory after download
        logger.info(f"[DEBUG] Output dir contents after download: {os.listdir(fixed_output_dir)}")
        
        logger.info(f"Video title: {video_title}")
        logger.info(f"Video path: {video_path}")
        logger.info(f"Audio path: {audio_path}")
        
        # Log contents of temp directory before transcription
        temp_dir = os.path.dirname(audio_path)
        if os.path.exists(temp_dir):
            logger.info(f"[DEBUG] Temp dir contents before transcription: {os.listdir(temp_dir)}")
        else:
            logger.error(f"[DEBUG] Temp dir does not exist: {temp_dir}")
        
        try:
            
            # Transcribe audio without segmentation
            logger.info("Transcribing audio...")
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_path, word_timestamps=True)
            
            # Get full transcript with word timestamps
            full_transcript = result['text']
            word_timestamps = []
            for segment in result['segments']:
                for word in segment.get('words', []):
                    word_timestamps.append({
                        'text': word['word'],
                        'start': word['start'],
                        'end': word['end']
                    })
            
            logger.info(f"Transcription completed. Full transcript length: {len(full_transcript)}")
            
            # Create a single segment with the full transcript
            video_segments = [VideoSegment(
                start=0,
                end=result['segments'][-1]['end'] if result['segments'] else 0,
                transcript=full_transcript,
                summary="",  # Will be generated on-demand
                confidence=1.0
            )]
            
            # Store word timestamps for precise extraction
            video_segments[0].words = word_timestamps
            
            # Calculate total duration
            total_duration = video_segments[0].end
            
            logger.info(f"Processing complete. Total duration: {total_duration}s")
            
            return VideoProcessResponse(
                success=True,
                video_id=video_id,
                video_title=video_title,
                segments=video_segments,
                total_duration=total_duration
            )
            
        finally:
            # Cleanup temporary files
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                logger.info("Cleaned up temporary audio file")
        
    except Exception as e:
        logger.error(f"Error processing video: {e}")
        return VideoProcessResponse(
            success=False,
            video_id="",
            video_title="",
            segments=[],
            total_duration=0,
            error=str(e)
        )

@app.post("/learning-point")
async def create_learning_point_local(request: dict):
    """Local endpoint for creating learning points"""
    try:
        segment_index = request.get("segment_index")
        timestamp = request.get("timestamp")
        transcript = request.get("transcript")
        current_time = request.get("current_time")
        segments = request.get("segments")
        
        logger.info(f"Creating learning point for segment {segment_index} at {timestamp}s (current_time: {current_time})")
        
        # Use extract_transcript_up_to_time to get transcript up to current_time
        relevant_transcript = ""
        if segments and current_time is not None:
            # Convert dicts to VideoSegment objects if needed
            from pydantic import parse_obj_as
            relevant_segments = parse_obj_as(List[VideoSegment], segments)
            logger.info(f"Processing request for time {current_time}s. Segments count: {len(relevant_segments)}")
            if relevant_segments and hasattr(relevant_segments[0], 'words'):
                logger.info(f"First segment has {len(relevant_segments[0].words) if relevant_segments[0].words else 0} words")
            relevant_transcript = extract_transcript_up_to_time(relevant_segments, float(current_time))
        else:
            relevant_transcript = transcript
        
        logger.info(f"Relevant transcript length: {len(relevant_transcript)}. Content: {relevant_transcript[:100]}...")
        
        # If paused time is less than 10 seconds, show not enough content message
        if current_time is not None and float(current_time) < 10:
            summary = "There is not enough content yet to explain. Please continue watching."
        elif not relevant_transcript or len(relevant_transcript.strip()) < 20:
            summary = "There is not enough content yet to explain. Please continue watching."
        else:
            # Generate summary for the specific portion up to the paused time
            summary = generate_ai_summary(relevant_transcript, is_time_specific=True)
        
        # Create learning point
        learning_point = LearningPoint(
            timestamp=timestamp,
            segment_index=segment_index,
            summary=summary,
            transcript=relevant_transcript,
            saved_at=datetime.now().isoformat()
        )
        
        # Store in memory
        learning_points.append(learning_point.dict())
        
        return {
            "success": True,
            "learning_point": learning_point.dict()
        }
        
    except Exception as e:
        logger.error(f"Error generating learning point: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/learning-points")
async def get_learning_points_local():
    """Local endpoint for getting learning points"""
    return {
        "success": True,
        "learning_points": learning_points
    }

@app.get("/health")
async def health_check_local():
    """Local health check"""
    return {"status": "ok", "service": "Interactive AI Video Player API (Local)"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Interactive AI Video Player API is running"}

@app.post("/generate-avatar")
async def generate_avatar_video(request: dict):
    """Generate avatar video using HeyGen-style API"""
    try:
        text = request.get("text", "")
        avatar_style = request.get("avatar_style", "professional")
        
        logger.info(f"Generating avatar video for text: {text[:50]}...")
        
        # For now, return a placeholder response
        # In a real implementation, this would call HeyGen API
        avatar_options = {
            "professional": "https://app.heygen.com/avatar/65f7c8b8e4b0b8b8b8b8b8b8",
            "casual": "https://app.heygen.com/avatar/65f7c8b8e4b0b8b8b8b8b8b9",
            "friendly": "https://app.heygen.com/avatar/65f7c8b8e4b0b8b8b8b8b8ba"
        }
        
        selected_avatar = avatar_options.get(avatar_style, avatar_options["professional"])
        
        return {
            "success": True,
            "avatar_url": selected_avatar,
            "video_url": f"{selected_avatar}/video.mp4",
            "message": "Avatar video generated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error generating avatar video: {e}")
        return {
            "success": False,
            "error": str(e)
        }

@app.post("/tts-narration")
async def tts_narration(request: dict):
    """Generate TTS audio for AI summary using ElevenLabs"""
    try:
        text = request.get("text", "")
        voice_id = request.get("voice_id", "RvxJMEXhmyfed4d7O5xn")
        if not text:
            return {"success": False, "error": "No text provided"}
        from narrator import ElevenLabsTTS
        tts = ElevenLabsTTS(voice_id=voice_id)
        audio_path = tts.synthesize_speech(text)
        # Read audio as base64
        import base64
        with open(audio_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")
        return {"success": True, "audio_base64": audio_base64}
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002) 