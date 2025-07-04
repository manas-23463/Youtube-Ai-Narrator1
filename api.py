from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import logging
from datetime import datetime
import traceback

# Import your existing modules
from youtube_downloader import YouTubeDownloader
from transcription import TranscriptionSegmenter
from narrator import create_narrations_for_segments
from video_processor import process_video_with_ai_narrations

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Ensure outputs directory exists
os.makedirs('./outputs', exist_ok=True)

def verify_api_keys():
    """Verify that required API keys are available"""
    required_keys = ['OPENAI_API_KEY', 'ELEVENLABS_API_KEY']
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    try:
        verify_api_keys()
        return jsonify({
            "status": "healthy",
            "message": "YouTube AI Narrator API is running",
            "timestamp": datetime.now().isoformat()
        }), 200
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/process-video', methods=['POST'])
def process_video():
    """
    Process a YouTube video with AI narrations
    
    Expected JSON body:
    {
        "youtube_url": "https://www.youtube.com/watch?v=...",
        "voice_id": "pNInz6obpgDQGcFmaJgB",  // Optional, defaults to Rachel
        "output_format": "mp4"  // Optional, always mp4 for now
    }
    """
    try:
        # Verify API keys first
        verify_api_keys()
        
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({
                "error": "No JSON data provided",
                "message": "Please provide a JSON body with youtube_url"
            }), 400
        
        youtube_url = data.get('youtube_url')
        if not youtube_url:
            return jsonify({
                "error": "Missing youtube_url",
                "message": "Please provide a youtube_url in the request body"
            }), 400
        
        voice_id = data.get('voice_id', 'pNInz6obpgDQGcFmaJgB')  # Default to Rachel
        
        logger.info(f"🎬 API: Processing YouTube video: {youtube_url}")
        
        # Create temporary directory for this request
        temp_dir = tempfile.mkdtemp(prefix='youtube_ai_narrator_api_')
        
        try:
            # Step 1: Download YouTube video
            logger.info("📥 API: Step 1 - Downloading YouTube video...")
            downloader = YouTubeDownloader(temp_dir)
            video_path, audio_path, video_title = downloader.download_video(youtube_url)
            logger.info(f"✅ API: Downloaded: {video_title}")
            
            # Step 2: Transcribe and segment audio
            logger.info("🎤 API: Step 2 - Transcribing and segmenting audio...")
            segmenter = TranscriptionSegmenter()
            segments = segmenter.segment_transcription(audio_path)
            logger.info(f"✅ API: Found {len(segments)} concept segments")
            
            # Step 3: Generate AI narrations
            logger.info("🤖 API: Step 3 - Generating AI narrations...")
            narrations = create_narrations_for_segments(segments, voice_id, video_title)
            logger.info(f"✅ API: Generated {len(narrations)} narrations")
            
            # Step 4: Process video with narrations
            logger.info("🎬 API: Step 4 - Processing video with narrations...")
            
            # Create output filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            output_filename = f"{safe_title}_AI_Narrated_{timestamp}.mp4"
            output_path = os.path.join('./outputs', output_filename)
            
            # Process the video
            final_video = process_video_with_ai_narrations(video_path, narrations, output_path)
            
            logger.info(f"✅ API: Processing complete! Output: {final_video}")
            
            # Get file size for response
            file_size = os.path.getsize(final_video)
            
            # Create response data
            response_data = {
                "status": "success",
                "message": "Video processed successfully",
                "video_title": video_title,
                "concepts_found": len(segments),
                "narrations_generated": len(narrations),
                "output_file": output_filename,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "download_url": f"http://localhost:5001/download/{output_filename}",
                "download_endpoint": f"/download/{output_filename}",
                "timestamp": datetime.now().isoformat(),
                "concept_breakdown": [
                    {
                        "concept": seg.concept_title,
                        "time_range": f"{seg.start_time:.1f}s - {seg.end_time:.1f}s",
                        "explanation": narrations[i].explanation_text[:100] + "..." if len(narrations[i].explanation_text) > 100 else narrations[i].explanation_text
                    }
                    for i, seg in enumerate(segments)
                ]
            }
            
            return jsonify(response_data), 200
            
        finally:
            # Cleanup downloader
            if 'downloader' in locals():
                downloader.cleanup()
            
    except Exception as e:
        logger.error(f"❌ API: Error processing video: {str(e)}")
        logger.error(f"❌ API: Traceback: {traceback.format_exc()}")
        
        return jsonify({
            "status": "error",
            "error": str(e),
            "message": "An error occurred while processing the video",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Download processed video file"""
    try:
        file_path = os.path.join('./outputs', filename)
        
        if not os.path.exists(file_path):
            return jsonify({
                "error": "File not found",
                "message": f"The file {filename} does not exist"
            }), 404
        
        # Send file with proper headers
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='video/mp4'
        )
        
    except Exception as e:
        logger.error(f"❌ API: Error downloading file: {str(e)}")
        return jsonify({
            "error": str(e),
            "message": "An error occurred while downloading the file"
        }), 500

@app.route('/list-files', methods=['GET'])
def list_files():
    """List all processed video files"""
    try:
        files = []
        outputs_dir = './outputs'
        
        if os.path.exists(outputs_dir):
            for filename in os.listdir(outputs_dir):
                if filename.endswith('.mp4'):
                    file_path = os.path.join(outputs_dir, filename)
                    file_stat = os.stat(file_path)
                    files.append({
                        "filename": filename,
                        "size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                        "created": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                        "download_url": f"http://localhost:5001/download/{filename}",
                        "download_endpoint": f"/download/{filename}"
                    })
        
        return jsonify({
            "status": "success",
            "files": sorted(files, key=lambda x: x['created'], reverse=True),
            "total_files": len(files)
        }), 200
        
    except Exception as e:
        logger.error(f"❌ API: Error listing files: {str(e)}")
        return jsonify({
            "error": str(e),
            "message": "An error occurred while listing files"
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": [
            "GET /health - Health check",
            "POST /process-video - Process YouTube video",
            "GET /download/<filename> - Download processed video",
            "GET /list-files - List all processed videos"
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500

if __name__ == '__main__':
    print("🚀 Starting YouTube AI Narrator API Server...")
    print("📋 Available endpoints:")
    print("   GET  /health           - Health check")
    print("   POST /process-video    - Process YouTube video")
    print("   GET  /download/<file>  - Download processed video")
    print("   GET  /list-files       - List all processed videos")
    print("")
    print("💡 Example request body for /process-video:")
    print('   {"youtube_url": "https://www.youtube.com/watch?v=...", "voice_id": "pNInz6obpgDQGcFmaJgB"}')
    print("")
    print("🌐 API will be available at: http://localhost:5001")
    
    app.run(debug=True, host='0.0.0.0', port=5001) 