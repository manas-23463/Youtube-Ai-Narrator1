#!/usr/bin/env python3
"""
YouTube AI Narrator - Main Script
Downloads YouTube videos, transcribes them, generates AI explanations, and creates narrated videos
"""

import os
import sys
import argparse
import logging
import tempfile
from datetime import datetime
from dotenv import load_dotenv

# Import our modules
from youtube_downloader import YouTubeDownloader
from transcription import TranscriptionSegmenter
from narrator import ConceptNarrator
from video_processor import VideoProcessor

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('youtube_ai_narrator.log')
    ]
)
logger = logging.getLogger(__name__)


class YouTubeAINarrator:
    """Main class that orchestrates the entire YouTube AI Narrator process"""
    
    def __init__(self, voice_id: str, whisper_model: str = "base", output_dir: str = None):
        """
        Initialize the YouTube AI Narrator
        
        Args:
            voice_id: ElevenLabs voice ID
            whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            output_dir: Output directory for files
        """
        self.voice_id = voice_id
        self.whisper_model = whisper_model
        self.output_dir = output_dir or os.getcwd()
        
        # Create temporary directory for processing
        self.temp_dir = tempfile.mkdtemp(prefix='youtube_ai_narrator_')
        
        # Initialize components
        self.downloader = None
        self.transcriber = None
        self.narrator = None
        self.video_processor = None
        
        # Check API keys
        self._check_api_keys()
        
    def _check_api_keys(self):
        """Check if required API keys are available"""
        required_keys = ['OPENAI_API_KEY', 'ELEVENLABS_API_KEY']
        missing_keys = []
        
        for key in required_keys:
            if not os.getenv(key):
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_keys)}")
        
        logger.info("API keys verified successfully")
    
    def process_video(self, youtube_url: str, output_filename: str = None) -> str:
        """
        Process a YouTube video with AI narrations
        
        Args:
            youtube_url: YouTube video URL
            output_filename: Output filename (if None, generates from video title)
            
        Returns:
            Path to the processed video file
        """
        try:
            logger.info(f"Starting YouTube AI Narrator processing for: {youtube_url}")
            
            # Step 1: Download video
            logger.info("Step 1: Downloading YouTube video...")
            self.downloader = YouTubeDownloader(self.temp_dir)
            video_path, audio_path, video_title = self.downloader.download_video(youtube_url)
            
            video_title = video_title.replace('/', '_').replace('\\', '_')
            logger.info(f"Downloaded: {video_title}")
            
            # Step 2: Transcribe and segment audio
            logger.info("Step 2: Transcribing and segmenting audio...")
            self.transcriber = TranscriptionSegmenter(self.whisper_model)
            segments = self.transcriber.segment_transcription(audio_path)
            
            if not segments:
                raise ValueError("No segments found in transcription")
            
            logger.info(f"Found {len(segments)} concept segments")
            
            # Step 3: Generate AI narrations
            logger.info("Step 3: Generating AI narrations...")
            self.narrator = ConceptNarrator(voice_id=self.voice_id)
            narrations = self.narrator.create_narrations(segments, video_title)
            
            if not narrations:
                raise ValueError("No narrations generated")
            
            logger.info(f"Generated {len(narrations)} narrations")
            
            # Step 4: Process video with narrations
            logger.info("Step 4: Processing video with narrations...")
            self.video_processor = VideoProcessor(self.temp_dir)
            
            # Generate output filename if not provided
            if not output_filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_title = safe_title[:50]  # Limit length
                output_filename = f"{safe_title}_AI_Narrated_{timestamp}.mp4"
            
            output_path = os.path.join(self.output_dir, output_filename)
            
            # Process the video
            final_video = self.video_processor.process_video_with_narrations(
                video_path, narrations, output_path
            )
            
            logger.info(f"✅ Processing complete! Output: {final_video}")
            
            # Print summary
            self._print_summary(video_title, segments, narrations, final_video)
            
            return final_video
            
        except Exception as e:
            logger.error(f"❌ Error processing video: {str(e)}")
            raise
        finally:
            # Cleanup
            self.cleanup()
    
    def _print_summary(self, video_title: str, segments, narrations, output_path: str):
        """Print a summary of the processing"""
        print("\n" + "="*80)
        print("🎬 YOUTUBE AI NARRATOR - PROCESSING COMPLETE")
        print("="*80)
        print(f"📹 Original Video: {video_title}")
        print(f"🎯 Concepts Found: {len(segments)}")
        print(f"🎙️  Narrations Generated: {len(narrations)}")
        print(f"📁 Output File: {output_path}")
        print("="*80)
        
        print("\n📝 CONCEPT BREAKDOWN:")
        for i, (segment, narration) in enumerate(zip(segments, narrations), 1):
            print(f"  {i}. {segment.concept_title}")
            print(f"     ⏱️  Time: {segment.start_time:.1f}s - {segment.end_time:.1f}s")
            print(f"     💬 Explanation: {narration.explanation_text[:100]}...")
            print()
        
        print("🚀 You can now play your AI-narrated video!")
        print("="*80)
    
    def cleanup(self):
        """Clean up temporary files and resources"""
        try:
            # Cleanup narrator
            if self.narrator:
                self.narrator.cleanup_narrations()
            
            # Cleanup video processor
            if self.video_processor:
                self.video_processor.cleanup()
            
            # Cleanup downloader
            if self.downloader:
                self.downloader.cleanup()
            
            # Remove temp directory
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
                logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
            
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="YouTube AI Narrator - Add AI explanations to educational videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "https://www.youtube.com/watch?v=example"
  python main.py "https://youtu.be/example" -m large
  python main.py "https://youtu.be/example" "custom_voice_id" -m small -q
  
Environment Variables Required:
  OPENAI_API_KEY      - Your OpenAI API key
  ELEVENLABS_API_KEY  - Your ElevenLabs API key

Note: Make sure ffmpeg is installed and available in your PATH.
      Default voice is Rachel (pNInz6obpgDQGcFmaJgB), outputs go to ./outputs/
        """
    )
    
    parser.add_argument(
        'youtube_url',
        help='YouTube video URL to process'
    )
    
    parser.add_argument(
        'voice_id',
        nargs='?',
        default='pNInz6obpgDQGcFmaJgB',  # Rachel voice as default
        help='ElevenLabs voice ID for narration (default: Rachel voice)'
    )
    
    parser.add_argument(
        '--model', '-m',
        choices=['tiny', 'base', 'small', 'medium', 'large'],
        default='base',
        help='Whisper model size (default: base)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        default='./outputs',
        help='Output directory for the processed video (default: ./outputs)'
    )
    
    parser.add_argument(
        '--output-filename', '-f',
        help='Custom output filename (optional)'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Disable verbose logging (verbose is default)'
    )
    
    args = parser.parse_args()
    
    # Set logging level (verbose by default, quiet to disable)
    if not args.quiet:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not args.youtube_url:
        print("❌ Error: YouTube URL is required")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Initialize and run the narrator
        narrator = YouTubeAINarrator(
            voice_id=args.voice_id,
            whisper_model=args.model,
            output_dir=args.output_dir
        )
        
        output_file = narrator.process_video(
            youtube_url=args.youtube_url,
            output_filename=args.output_filename
        )
        
        print(f"\n🎉 Success! Your AI-narrated video is ready: {output_file}")
        
    except KeyboardInterrupt:
        print("\n⏹️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        logger.exception("Full error details:")
        sys.exit(1)


if __name__ == "__main__":
    main() 