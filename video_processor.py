"""
Video Processing Module
Handles video editing using ffmpeg to insert AI narrations between concepts
"""

import os
import subprocess
import tempfile
import logging
from typing import List, Tuple
from pydub import AudioSegment
from narrator import NarrationSegment
from transcription import Segment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles video processing and editing operations"""
    
    def __init__(self, temp_dir: str = None):
        """
        Initialize the video processor
        
        Args:
            temp_dir: Directory for temporary files
        """
        self.temp_dir = temp_dir or tempfile.mkdtemp()
        self.temp_files = []
        
    def add_temp_file(self, filepath: str):
        """Add a file to the cleanup list"""
        self.temp_files.append(filepath)
    
    def create_pause_video(self, duration: float, video_width: int, video_height: int, background_color: str = "black") -> str:
        """
        Create a pause video segment with a solid background
        
        Args:
            duration: Duration of pause in seconds
            video_width: Width of the video
            video_height: Height of the video
            background_color: Background color for pause
            
        Returns:
            Path to pause video file
        """
        pause_video_path = os.path.join(self.temp_dir, f"pause_{duration:.1f}s.mp4")
        
        try:
            # Create a black video using ffmpeg
            cmd = [
                'ffmpeg', '-f', 'lavfi',
                '-i', f'color=c={background_color}:size={video_width}x{video_height}:duration={duration}:rate=30',
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-y',  # Overwrite output file
                pause_video_path
            ]
            
            logger.info(f"Creating pause video: {duration:.1f}s")
            subprocess.run(cmd, check=True, capture_output=True)
            self.add_temp_file(pause_video_path)
            
            return pause_video_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating pause video: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr.decode()}")
            raise
    
    def create_segment_with_faded_audio(self, video_path: str, start_time: float, duration: float, fade_out: bool = False) -> str:
        """
        Create a video segment with optional audio fade
        
        Args:
            video_path: Path to original video
            start_time: Start time of segment
            duration: Duration of segment
            fade_out: Whether to fade out audio at the end
            
        Returns:
            Path to video segment
        """
        fade_suffix = "_fadeout" if fade_out else ""
        output_path = os.path.join(self.temp_dir, f"segment_{start_time:.1f}s{fade_suffix}.mp4")
        
        try:
            if fade_out and duration > 2.0:
                # Fade out audio in the last 1 second
                fade_start = duration - 1.0
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-af', f'afade=t=out:st={fade_start}:d=1',
                    '-c:v', 'copy',
                    '-c:a', 'aac',
                    '-y',
                    output_path
                ]
            else:
                # Regular segment without fade
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-c', 'copy',
                    '-y',
                    output_path
                ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            self.add_temp_file(output_path)
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating segment: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr.decode()}")
            raise
    
    def create_video_segment(self, video_path: str, start_time: float, duration: float) -> str:
        """
        Create a clean video segment from original video with consistent encoding
        
        Args:
            video_path: Path to original video
            start_time: Start time of segment
            duration: Duration of segment
            
        Returns:
            Path to video segment
        """
        output_path = os.path.join(self.temp_dir, f"video_segment_{start_time:.1f}s_to_{start_time + duration:.1f}s.mp4")
        
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-ss', str(start_time),
                '-t', str(duration),
                '-c:v', 'libx264',  # Re-encode video for consistency
                '-c:a', 'aac',      # Re-encode audio to match narration format
                '-b:a', '128k',     # Set audio bitrate
                '-r', '30',         # Set consistent frame rate
                '-pix_fmt', 'yuv420p',  # Consistent pixel format
                '-avoid_negative_ts', 'make_zero',
                '-y',
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            self.add_temp_file(output_path)
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating video segment: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr.decode()}")
            raise
    
    def create_video_with_volume_controlled_narrations(self, video_path: str, narrations: List[NarrationSegment], output_path: str) -> str:
        """
        Create video with seamless audio transitions - original video plays continuously, 
        audio switches between original and AI narration using smart segmentation
        
        Args:
            video_path: Path to original video
            narrations: List of narration segments
            output_path: Path for output video
            
        Returns:
            Path to processed video
        """
        try:
            # Get video duration
            _, _, duration = self.get_video_properties(video_path)
            
            # Create audio timeline segments
            audio_segments = []
            current_time = 0.0
            
            for i, narration in enumerate(narrations):
                narration_start = narration.original_segment.end_time
                
                # 1. Add original audio segment (from current_time to narration_start)
                if narration_start > current_time:
                    original_segment_duration = narration_start - current_time
                    if original_segment_duration > 0.1:  # Only if segment is meaningful
                        original_segment = self.create_audio_segment(
                            video_path, current_time, original_segment_duration, 'original'
                        )
                        audio_segments.append(original_segment)
                        logger.info(f"Added original audio: {current_time:.1f}s - {narration_start:.1f}s")
                
                # 2. Add AI narration segment
                narration_segment = self.create_audio_segment(
                    narration.audio_path, 0, narration.duration, 'narration'
                )
                audio_segments.append(narration_segment)
                logger.info(f"Added AI narration {i+1}: {narration.duration:.1f}s")
                
                current_time = narration_start + narration.duration
            
            # 3. Add remaining original audio (if any)
            if current_time < duration - 0.1:
                remaining_duration = duration - current_time
                remaining_segment = self.create_audio_segment(
                    video_path, current_time, remaining_duration, 'original'
                )
                audio_segments.append(remaining_segment)
                logger.info(f"Added final original audio: {current_time:.1f}s - {duration:.1f}s")
            
            # 4. Concatenate all audio segments
            final_audio = self.concatenate_audio_segments(audio_segments)
            
            # 5. Combine with original video
            cmd = [
                'ffmpeg',
                '-i', video_path,      # Original video
                '-i', final_audio,     # New audio timeline
                '-map', '0:v',         # Use original video
                '-map', '1:a',         # Use new audio
                '-c:v', 'copy',        # Copy video (no re-encoding)
                '-c:a', 'aac',         # Re-encode audio
                '-b:a', '128k',
                '-shortest',           # Match shortest stream
                '-y',
                output_path
            ]
            
            logger.info(f"Creating final video with seamless audio transitions")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            logger.info(f"Seamless video processing complete: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating volume-controlled video: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr}")
            logger.error(f"FFmpeg stdout: {e.stdout}")
            raise

    def create_audio_segment(self, source_path: str, start_time: float, duration: float, segment_type: str) -> str:
        """
        Create an audio segment from source file
        
        Args:
            source_path: Path to source audio/video file
            start_time: Start time in seconds
            duration: Duration in seconds  
            segment_type: 'original' or 'narration'
            
        Returns:
            Path to audio segment
        """
        output_path = os.path.join(self.temp_dir, f"audio_{segment_type}_{start_time:.1f}s_{duration:.1f}s.wav")
        
        try:
            if segment_type == 'narration':
                # For narration files, just copy and convert to WAV
                cmd = [
                    'ffmpeg',
                    '-i', source_path,
                    '-t', str(duration),
                    '-acodec', 'pcm_s16le',
                    '-ar', '44100',
                    '-ac', '2',
                    '-y',
                    output_path
                ]
            else:
                # For original video, extract audio segment
                cmd = [
                    'ffmpeg',
                    '-i', source_path,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-vn',                 # No video
                    '-acodec', 'pcm_s16le',
                    '-ar', '44100',
                    '-ac', '2',
                    '-y',
                    output_path
                ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            self.add_temp_file(output_path)
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating audio segment: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr.decode()}")
            raise

    def concatenate_audio_segments(self, audio_segments: List[str]) -> str:
        """
        Concatenate multiple audio segments into one file
        
        Args:
            audio_segments: List of audio file paths
            
        Returns:
            Path to concatenated audio file
        """
        output_path = os.path.join(self.temp_dir, "final_audio_timeline.wav")
        
        try:
            # Create file list for concatenation
            filelist_path = os.path.join(self.temp_dir, "audio_filelist.txt")
            with open(filelist_path, 'w') as f:
                for segment in audio_segments:
                    f.write(f"file '{os.path.abspath(segment)}'\n")
            
            self.add_temp_file(filelist_path)
            
            # Concatenate audio files
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',  
                '-i', filelist_path,
                '-c', 'copy',
                '-y',
                output_path
            ]
            
            logger.info(f"Concatenating {len(audio_segments)} audio segments")
            subprocess.run(cmd, check=True, capture_output=True)
            self.add_temp_file(output_path)
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error concatenating audio segments: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr.decode()}")
            raise

    def create_video_with_freeze_narrations(self, video_path: str, narrations: List[NarrationSegment], 
                                          output_path: str, width: int, height: int, duration: float) -> str:
        """
        Create video with freeze-and-resume approach
        
        Args:
            video_path: Path to original video
            narrations: List of narration segments
            output_path: Path for output video
            width: Video width
            height: Video height
            duration: Video duration
            
        Returns:
            Path to processed video
        """
        try:
            # Create video sequence: [video_segment] + [freeze_frame] + [video_segment] + [freeze_frame] + ...
            video_segments = []
            last_end_time = 0.0
            
            for i, narration in enumerate(narrations):
                segment = narration.original_segment
                
                # 1. Add original video segment (from last_end_time to segment.end_time)
                if segment.start_time > last_end_time:
                    # Add gap segment if needed
                    gap_duration = segment.start_time - last_end_time
                    if gap_duration > 0.1:
                        gap_segment = self.create_video_segment(
                            video_path, last_end_time, gap_duration
                        )
                        video_segments.append(gap_segment)
                        logger.info(f"Added gap segment: {last_end_time:.1f}s - {segment.start_time:.1f}s")
                
                # Add the concept segment (normal video with audio)
                concept_duration = segment.end_time - segment.start_time
                concept_segment = self.create_video_segment(
                    video_path, segment.start_time, concept_duration
                )
                video_segments.append(concept_segment)
                logger.info(f"Added concept segment {i+1}: {segment.start_time:.1f}s - {segment.end_time:.1f}s")
                
                # 2. Add freeze frame with AI narration
                freeze_frame = self.create_narrated_freeze_frame(
                    video_path, segment.end_time - 0.1, narration.audio_path, 
                    narration.duration, width, height
                )
                video_segments.append(freeze_frame)
                logger.info(f"Added freeze frame with narration {i+1}: {narration.duration:.1f}s")
                
                last_end_time = segment.end_time
            
            # 3. Add remaining video after last concept (if any)
            if last_end_time < duration - 0.1:
                remaining_duration = duration - last_end_time
                remaining_segment = self.create_video_segment(
                    video_path, last_end_time, remaining_duration
                )
                video_segments.append(remaining_segment)
                logger.info(f"Added final segment: {last_end_time:.1f}s - {duration:.1f}s")
            
            # 4. Concatenate all video segments
            logger.info(f"Concatenating {len(video_segments)} video segments")
            self.concatenate_videos(video_segments, output_path)
            
            logger.info(f"Freeze-and-resume video processing complete: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating freeze-and-resume video: {e}")
            raise

    def create_narration_pause(self, narration_audio: str, duration: float, width: int, height: int) -> str:
        """
        Create a pause segment with AI narration (black screen + narration audio)
        
        Args:
            narration_audio: Path to narration audio
            duration: Duration of narration
            width: Video width
            height: Video height
            
        Returns:
            Path to narration pause video
        """
        output_path = os.path.join(self.temp_dir, f"narration_pause_{duration:.1f}s.mp4")
        
        try:
            # Add small padding to narration duration
            total_duration = duration + 0.5
            
            cmd = [
                'ffmpeg',
                '-f', 'lavfi', '-i', f'color=c=black:size={width}x{height}:duration={total_duration}:rate=30',
                '-i', narration_audio,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-pix_fmt', 'yuv420p',
                '-shortest',
                '-y',
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            self.add_temp_file(output_path)
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating narration pause: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr.decode()}")
            raise
    
    def create_narrated_freeze_frame(self, video_path: str, freeze_time: float, narration_audio: str, 
                                   narration_duration: float, video_width: int, video_height: int) -> str:
        """
        Create a video that freezes on a specific frame and plays narration audio
        
        Args:
            video_path: Path to original video
            freeze_time: Time point to freeze at
            narration_audio: Path to narration audio
            narration_duration: Duration of narration
            video_width: Video width
            video_height: Video height
            
        Returns:
            Path to freeze frame video with narration
        """
        output_path = os.path.join(self.temp_dir, f"freeze_narration_{freeze_time:.1f}s.mp4")
        
        try:
            # Step 1: Validate narration audio file
            if not os.path.exists(narration_audio):
                raise FileNotFoundError(f"Narration audio file not found: {narration_audio}")
            
            # Check if audio file has content
            audio_info = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration,size',
                '-of', 'default=noprint_wrappers=1', narration_audio
            ], capture_output=True, text=True)
            
            if audio_info.returncode != 0:
                logger.error(f"Invalid narration audio file: {narration_audio}")
                raise ValueError(f"Cannot read narration audio: {narration_audio}")
            
            # Extract duration and size from ffprobe output
            audio_size = None
            audio_duration = None
            for line in audio_info.stdout.strip().split('\n'):
                if line.startswith('size='):
                    audio_size = int(line.split('=')[1])
                elif line.startswith('duration='):
                    audio_duration = float(line.split('=')[1])
            
            if audio_size is None or audio_size < 1000:  # Less than 1KB is likely empty
                logger.error(f"Narration audio file is too small or empty: {narration_audio} ({audio_size} bytes)")
                raise ValueError(f"Narration audio file appears to be empty: {narration_audio}")
            
            logger.info(f"Validated narration audio: {audio_size} bytes, {audio_duration:.1f}s duration")
            
            # Step 2: Extract a single frame as image
            frame_image = os.path.join(self.temp_dir, f"freeze_frame_{freeze_time:.1f}s.png")
            extract_cmd = [
                'ffmpeg',
                '-ss', str(freeze_time),
                '-i', video_path,
                '-vframes', '1',
                '-y',
                frame_image
            ]
            subprocess.run(extract_cmd, check=True, capture_output=True)
            self.add_temp_file(frame_image)
            
            # Step 3: Create video from frame + narration audio with consistent encoding
            cmd = [
                'ffmpeg',
                '-loop', '1',             # Loop the image input
                '-i', frame_image,        # Input frame image
                '-i', narration_audio,    # Input narration audio
                '-t', str(narration_duration),  # Duration matches narration
                '-vf', f'scale={video_width}:{video_height}',
                '-af', 'pan=stereo|c0=c0|c1=c0',  # Convert mono to stereo (duplicate mono to both channels)
                '-c:v', 'libx264',        # Same video codec as segments
                '-c:a', 'aac',            # Same audio codec as segments
                '-b:a', '128k',           # Same audio bitrate as segments
                '-ac', '2',               # Force stereo output to match original video
                '-r', '30',               # Same frame rate as segments
                '-pix_fmt', 'yuv420p',    # Same pixel format as segments
                '-shortest',              # End when shortest stream ends
                '-avoid_negative_ts', 'make_zero',  # Handle timestamps
                '-y',
                output_path
            ]
            
            logger.info(f"Creating freeze frame with narration: {freeze_time:.1f}s, duration: {narration_duration:.1f}s")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            self.add_temp_file(output_path)
            
            # Step 4: Validate the created freeze frame has audio
            freeze_info = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_streams', '-select_streams', 'a',
                '-of', 'default=noprint_wrappers=1:nokey=1', output_path
            ], capture_output=True, text=True)
            
            if freeze_info.returncode != 0 or not freeze_info.stdout.strip():
                logger.error(f"Created freeze frame has no audio stream: {output_path}")
                logger.error(f"FFmpeg stdout: {result.stdout}")
                logger.error(f"FFmpeg stderr: {result.stderr}")
                raise ValueError(f"Freeze frame creation failed - no audio stream in output")
            else:
                logger.info(f"✅ Freeze frame created successfully with audio stream")
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating freeze frame: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr}")
            logger.error(f"FFmpeg stdout: {e.stdout}")
            raise
    
    def split_video_at_timestamps(self, video_path: str, timestamps: List[float]) -> List[str]:
        """
        Split video at specified timestamps
        
        Args:
            video_path: Path to input video
            timestamps: List of timestamps to split at
            
        Returns:
            List of paths to video segments
        """
        segments = []
        
        try:
            # Add start and end timestamps
            all_timestamps = [0.0] + timestamps
            
            # Get video duration
            result = subprocess.run([
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', video_path
            ], capture_output=True, text=True, check=True)
            
            video_duration = float(result.stdout.strip())
            all_timestamps.append(video_duration)
            
            # Split video into segments
            for i in range(len(all_timestamps) - 1):
                start_time = all_timestamps[i]
                end_time = all_timestamps[i + 1]
                duration = end_time - start_time
                
                if duration <= 0:
                    continue
                
                segment_path = os.path.join(self.temp_dir, f"segment_{i:03d}.mp4")
                
                cmd = [
                    'ffmpeg',
                    '-i', video_path,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-c', 'copy',
                    '-avoid_negative_ts', 'make_zero',
                    '-y',
                    segment_path
                ]
                
                logger.info(f"Creating segment {i+1}: {start_time:.1f}s - {end_time:.1f}s")
                subprocess.run(cmd, check=True, capture_output=True)
                
                segments.append(segment_path)
                self.add_temp_file(segment_path)
            
            logger.info(f"Created {len(segments)} video segments")
            return segments
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error splitting video: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr.decode()}")
            raise
        except Exception as e:
            logger.error(f"Error splitting video: {str(e)}")
            raise
    
    def get_video_properties(self, video_path: str) -> Tuple[int, int, float]:
        """
        Get video properties (width, height, duration)
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (width, height, duration)
        """
        try:
            # Get video properties using ffprobe
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            import json
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in data['streams']:
                if stream['codec_type'] == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            duration = float(data['format']['duration'])
            
            return width, height, duration
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error getting video properties: {e}")
            raise
        except Exception as e:
            logger.error(f"Error parsing video properties: {str(e)}")
            raise
    
    def concatenate_videos(self, video_paths: List[str], output_path: str) -> str:
        """
        Concatenate multiple videos into one with precise audio sync
        
        Args:
            video_paths: List of video file paths
            output_path: Output file path
            
        Returns:
            Path to concatenated video
        """
        try:
            # Create a file list for ffmpeg concat
            filelist_path = os.path.join(self.temp_dir, "filelist.txt")
            with open(filelist_path, 'w') as f:
                for video_path in video_paths:
                    f.write(f"file '{os.path.abspath(video_path)}'\n")
            
            self.add_temp_file(filelist_path)
            
            # Concatenate videos with precise audio sync
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', filelist_path,
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',  # Handle timestamp alignment
                '-fflags', '+genpts',               # Generate new timestamps
                '-y',
                output_path
            ]
            
            logger.info(f"Concatenating {len(video_paths)} videos with audio sync")
            subprocess.run(cmd, check=True, capture_output=True)
            
            logger.info(f"Video concatenation complete: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error concatenating videos: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr.decode()}")
            raise
    
    def concatenate_videos_with_reencoding(self, video_paths: List[str], output_path: str) -> str:
        """
        Concatenate multiple videos with re-encoding for smooth audio transitions
        
        Args:
            video_paths: List of video file paths
            output_path: Output file path
            
        Returns:
            Path to concatenated video
        """
        try:
            # Create a file list for ffmpeg concat
            filelist_path = os.path.join(self.temp_dir, "filelist.txt")
            with open(filelist_path, 'w') as f:
                for video_path in video_paths:
                    f.write(f"file '{os.path.abspath(video_path)}'\n")
            
            self.add_temp_file(filelist_path)
            
            # Concatenate videos with re-encoding for smooth transitions
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', filelist_path,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-b:a', '128k',
                '-ar', '44100',
                '-pix_fmt', 'yuv420p',
                '-y',
                output_path
            ]
            
            logger.info(f"Concatenating {len(video_paths)} videos with re-encoding")
            subprocess.run(cmd, check=True, capture_output=True)
            
            logger.info(f"Video concatenation complete: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Error concatenating videos: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr.decode()}")
            raise
    
    def process_video_with_narrations(self, video_path: str, narrations: List[NarrationSegment], output_path: str) -> str:
        """
        Process video with freeze-and-resume approach:
        1. Original video plays normally until concept ends
        2. Video freezes on last frame during AI narration  
        3. Video resumes from exact point after narration
        
        Args:
            video_path: Path to original video
            narrations: List of narration segments  
            output_path: Path for output video
            
        Returns:
            Path to processed video
        """
        try:
            logger.info("Starting freeze-and-resume video processing with narrations")
            
            # Get video properties
            width, height, duration = self.get_video_properties(video_path)
            logger.info(f"Video properties: {width}x{height}, {duration:.1f}s")
            
            # Create video segments with freeze frames for narrations
            return self.create_video_with_freeze_narrations(video_path, narrations, output_path, width, height, duration)
            
        except Exception as e:
            logger.error(f"Error processing video with narrations: {str(e)}")
            raise
    

    
    def cleanup(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up: {temp_file}")
            except Exception as e:
                logger.warning(f"Error cleaning up {temp_file}: {str(e)}")
        
        # Try to remove temp directory if empty
        try:
            if os.path.exists(self.temp_dir) and not os.listdir(self.temp_dir):
                os.rmdir(self.temp_dir)
                logger.debug(f"Removed temp directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Error removing temp directory: {str(e)}")


def process_video_with_ai_narrations(video_path: str, narrations: List[NarrationSegment], output_path: str) -> str:
    """
    Convenience function to process video with AI narrations
    
    Args:
        video_path: Path to original video
        narrations: List of narration segments
        output_path: Path for output video
        
    Returns:
        Path to processed video
    """
    processor = VideoProcessor()
    try:
        return processor.process_video_with_narrations(video_path, narrations, output_path)
    finally:
        processor.cleanup()


if __name__ == "__main__":
    # Test the video processor
    test_video = "test_video.mp4"  # Replace with actual video file
    if os.path.exists(test_video):
        try:
            processor = VideoProcessor()
            
            # Get video properties
            width, height, duration = processor.get_video_properties(test_video)
            print(f"Video properties: {width}x{height}, {duration:.1f}s")
            
            # Test splitting video
            timestamps = [30.0, 60.0, 90.0]  # Example timestamps
            segments = processor.split_video_at_timestamps(test_video, timestamps)
            print(f"Created {len(segments)} segments")
            
            # Cleanup
            processor.cleanup()
            
        except Exception as e:
            print(f"Error testing video processor: {e}")
    else:
        print(f"Test video file not found: {test_video}")
        print("Make sure ffmpeg is installed and available in PATH") 