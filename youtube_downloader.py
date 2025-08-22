"""
YouTube Video Downloader Module
Downloads YouTube videos with audio and video streams
"""

import os
import yt_dlp
import tempfile
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YouTubeDownloader:
    """Downloads YouTube videos and extracts audio/video information"""
    
    def __init__(self, output_dir: str = None, cookies_file: str = None):
        """
        Initialize the YouTube downloader
        
        Args:
            output_dir: Directory to save downloaded files. If None, uses temp directory
            cookies_file: Path to cookies file exported from browser (optional)
        """
        self.output_dir = output_dir or tempfile.mkdtemp()
        self.cookies_file = cookies_file
        self.video_path = None
        self.audio_path = None
        self.video_info = None
        
    def download_video(self, url: str) -> Tuple[str, str, str]:
        """
        Download YouTube video and extract audio
        
        Args:
            url: YouTube video URL
            
        Returns:
            Tuple of (video_path, audio_path, video_title)
        """
        try:
            # Configure yt-dlp options with better bot detection handling
            ydl_opts = {
                'format': 'best[ext=mp4]',
                'outtmpl': os.path.join(self.output_dir, '%(title)s.%(ext)s'),
                'writesubtitles': False,
                'writeautomaticsub': False,
                # Add user agent to avoid bot detection
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                },
                # Add retry options
                'retries': 5,
                'fragment_retries': 5,
                # Add age verification bypass
                'age_limit': 0,
                # Add geo bypass
                'geo_bypass': True,
                'geo_bypass_country': 'US',
                # Add additional extractor args
                'extractor_args': {
                    'youtube': {
                        'skip': ['dash', 'live'],
                        'player_client': ['web'],
                        'player_skip': ['webpage', 'configs'],
                    }
                }
            }
            
            # Add cookies file if provided
            if self.cookies_file and os.path.exists(self.cookies_file):
                ydl_opts['cookiefile'] = self.cookies_file
                logger.info(f"Using cookies file: {self.cookies_file}")
            
            # Download video
            logger.info(f"Downloading video from: {url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract video info
                info = ydl.extract_info(url, download=False)
                self.video_info = info
                
                # Download the video
                ydl.download([url])
                
                # Get the downloaded video path
                video_filename = ydl.prepare_filename(info)
                self.video_path = video_filename
                
            # Extract audio using yt-dlp with same options
            audio_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(self.output_dir, '%(title)s_audio.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                # Add same headers and options for audio
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                },
                'retries': 5,
                'fragment_retries': 5,
                'age_limit': 0,
                'geo_bypass': True,
                'geo_bypass_country': 'US',
                'extractor_args': {
                    'youtube': {
                        'skip': ['dash', 'live'],
                        'player_client': ['web'],
                        'player_skip': ['webpage', 'configs'],
                    }
                }
            }
            
            # Add cookies file if provided
            if self.cookies_file and os.path.exists(self.cookies_file):
                audio_opts['cookiefile'] = self.cookies_file
            
            logger.info("Extracting audio...")
            with yt_dlp.YoutubeDL(audio_opts) as ydl:
                ydl.download([url])
                # The audio template already includes _audio, so we construct the correct path
                import re
                clean_title = info['title'].replace('/', '_').replace('\\', '_')
                # Remove non-ASCII characters
                clean_title = re.sub(r'[^\x00-\x7F]+', '_', clean_title)
                audio_filename = os.path.join(self.output_dir, f"{clean_title}_audio.wav")
                self.audio_path = audio_filename
                # Log file existence
                if os.path.exists(self.audio_path):
                    logger.info(f"[CHECK] Audio file exists: {self.audio_path}")
                else:
                    logger.error(f"[CHECK] Audio file MISSING: {self.audio_path}")
                    # Fallback: search for any _audio.wav file in output dir
                    for fname in os.listdir(self.output_dir):
                        if fname.endswith('_audio.wav'):
                            fallback_path = os.path.join(self.output_dir, fname)
                            logger.warning(f"[FALLBACK] Using found audio file: {fallback_path}")
                            self.audio_path = fallback_path
                            break
                # Log all files in output directory
                logger.info(f"[DEBUG] All files in output dir: {os.listdir(self.output_dir)}")
                logger.info(f"[DEBUG] Expected audio filename: {self.audio_path}")
            
            logger.info(f"Video downloaded: {self.video_path}")
            logger.info(f"Audio extracted: {self.audio_path}")
            
            return self.video_path, self.audio_path, self.video_info.get('title', 'Unknown Title')
            
        except Exception as e:
            logger.error(f"Error downloading video: {str(e)}")
            # If the first attempt fails, try with a different approach
            if "Sign in to confirm you're not a bot" in str(e) or "could not find chrome cookies" in str(e):
                logger.info("Bot detection or cookie issue detected, trying alternative approach...")
                return self._download_with_alternative_method(url)
            raise
    
    def _download_with_alternative_method(self, url: str) -> Tuple[str, str, str]:
        """Alternative download method when bot detection occurs"""
        try:
            # Try with different user agent and more robust settings
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': os.path.join(self.output_dir, '%(title)s.%(ext)s'),
                'writesubtitles': False,
                'writeautomaticsub': False,
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-us,en;q=0.5',
                    'Accept-Encoding': 'gzip,deflate',
                    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7',
                    'Connection': 'keep-alive',
                },
                'retries': 10,
                'fragment_retries': 10,
                'age_limit': 0,
                'geo_bypass': True,
                'geo_bypass_country': 'US',
                'extractor_args': {
                    'youtube': {
                        'player_client': ['web', 'android'],
                        'player_skip': ['webpage'],
                    }
                },
                'socket_timeout': 30,
                'extractor_retries': 3,
            }
            
            logger.info("Trying alternative download method...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                self.video_info = info
                ydl.download([url])
                video_filename = ydl.prepare_filename(info)
                self.video_path = video_filename
            
            # Extract audio
            audio_opts = {
                'format': 'bestaudio/best',
                'outtmpl': os.path.join(self.output_dir, '%(title)s_audio.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-us,en;q=0.5',
                    'Accept-Encoding': 'gzip,deflate',
                    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.7',
                    'Connection': 'keep-alive',
                },
                'retries': 10,
                'fragment_retries': 10,
                'age_limit': 0,
                'geo_bypass': True,
                'geo_bypass_country': 'US',
                'extractor_args': {
                    'youtube': {
                        'player_client': ['web', 'android'],
                        'player_skip': ['webpage'],
                    }
                },
                'socket_timeout': 30,
                'extractor_retries': 3,
            }
            
            with yt_dlp.YoutubeDL(audio_opts) as ydl:
                ydl.download([url])
                clean_title = info['title'].replace('/', '_').replace('\\', '_')
                audio_filename = os.path.join(self.output_dir, f"{clean_title}_audio.wav")
                self.audio_path = audio_filename
            
            logger.info(f"Alternative method successful: {self.video_path}")
            return self.video_path, self.audio_path, self.video_info.get('title', 'Unknown Title')
            
        except Exception as e:
            logger.error(f"Alternative download method also failed: {str(e)}")
            # Try one more method with different approach
            if "Sign in to confirm you're not a bot" in str(e):
                logger.info("Trying final fallback method...")
                return self._download_with_final_fallback(url)
            raise
    
    def _download_with_final_fallback(self, url: str) -> Tuple[str, str, str]:
        """Final fallback method using different yt-dlp configuration"""
        try:
            # Use a completely different approach
            ydl_opts = {
                'format': 'worst[ext=mp4]/worst',  # Try worst quality first
                'outtmpl': os.path.join(self.output_dir, '%(title)s.%(ext)s'),
                'writesubtitles': False,
                'writeautomaticsub': False,
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': '*/*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                },
                'retries': 15,
                'fragment_retries': 15,
                'age_limit': 0,
                'geo_bypass': True,
                'geo_bypass_country': 'US',
                'extractor_args': {
                    'youtube': {
                        'player_client': ['web'],
                        'player_skip': [],
                        'skip': [],
                    }
                },
                'socket_timeout': 60,
                'extractor_retries': 5,
                'ignoreerrors': True,
            }
            
            logger.info("Trying final fallback download method...")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if not info:
                    raise Exception("Could not extract video info")
                self.video_info = info
                ydl.download([url])
                video_filename = ydl.prepare_filename(info)
                self.video_path = video_filename
            
            # Extract audio with same approach
            audio_opts = {
                'format': 'worstaudio/worst',
                'outtmpl': os.path.join(self.output_dir, '%(title)s_audio.%(ext)s'),
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                    'Accept': '*/*',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br',
                    'DNT': '1',
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                },
                'retries': 15,
                'fragment_retries': 15,
                'age_limit': 0,
                'geo_bypass': True,
                'geo_bypass_country': 'US',
                'extractor_args': {
                    'youtube': {
                        'player_client': ['web'],
                        'player_skip': [],
                        'skip': [],
                    }
                },
                'socket_timeout': 60,
                'extractor_retries': 5,
                'ignoreerrors': True,
            }
            
            with yt_dlp.YoutubeDL(audio_opts) as ydl:
                ydl.download([url])
                clean_title = info['title'].replace('/', '_').replace('\\', '_')
                audio_filename = os.path.join(self.output_dir, f"{clean_title}_audio.wav")
                self.audio_path = audio_filename
            
            logger.info(f"Final fallback method successful: {self.video_path}")
            return self.video_path, self.audio_path, self.video_info.get('title', 'Unknown Title')
            
        except Exception as e:
            logger.error(f"All download methods failed: {str(e)}")
            raise Exception(f"Unable to download video after trying multiple methods. YouTube may have blocked automated access to this video. Error: {str(e)}")
    
    def get_video_duration(self) -> Optional[float]:
        """Get video duration in seconds"""
        if self.video_info:
            return self.video_info.get('duration', None)
        return None
    
    def get_video_title(self) -> Optional[str]:
        """Get video title"""
        if self.video_info:
            return self.video_info.get('title', 'Unknown Title')
        return None
    
    def extract_video_chapters(self, url: str):
        """
        Extract chapter information from a YouTube video without downloading
        
        Args:
            url: YouTube video URL
            
        Returns:
            List of chapter dictionaries with 'start_time', 'end_time', and 'title'
        """
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True,
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                },
                'retries': 3,
                'fragment_retries': 3,
                'age_limit': 0,
                'geo_bypass': True,
                'geo_bypass_country': 'US',
            }
            
            # Add cookies file if provided
            if self.cookies_file and os.path.exists(self.cookies_file):
                ydl_opts['cookiefile'] = self.cookies_file
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                chapters = info.get('chapters', [])
                if chapters:
                    logger.info(f"Found {len(chapters)} chapters in video")
                    return chapters
                else:
                    logger.info("No chapters found in video")
                    return []
                    
        except Exception as e:
            logger.error(f"Error extracting chapters: {str(e)}")
            return []

    def get_video_info_without_download(self, url: str):
        """
        Get video information without downloading the actual video
        
        Args:
            url: YouTube video URL
            
        Returns:
            Video info dictionary or None if failed
        """
        try:
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'http_headers': {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                },
                'retries': 3,
                'fragment_retries': 3,
                'age_limit': 0,
                'geo_bypass': True,
                'geo_bypass_country': 'US',
            }
            
            # Add cookies file if provided
            if self.cookies_file and os.path.exists(self.cookies_file):
                ydl_opts['cookiefile'] = self.cookies_file
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info
                    
        except Exception as e:
            logger.error(f"Error getting video info: {str(e)}")
            return None
    
    def cleanup(self):
        """Clean up downloaded files"""
        try:
            if self.video_path and os.path.exists(self.video_path):
                os.remove(self.video_path)
                logger.info(f"Cleaned up video: {self.video_path}")
                
            if self.audio_path and os.path.exists(self.audio_path):
                os.remove(self.audio_path)
                logger.info(f"Cleaned up audio: {self.audio_path}")
                
        except Exception as e:
            logger.warning(f"Error during cleanup: {str(e)}")


def download_youtube_video(url: str, output_dir: str = None, cookies_file: str = None) -> Tuple[str, str, str]:
    """
    Convenience function to download a YouTube video
    
    Args:
        url: YouTube video URL
        output_dir: Output directory for downloaded files
        cookies_file: Path to cookies file exported from browser (optional)
        
    Returns:
        Tuple of (video_path, audio_path, video_title)
    """
    downloader = YouTubeDownloader(output_dir, cookies_file)
    return downloader.download_video(url)


if __name__ == "__main__":
    # Test the downloader
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Example URL
    try:
        video_path, audio_path, title = download_youtube_video(test_url)
        print(f"Downloaded: {video_path}")
        print(f"Audio: {audio_path}")
        print(f"Title: {title}")
    except Exception as e:
        print(f"Error: {e}") 