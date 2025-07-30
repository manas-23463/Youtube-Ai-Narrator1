"""
Transcription and Segmentation Module
Uses OpenAI Whisper to transcribe audio and segment it into logical concepts
"""

import whisper
import os
from typing import List, Dict, Tuple
import logging
import re
import numpy as np
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Segment:
    """Represents a logical segment/concept in the transcription"""
    start_time: float
    end_time: float
    text: str
    confidence: float = 0.0
    concept_title: str = ""


class TranscriptionSegmenter:
    """Handles transcription and intelligent segmentation of audio content"""
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize the transcription segmenter
        
        Args:
            model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
        """
        self.model_size = model_size
        self.model = None
        self.segments = []
        
    def load_model(self):
        """Load the Whisper model"""
        if self.model is None:
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """
        Transcribe audio file using Whisper
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Whisper transcription result
        """
        self.load_model()
        
        if not os.path.exists(audio_path):
            logger.error(f"[CHECK] Audio file does NOT exist before transcription: {audio_path}")
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        else:
            logger.info(f"[CHECK] Audio file exists before transcription: {audio_path}")
        logger.info(f"Transcribing audio: {audio_path}")
        result = self.model.transcribe(
            audio_path,
            word_timestamps=True,
            verbose=False
        )
        
        logger.info(f"Transcription completed. Found {len(result['segments'])} segments")
        return result
    
    def _detect_concept_boundaries(self, segments: List[Dict]) -> List[int]:
        """
        Detect logical concept boundaries in the transcription
        Uses heuristics like pauses, topic changes, and sentence structure
        
        Args:
            segments: Whisper segments with word-level timestamps
            
        Returns:
            List of segment indices where new concepts begin
        """
        boundaries = [0]  # Always start with first segment
        
        for i in range(1, len(segments)):
            current_segment = segments[i]
            previous_segment = segments[i-1]
            
            # Calculate pause duration between segments
            pause_duration = current_segment['start'] - previous_segment['end']
            
            # Detect boundaries based on multiple criteria
            is_boundary = False
            
            # 1. Long pause (>2 seconds indicates concept change)
            if pause_duration > 2.0:
                is_boundary = True
                logger.debug(f"Long pause detected at {current_segment['start']:.2f}s")
            
            # 2. Sentence endings followed by new topics
            prev_text = previous_segment['text'].strip()
            curr_text = current_segment['text'].strip()
            
            if (prev_text.endswith('.') or prev_text.endswith('!') or prev_text.endswith('?')) and pause_duration > 1.0:
                # Check for topic change indicators
                topic_indicators = [
                    'now', 'next', 'so', 'okay', 'alright', 'moving on',
                    'let\'s', 'another', 'also', 'furthermore', 'however',
                    'but', 'on the other hand', 'in contrast', 'meanwhile'
                ]
                
                if any(curr_text.lower().startswith(indicator) for indicator in topic_indicators):
                    is_boundary = True
                    logger.debug(f"Topic change detected at {current_segment['start']:.2f}s")
            
            # 3. Significant change in speaking pace or volume (approximated by text length changes)
            if len(curr_text) > 100 and len(prev_text) < 50:
                is_boundary = True
                logger.debug(f"Speaking pattern change at {current_segment['start']:.2f}s")
            
            # 4. Question to statement transition (or vice versa)
            if prev_text.endswith('?') and not curr_text.endswith('?') and pause_duration > 0.8:
                is_boundary = True
                logger.debug(f"Q&A transition at {current_segment['start']:.2f}s")
            
            if is_boundary:
                boundaries.append(i)
        
        # Ensure we don't have too many or too few segments
        # Aim for concepts every 30-120 seconds
        filtered_boundaries = self._filter_boundaries_by_duration(segments, boundaries)
        
        logger.info(f"Detected {len(filtered_boundaries)} concept boundaries")
        return filtered_boundaries
    
    def _filter_boundaries_by_duration(self, segments: List[Dict], boundaries: List[int]) -> List[int]:
        """
        Filter boundaries to ensure reasonable concept durations
        
        Args:
            segments: Whisper segments
            boundaries: Detected boundary indices
            
        Returns:
            Filtered boundary indices
        """
        if len(boundaries) <= 1:
            return boundaries
        
        filtered = [boundaries[0]]  # Always keep the first boundary
        
        for i in range(1, len(boundaries)):
            prev_boundary_idx = filtered[-1]
            curr_boundary_idx = boundaries[i]
            
            # Calculate duration of current concept
            start_time = segments[prev_boundary_idx]['start']
            end_time = segments[curr_boundary_idx]['start']
            duration = end_time - start_time
            
            # Only keep boundary if concept is long enough (min 15 seconds)
            if duration >= 15.0:
                filtered.append(curr_boundary_idx)
            else:
                logger.debug(f"Skipping short concept ({duration:.1f}s) at {start_time:.2f}s")
        
        return filtered
    
    def segment_transcription(self, audio_path: str) -> List[Segment]:
        """
        Transcribe audio and segment it into logical concepts
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            List of concept segments
        """
        # Transcribe the audio
        result = self.transcribe_audio(audio_path)
        
        # Detect concept boundaries
        boundaries = self._detect_concept_boundaries(result['segments'])
        
        # Create concept segments
        concepts = []
        for i in range(len(boundaries)):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1] if i + 1 < len(boundaries) else len(result['segments'])
            
            # Combine text from all segments in this concept
            concept_segments = result['segments'][start_idx:end_idx]
            combined_text = ' '.join([seg['text'].strip() for seg in concept_segments])
            
            # Calculate concept timing
            start_time = concept_segments[0]['start']
            end_time = concept_segments[-1]['end']
            
            # Calculate average confidence
            avg_confidence = np.mean([seg.get('confidence', 1.0) for seg in concept_segments])
            
            # Generate concept title (first few words)
            concept_title = self._generate_concept_title(combined_text)
            
            segment = Segment(
                start_time=start_time,
                end_time=end_time,
                text=combined_text,
                confidence=avg_confidence,
                concept_title=concept_title
            )
            
            concepts.append(segment)
            logger.info(f"Concept {i+1}: '{concept_title}' ({start_time:.1f}s - {end_time:.1f}s)")
        
        self.segments = concepts
        return concepts
    
    def _generate_concept_title(self, text: str) -> str:
        """
        Generate a short title for a concept based on its content
        
        Args:
            text: Full text of the concept
            
        Returns:
            Short concept title
        """
        # Clean and truncate text
        clean_text = re.sub(r'[^\w\s]', '', text.strip())
        words = clean_text.split()
        
        # Take first 3-5 meaningful words
        title_words = []
        for word in words[:8]:
            if len(word) > 2 and word.lower() not in ['the', 'and', 'but', 'for', 'are', 'will', 'this', 'that']:
                title_words.append(word)
            if len(title_words) >= 4:
                break
        
        title = ' '.join(title_words[:4])
        return title if len(title) > 0 else "Concept"
    
    def get_segments(self) -> List[Segment]:
        """Get the current segments"""
        return self.segments
    
    def export_segments_to_dict(self) -> List[Dict]:
        """Export segments as a list of dictionaries"""
        return [
            {
                'start_time': seg.start_time,
                'end_time': seg.end_time,
                'text': seg.text,
                'confidence': seg.confidence,
                'concept_title': seg.concept_title
            }
            for seg in self.segments
        ]


def transcribe_and_segment(audio_path: str, model_size: str = "base") -> List[Segment]:
    """
    Convenience function to transcribe and segment audio
    
    Args:
        audio_path: Path to audio file
        model_size: Whisper model size
        
    Returns:
        List of concept segments
    """
    segmenter = TranscriptionSegmenter(model_size)
    return segmenter.segment_transcription(audio_path)


if __name__ == "__main__":
    # Test the transcription segmenter
    test_audio = "test_audio.wav"  # Replace with actual audio file
    if os.path.exists(test_audio):
        try:
            segments = transcribe_and_segment(test_audio)
            print(f"Found {len(segments)} concepts:")
            for i, seg in enumerate(segments):
                print(f"{i+1}. {seg.concept_title} ({seg.start_time:.1f}s - {seg.end_time:.1f}s)")
                print(f"   Text: {seg.text[:100]}...")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Test audio file not found: {test_audio}") 