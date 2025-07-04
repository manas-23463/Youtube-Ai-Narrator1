"""
AI Narrator Module
Generates explanations using OpenAI GPT-4 and synthesizes them using ElevenLabs TTS
"""

import os
import openai
import requests
import tempfile
from typing import List, Optional
import logging
from dataclasses import dataclass
from transcription import Segment
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class NarrationSegment:
    """Represents a generated narration for a concept"""
    original_segment: Segment
    explanation_text: str
    audio_path: str
    duration: float = 0.0


class AIExplainer:
    """Generates explanations for concepts using OpenAI GPT-4"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the AI explainer
        
        Args:
            api_key: OpenAI API key (if None, uses environment variable)
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
        
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
    
    def generate_explanation(self, concept_text: str, video_title: str = "") -> str:
        """
        Generate a simple explanation for a concept using GPT-4
        
        Args:
            concept_text: The original concept text from transcription
            video_title: Title of the video for context
            
        Returns:
            AI-generated explanation text
        """
        try:
            # Create a prompt that asks for a simple explanation
            prompt = self._create_explanation_prompt(concept_text, video_title)
            
            logger.info(f"Generating explanation for concept: {concept_text[:50]}...")
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an educational AI assistant that explains complex concepts in simple, clear terms. Your explanations should be concise (30-60 seconds when spoken), engaging, and accessible to a general audience."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            explanation = response.choices[0].message.content.strip()
            logger.info(f"Generated explanation: {explanation[:100]}...")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            # Fallback explanation
            return f"Let me explain this concept in simpler terms: {concept_text[:100]}..."
    
    def _create_explanation_prompt(self, concept_text: str, video_title: str = "") -> str:
        """
        Create a prompt for GPT-4 to generate explanations
        
        Args:
            concept_text: Original concept text
            video_title: Video title for context
            
        Returns:
            Formatted prompt
        """
        context = f"from the video '{video_title}'" if video_title else ""
        
        prompt = f"""
The following is a transcript segment {context}:

"{concept_text}"

Please provide a clear, simple explanation of the key concept or idea presented in this segment. Your explanation should:

1. Be written in plain English that anyone can understand
2. Focus on the main idea, not just repeat what was said
3. Be concise (30-60 seconds when spoken aloud)
4. Use analogies or examples if helpful
5. Avoid jargon and technical terms unless necessary
6. Start with phrases like "In simple terms..." or "What this means is..." or "To put it simply..."

Write your explanation as if you're a helpful teacher explaining this concept to a student who needs clarification.
"""
        return prompt


class ElevenLabsTTS:
    """Synthesizes text to speech using ElevenLabs API"""
    
    def __init__(self, api_key: str = None, voice_id: str = None):
        """
        Initialize ElevenLabs TTS
        
        Args:
            api_key: ElevenLabs API key (if None, uses environment variable)
            voice_id: Voice ID to use for synthesis
        """
        self.api_key = api_key or os.getenv('ELEVENLABS_API_KEY')
        if not self.api_key:
            raise ValueError("ElevenLabs API key not provided. Set ELEVENLABS_API_KEY environment variable.")
        
        self.voice_id = voice_id
        if not self.voice_id:
            raise ValueError("Voice ID is required for ElevenLabs TTS.")
        
        self.base_url = "https://api.elevenlabs.io/v1"
        self.headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.api_key
        }
    
    def synthesize_speech(self, text: str, output_path: str = None) -> str:
        """
        Synthesize text to speech using ElevenLabs
        
        Args:
            text: Text to synthesize
            output_path: Output file path (if None, creates temp file)
            
        Returns:
            Path to generated audio file
        """
        try:
            if output_path is None:
                output_path = tempfile.mktemp(suffix='.mp3')
            
            logger.info(f"Synthesizing speech: {text[:50]}...")
            
            # Prepare the request
            url = f"{self.base_url}/text-to-speech/{self.voice_id}"
            data = {
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
            }
            
            # Make the request
            response = requests.post(url, json=data, headers=self.headers)
            response.raise_for_status()
            
            # Save the audio
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"Speech synthesized: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error synthesizing speech: {str(e)}")
            raise
    
    def get_available_voices(self) -> List[dict]:
        """
        Get available voices from ElevenLabs
        
        Returns:
            List of available voices
        """
        try:
            url = f"{self.base_url}/voices"
            headers = {"xi-api-key": self.api_key}
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            voices = response.json()['voices']
            return voices
            
        except Exception as e:
            logger.error(f"Error fetching voices: {str(e)}")
            return []


class ConceptNarrator:
    """Main class that combines AI explanation generation and TTS"""
    
    def __init__(self, openai_api_key: str = None, elevenlabs_api_key: str = None, voice_id: str = None):
        """
        Initialize the concept narrator
        
        Args:
            openai_api_key: OpenAI API key
            elevenlabs_api_key: ElevenLabs API key
            voice_id: ElevenLabs voice ID
        """
        self.explainer = AIExplainer(openai_api_key)
        self.tts = ElevenLabsTTS(elevenlabs_api_key, voice_id)
        self.narrations = []
    
    def create_narrations(self, segments: List[Segment], video_title: str = "") -> List[NarrationSegment]:
        """
        Create narrations for all segments
        
        Args:
            segments: List of concept segments
            video_title: Video title for context
            
        Returns:
            List of narration segments with generated audio
        """
        narrations = []
        logger.info(f"Received {len(segments)} segments for narration generation.")
        print(f"[DEBUG] Received {len(segments)} segments for narration generation.")
        for i, segment in enumerate(segments):
            logger.info(f"Processing segment {i+1}/{len(segments)}: {getattr(segment, 'concept_title', 'N/A')} | Text: {getattr(segment, 'text', '')[:100]}")
            print(f"[DEBUG] Processing segment {i+1}/{len(segments)}: {getattr(segment, 'concept_title', 'N/A')} | Text: {getattr(segment, 'text', '')[:100]}")
            try:
                explanation = self.explainer.generate_explanation(segment.text, video_title)
                logger.info(f"Explanation for segment {i+1}: {explanation[:200]}")
                print(f"[DEBUG] Explanation for segment {i+1}: {explanation[:200]}")
                audio_path = tempfile.mktemp(suffix=f'_narration_{i+1}.mp3')
                self.tts.synthesize_speech(explanation, audio_path)
                narration = NarrationSegment(
                    original_segment=segment,
                    explanation_text=explanation,
                    audio_path=audio_path,
                    duration=self._estimate_audio_duration(explanation)
                )
                narrations.append(narration)
                logger.info(f"Created narration for: {getattr(segment, 'concept_title', 'N/A')}")
                print(f"[DEBUG] Created narration for: {getattr(segment, 'concept_title', 'N/A')}")
            except Exception as e:
                logger.error(f"Error creating narration for segment {i+1}: {str(e)}")
                print(f"[DEBUG] Error creating narration for segment {i+1}: {str(e)}")
                fallback_text = f"Let me briefly explain this concept: {getattr(segment, 'text', '')[:100]}..."
                try:
                    logger.info(f"Attempting fallback narration for segment {i+1}.")
                    print(f"[DEBUG] Attempting fallback narration for segment {i+1}.")
                    audio_path = tempfile.mktemp(suffix=f'_narration_fallback_{i+1}.mp3')
                    self.tts.synthesize_speech(fallback_text, audio_path)
                    narration = NarrationSegment(
                        original_segment=segment,
                        explanation_text=fallback_text,
                        audio_path=audio_path,
                        duration=self._estimate_audio_duration(fallback_text)
                    )
                    narrations.append(narration)
                except Exception as fallback_error:
                    logger.error(f"Fallback narration also failed for segment {i+1}: {str(fallback_error)}")
                    print(f"[DEBUG] Fallback narration also failed for segment {i+1}: {str(fallback_error)}")
                    continue
        self.narrations = narrations
        logger.info(f"Created {len(narrations)} narrations successfully. Segments received: {len(segments)}")
        print(f"[DEBUG] Created {len(narrations)} narrations successfully. Segments received: {len(segments)}")
        if len(narrations) == 0:
            logger.error("No narrations were generated. Check segment input, API keys, and service responses.")
            print("[DEBUG] No narrations were generated. Check segment input, API keys, and service responses.")
        return narrations
    
    def _estimate_audio_duration(self, text: str) -> float:
        """
        Estimate audio duration based on text length
        Assumes average speaking rate of ~150 words per minute
        
        Args:
            text: Text to estimate duration for
            
        Returns:
            Estimated duration in seconds
        """
        word_count = len(text.split())
        words_per_minute = 150
        duration_minutes = word_count / words_per_minute
        return duration_minutes * 60
    
    def cleanup_narrations(self):
        """Clean up generated narration audio files"""
        for narration in self.narrations:
            try:
                if os.path.exists(narration.audio_path):
                    os.remove(narration.audio_path)
                    logger.info(f"Cleaned up narration: {narration.audio_path}")
            except Exception as e:
                logger.warning(f"Error cleaning up {narration.audio_path}: {str(e)}")


def create_narrations_for_segments(segments: List[Segment], voice_id: str, video_title: str = "") -> List[NarrationSegment]:
    """
    Convenience function to create narrations for segments
    
    Args:
        segments: List of concept segments
        voice_id: ElevenLabs voice ID
        video_title: Video title for context
        
    Returns:
        List of narration segments
    """
    narrator = ConceptNarrator(voice_id=voice_id)
    return narrator.create_narrations(segments, video_title)


if __name__ == "__main__":
    # Test the narrator
    from transcription import Segment
    
    # Create a test segment
    test_segment = Segment(
        start_time=0.0,
        end_time=30.0,
        text="Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed.",
        concept_title="Machine Learning Introduction"
    )
    
    # Test voice ID (replace with actual voice ID)
    test_voice_id = "pNInz6obpgDQGcFmaJgB"  # Example voice ID
    
    try:
        narrator = ConceptNarrator(voice_id=test_voice_id)
        narrations = narrator.create_narrations([test_segment], "AI Tutorial Video")
        
        if narrations:
            print(f"Generated narration: {narrations[0].explanation_text}")
            print(f"Audio file: {narrations[0].audio_path}")
        
        # Cleanup
        narrator.cleanup_narrations()
        
    except Exception as e:
        print(f"Error testing narrator: {e}")
        print("Make sure you have set OPENAI_API_KEY and ELEVENLABS_API_KEY environment variables") 