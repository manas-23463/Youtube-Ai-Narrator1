#!/usr/bin/env python3
"""
Utility script to list available ElevenLabs voices
Run this to find voice IDs for use with the YouTube AI Narrator
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """List available ElevenLabs voices"""
    print("🎙️ ElevenLabs Voice Finder")
    print("=" * 50)
    
    # Check if API key is available
    api_key = os.getenv('ELEVENLABS_API_KEY')
    if not api_key:
        print("❌ Error: ELEVENLABS_API_KEY not found in environment variables")
        print("\nPlease set your ElevenLabs API key:")
        print("1. Create a .env file with: ELEVENLABS_API_KEY=your_api_key_here")
        print("2. Or export it: export ELEVENLABS_API_KEY='your_api_key_here'")
        print("\nGet your API key from: https://elevenlabs.io/")
        sys.exit(1)
    
    try:
        # Import and use the ElevenLabs TTS class
        from narrator import ElevenLabsTTS
        
        # Initialize with a dummy voice ID just to access the API
        tts = ElevenLabsTTS(api_key=api_key, voice_id="dummy")
        
        # Get available voices
        print("Fetching available voices...")
        voices = tts.get_available_voices()
        
        if not voices:
            print("❌ No voices found. Please check your API key.")
            return
        
        print(f"\n✅ Found {len(voices)} available voices:\n")
        
        # Display voices in a nice format
        for i, voice in enumerate(voices, 1):
            name = voice.get('name', 'Unknown')
            voice_id = voice.get('voice_id', 'Unknown')
            category = voice.get('category', 'Unknown')
            
            print(f"{i:2d}. {name}")
            print(f"    ID: {voice_id}")
            print(f"    Category: {category}")
            
            # Show additional info if available
            if 'labels' in voice:
                labels = voice['labels']
                if labels:
                    accent = labels.get('accent', '')
                    description = labels.get('description', '')
                    age = labels.get('age', '')
                    gender = labels.get('gender', '')
                    
                    details = []
                    if gender: details.append(f"Gender: {gender}")
                    if age: details.append(f"Age: {age}")
                    if accent: details.append(f"Accent: {accent}")
                    if description: details.append(f"Description: {description}")
                    
                    if details:
                        print(f"    {' | '.join(details)}")
            
            print()
        
        print("=" * 50)
        print("💡 To use a voice, copy its ID and use it with the main script:")
        print(f'   python main.py "YOUTUBE_URL" "VOICE_ID"')
        print("\n🎯 Popular voices for narration:")
        
        # Suggest some good voices for narration
        suggested_categories = ['narration', 'professional', 'conversational']
        suggestions = []
        
        for voice in voices:
            category = voice.get('category', '').lower()
            if any(cat in category for cat in suggested_categories):
                suggestions.append(voice)
        
        if suggestions:
            for voice in suggestions[:5]:  # Show top 5 suggestions
                name = voice.get('name', 'Unknown')
                voice_id = voice.get('voice_id', 'Unknown')
                print(f"   • {name}: {voice_id}")
        else:
            # If no specific categories found, show first few voices
            for voice in voices[:3]:
                name = voice.get('name', 'Unknown')
                voice_id = voice.get('voice_id', 'Unknown')
                print(f"   • {name}: {voice_id}")
        
    except ImportError:
        print("❌ Error: Required modules not installed")
        print("Please run: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error fetching voices: {str(e)}")
        print("Please check your API key and internet connection")


if __name__ == "__main__":
    main() 