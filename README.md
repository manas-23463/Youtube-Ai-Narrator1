# 🎬 YouTube AI Narrator

Transform educational YouTube videos by adding AI-powered explanations between concepts! This tool downloads YouTube videos, transcribes them, segments them into logical concepts, and inserts AI-generated narrations that explain each concept in simple terms.

## ✨ Features

- **📥 YouTube Download**: Downloads videos with high quality using `yt-dlp`
- **🎙️ Smart Transcription**: Uses OpenAI Whisper for accurate transcription
- **🧠 Intelligent Segmentation**: Automatically identifies logical concept boundaries
- **🤖 AI Explanations**: Generates clear, simple explanations using GPT-4
- **🗣️ Natural Speech**: Synthesizes explanations using ElevenLabs TTS
- **🎞️ Video Processing**: Seamlessly integrates narrations into the original video
- **☁️ Cloud API**: Deployed on Modal for easy access
- **🔧 Modular Design**: Clean, well-documented, and extensible codebase

## 🚀 Quick Start

### Option 1: Use the Cloud API (Recommended)

The YouTube AI Narrator is deployed on Modal and ready to use!

**API Base URL:** `https://manas--youtube-ai-narrator-api-process-video-api.modal.run`

#### Available Endpoints:

1. **Health Check**
   ```bash
   curl https://manas--youtube-ai-narrator-api-health-check.modal.run
   ```

2. **Get Available Voices**
   ```bash
   curl https://manas--youtube-ai-narrator-api-get-voices.modal.run
   ```

3. **Process Video**
   ```bash
   curl -X POST https://manas--youtube-ai-narrator-api-process-video-api.modal.run \
     -H "Content-Type: application/json" \
     -d '{
       "youtube_url": "https://www.youtube.com/watch?v=YOUR_VIDEO_ID",
       "voice_id": "21m00Tcm4TlvDq8ikWAM"
     }'
   ```

#### Download the Processed Video:
```bash
# Download and save the video locally
curl -X POST https://manas--youtube-ai-narrator-api-process-video-api.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "youtube_url": "https://www.youtube.com/watch?v=YOUR_VIDEO_ID",
    "voice_id": "21m00Tcm4TlvDq8ikWAM"
  }' | python3 -c "
import json, sys, base64
response = json.load(sys.stdin)
if response['success']:
    video_data = response['download_url'].split(',')[1]
    filename = response['output_filename']
    with open(filename, 'wb') as f:
        f.write(base64.b64decode(video_data))
    print(f'✅ Video saved to: {filename}')
else:
    print(f'❌ Error: {response[\"error\"]}')
"
```

### Option 2: Local Installation

## 🛠️ Prerequisites

### Required Software
1. **Python 3.8+**
2. **FFmpeg** - For video processing
   - **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg` (Ubuntu/Debian) or `sudo yum install ffmpeg` (CentOS/RHEL)

### API Keys
You'll need API keys for:
1. **OpenAI** (for GPT-4 and Whisper) - Get from [OpenAI Platform](https://platform.openai.com/)
2. **ElevenLabs** (for text-to-speech) - Get from [ElevenLabs](https://elevenlabs.io/)

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd youtube-ai-narrator
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
   ```

   Or set them in your shell:
   ```bash
   export OPENAI_API_KEY="your_openai_api_key_here"
   export ELEVENLABS_API_KEY="your_elevenlabs_api_key_here"
   ```

4. **Get your ElevenLabs voice ID**:
   You can find available voices by running:
   ```python
   from narrator import ElevenLabsTTS
   tts = ElevenLabsTTS()
   voices = tts.get_available_voices()
   for voice in voices[:5]:  # Show first 5 voices
       print(f"Name: {voice['name']}, ID: {voice['voice_id']}")
   ```

## 🚀 Usage

### Local Usage

```bash
python main.py "YOUTUBE_URL" "ELEVENLABS_VOICE_ID"
```

### Examples

```bash
# Basic usage with a YouTube video
python main.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" "pNInz6obpgDQGcFmaJgB"

# Use a larger Whisper model for better transcription
python main.py "https://youtu.be/example" "voice_id_here" --model large

# Specify output directory and filename
python main.py "https://www.youtube.com/watch?v=example" "voice_id_here" \
  --output-dir ./outputs \
  --output-filename "my_narrated_video.mp4"

# Enable verbose logging
python main.py "https://www.youtube.com/watch?v=example" "voice_id_here" --verbose
```

### Command Line Options

- `youtube_url` - YouTube video URL (required)
- `voice_id` - ElevenLabs voice ID (required)
- `--model` - Whisper model size: `tiny`, `base`, `small`, `medium`, `large` (default: `base`)
- `--output-dir` - Output directory (default: current directory)
- `--output-filename` - Custom output filename (optional)
- `--verbose` - Enable verbose logging

## 🌐 API Reference

### Base URL
```
https://manas--youtube-ai-narrator-api-process-video-api.modal.run
```

### Endpoints

#### 1. Health Check
- **URL:** `https://manas--youtube-ai-narrator-api-health-check.modal.run`
- **Method:** `GET`
- **Response:**
  ```json
  {
    "status": "ok"
  }
  ```

#### 2. Get Available Voices
- **URL:** `https://manas--youtube-ai-narrator-api-get-voices.modal.run`
- **Method:** `GET`
- **Response:**
  ```json
  {
    "success": true,
    "voices": [
      {
        "name": "Rachel",
        "voice_id": "21m00Tcm4TlvDq8ikWAM",
        "category": "premade",
        "description": "Professional female voice"
      }
    ],
    "count": 52
  }
  ```

#### 3. Process Video
- **URL:** `https://manas--youtube-ai-narrator-api-process-video-api.modal.run`
- **Method:** `POST`
- **Headers:** `Content-Type: application/json`
- **Request Body:**
  ```json
  {
    "youtube_url": "https://www.youtube.com/watch?v=YOUR_VIDEO_ID",
    "voice_id": "21m00Tcm4TlvDq8ikWAM",
    "whisper_model": "base",
    "output_filename": "my_video.mp4"
  }
  ```
- **Response:**
  ```json
  {
    "success": true,
    "video_title": "Original Video Title",
    "segments_count": 5,
    "narrations_count": 5,
    "output_filename": "my_video.mp4",
    "download_url": "data:video/mp4;base64,<base64_encoded_video_data>"
  }
  ```

### API Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `youtube_url` | string | ✅ | - | YouTube video URL |
| `voice_id` | string | ❌ | `21m00Tcm4TlvDq8ikWAM` | ElevenLabs voice ID |
| `whisper_model` | string | ❌ | `base` | Whisper model: `tiny`, `base`, `small`, `medium`, `large` |
| `output_filename` | string | ❌ | auto-generated | Custom output filename |

## 🎯 How It Works

1. **📥 Download**: Downloads the YouTube video and extracts audio
2. **📝 Transcribe**: Uses Whisper to transcribe the audio with timestamps
3. **✂️ Segment**: Intelligently segments transcription into logical concepts based on:
   - Natural pauses (>2 seconds)
   - Topic transition indicators
   - Sentence boundaries
   - Speaking pattern changes
4. **🤖 Explain**: Uses GPT-4 to generate simple explanations for each concept
5. **🗣️ Synthesize**: Converts explanations to speech using ElevenLabs TTS
6. **🎬 Combine**: Creates pause segments with narrations and combines with original video

## 📁 Project Structure

```
youtube-ai-narrator/
├── main.py                 # Main orchestration script
├── modal_api.py           # Modal cloud API deployment
├── deploy_modal.py        # Modal deployment script
├── test_api.py           # API testing client
├── youtube_downloader.py   # YouTube video downloading
├── transcription.py        # Whisper transcription & segmentation
├── narrator.py            # GPT-4 explanations & ElevenLabs TTS
├── video_processor.py     # FFmpeg video processing
├── get_voices.py          # Voice management utilities
├── requirements.txt       # Python dependencies
├── outputs/              # Generated videos (local)
├── MODAL_DEPLOYMENT.md   # Modal deployment guide
├── COOKIES_GUIDE.md      # YouTube cookies setup
└── README.md             # This file
```

## ☁️ Cloud Deployment

### Current Deployment Status
- **✅ Deployed**: Yes
- **Platform**: Modal
- **Status**: Active
- **URL**: `https://manas--youtube-ai-narrator-api-process-video-api.modal.run`

### Deploy Your Own Instance

1. **Install Modal CLI**:
   ```bash
   pip install modal
   ```

2. **Authenticate with Modal**:
   ```bash
   modal token new
   ```

3. **Deploy**:
   ```bash
   python deploy_modal.py
   ```

For detailed deployment instructions, see [MODAL_DEPLOYMENT.md](MODAL_DEPLOYMENT.md).

## ⚙️ Configuration

### Whisper Models
- `tiny` - Fastest, lowest quality (~39 MB)
- `base` - Good balance (~74 MB) - **Default**
- `small` - Better quality (~244 MB)
- `medium` - High quality (~769 MB)
- `large` - Highest quality (~1550 MB)

### Concept Segmentation Parameters
You can adjust segmentation sensitivity in `transcription.py`:
- Minimum concept duration (default: 15 seconds)
- Pause detection threshold (default: 2 seconds)
- Topic change indicators

### Narration Settings
Modify narration parameters in `narrator.py`:
- GPT-4 prompt templates
- TTS voice settings (stability, similarity_boost)
- Explanation length (default: 30-60 seconds when spoken)

## 🔍 Troubleshooting

### Common Issues

1. **FFmpeg not found**:
   ```
   Error: FFmpeg not found in PATH
   ```
   **Solution**: Install FFmpeg and ensure it's in your system PATH.

2. **API key errors**:
   ```
   Error: Missing required environment variables
   ```
   **Solution**: Set your `OPENAI_API_KEY` and `ELEVENLABS_API_KEY` environment variables.

3. **Memory issues with large models**:
   ```
   RuntimeError: CUDA out of memory
   ```
   **Solution**: Use a smaller Whisper model (`--model base` instead of `--model large`).

4. **Long processing times**:
   - Use smaller Whisper models (`tiny` or `base`)
   - Process shorter videos first
   - Consider using GPU acceleration if available

5. **API timeout errors**:
   ```
   Error: Request timeout
   ```
   **Solution**: The cloud API has a 1-hour timeout. For very long videos, consider processing locally.

### Debug Mode

Enable verbose logging to see detailed processing information:
```bash
python main.py "URL" "VOICE_ID" --verbose
```

## 🎨 Customization

### Adding Custom Voices
1. Visit [ElevenLabs Voice Lab](https://elevenlabs.io/voice-lab)
2. Create or clone a voice
3. Copy the voice ID and use it in the script

### Modifying AI Prompts
Edit the prompt template in `narrator.py` → `AIExplainer._create_explanation_prompt()` to:
- Change explanation style
- Add specific instructions
- Target different audiences

### Adjusting Video Processing
Modify `video_processor.py` to:
- Change pause screen appearance
- Add custom graphics during narrations
- Adjust audio mixing levels

## 📊 Performance Tips

- **Use GPU acceleration**: Install CUDA-compatible PyTorch for faster Whisper processing
- **Batch processing**: Process multiple videos by modifying the main script
- **Optimize for speed**: Use `--model tiny` for quick tests
- **Optimize for quality**: Use `--model large` for final outputs
- **Cloud processing**: Use the Modal API for heavy processing without local resources

## 🗂️ Output Storage

### Local Processing
- **Location**: `./outputs/` directory (default)
- **Format**: MP4 files with AI narrations
- **Naming**: `{Video_Title}_AI_Narrated_{Timestamp}.mp4`

### Cloud API
- **Storage**: No permanent storage (on-demand processing)
- **Delivery**: Base64-encoded data URL in API response
- **Download**: Must be saved locally by the client

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenAI** for Whisper and GPT-4
- **ElevenLabs** for high-quality text-to-speech
- **yt-dlp** for reliable YouTube downloading
- **FFmpeg** for video processing capabilities
- **Modal** for cloud infrastructure

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the logs in `youtube_ai_narrator.log`
3. Test the cloud API endpoints
4. Open an issue with detailed error information

## 📈 Recent Updates

- ✅ **Cloud API**: Deployed on Modal for easy access
- ✅ **Voice Management**: Added voice listing and selection
- ✅ **Error Handling**: Improved error messages and debugging
- ✅ **Video Processing**: Optimized FFmpeg-based processing
- ✅ **Documentation**: Comprehensive API documentation

---

**Happy narrating! 🎉** 