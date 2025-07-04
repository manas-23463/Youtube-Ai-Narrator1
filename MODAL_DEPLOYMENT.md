# YouTube AI Narrator - Modal Deployment Guide

This guide will help you deploy the YouTube AI Narrator as a cloud API using Modal.

## Prerequisites

1. **Modal Account**: Sign up at [modal.com](https://modal.com)
2. **API Keys**: You need:
   - OpenAI API key
   - ElevenLabs API key
3. **Python Environment**: Python 3.11+ with Modal CLI installed

## Installation

1. **Install Modal CLI**:
   ```bash
   pip install modal
   ```

2. **Authenticate with Modal**:
   ```bash
   modal token new
   ```

3. **Set up environment variables**:
   Create a `.env` file in your project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
   ```

## Deployment

### Option 1: Quick Deploy (Recommended)

Run the deployment script:
```bash
python deploy_modal.py
```

This will:
- Set up Modal secrets with your API keys
- Deploy the API to Modal
- Provide you with the public URL

### Option 2: Manual Deploy

1. **Set up secrets manually**:
   ```bash
   modal secret create youtube-ai-narrator-secrets \
     OPENAI_API_KEY=your_openai_api_key_here \
     ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
   ```

2. **Deploy the API**:
   ```bash
   modal deploy modal_api.py
   ```

## API Endpoints

Once deployed, you'll get a public URL like: `https://your-app.modal.run`

### Available Endpoints:

1. **Health Check** - `GET /health_check`
   ```bash
   curl https://your-app.modal.run/health_check
   ```

2. **Get Voices** - `GET /get_voices`
   ```bash
   curl https://your-app.modal.run/get_voices
   ```

3. **Process Video** - `POST /process_video_api`
   ```bash
   curl -X POST https://your-app.modal.run/process_video_api \
     -H "Content-Type: application/json" \
     -d '{
       "youtube_url": "https://www.youtube.com/watch?v=your_video_id",
       "voice_id": "21m00Tcm4TlvDq8ikWAM",
       "whisper_model": "base",
       "output_filename": "my_video.mp4"
     }'
   ```

## API Request Format

### Process Video Request:
```json
{
  "youtube_url": "https://www.youtube.com/watch?v=your_video_id",
  "voice_id": "21m00Tcm4TlvDq8ikWAM",  // Optional, uses default if not provided
  "whisper_model": "base",             // "tiny", "base", "small", "medium", "large"
  "output_filename": "my_video.mp4"    // Optional
}
```

### Process Video Response:
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

## Using the Python Client

Use the provided test client to interact with your API:

```python
from test_api import YouTubeAINarratorClient

# Initialize client
client = YouTubeAINarratorClient("https://your-app.modal.run")

# Process a video
result = client.process_video(
    youtube_url="https://www.youtube.com/watch?v=your_video_id",
    whisper_model="base"
)

# Download the result
if result["success"]:
    client.download_video(result, "output_video.mp4")
```

## Configuration Options

### Whisper Models:
- `tiny`: Fastest, least accurate
- `base`: Good balance (default)
- `small`: Better accuracy
- `medium`: High accuracy
- `large`: Best accuracy, slowest

### Voice IDs:
Get available voices using the `/get_voices` endpoint or check your ElevenLabs dashboard.

## Resource Allocation

The Modal deployment uses:
- **GPU**: T4 (for Whisper transcription)
- **Memory**: 8GB RAM
- **CPU**: 4 cores
- **Timeout**: 1 hour per request

## Cost Estimation

Modal pricing (as of 2024):
- **GPU**: ~$0.60/hour for T4
- **CPU/Memory**: ~$0.10/hour
- **Typical processing time**: 5-15 minutes per video

## Troubleshooting

### Common Issues:

1. **Missing API Keys**:
   - Ensure your `.env` file has the correct API keys
   - Check that Modal secrets are properly set

2. **Deployment Fails**:
   - Check Modal logs: `modal logs youtube-ai-narrator-api`
   - Ensure all dependencies are in the requirements

3. **Processing Timeout**:
   - Use smaller Whisper models for faster processing
   - Check video length (longer videos take more time)

4. **Memory Issues**:
   - The deployment uses 8GB RAM, which should be sufficient for most videos
   - For very long videos, consider processing in segments

### Getting Help:

1. **Check Modal logs**:
   ```bash
   modal logs youtube-ai-narrator-api
   ```

2. **View deployment status**:
   ```bash
   modal app list
   ```

3. **Redeploy if needed**:
   ```bash
   modal deploy modal_api.py
   ```

## Security Notes

- API keys are stored securely in Modal secrets
- The API is public - consider adding authentication if needed
- Video data is returned as base64 in the response
- Temporary files are cleaned up automatically

## Scaling

The Modal deployment automatically scales:
- Each request runs in its own container
- Multiple videos can be processed simultaneously
- No need to manage infrastructure

## Example Usage

Here's a complete example of processing a video:

```python
import requests
import base64

# API endpoint
url = "https://your-app.modal.run/process_video_api"

# Request payload
payload = {
    "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "whisper_model": "base"
}

# Send request
response = requests.post(url, json=payload)
result = response.json()

if result["success"]:
    # Download the video
    data_url = result["download_url"]
    base64_data = data_url.split(",", 1)[1]
    video_data = base64.b64decode(base64_data)
    
    with open("ai_narrated_video.mp4", "wb") as f:
        f.write(video_data)
    
    print(f"✅ Video saved as: ai_narrated_video.mp4")
else:
    print(f"❌ Error: {result['error']}")
```

## Support

For issues with:
- **Modal deployment**: Check Modal documentation and logs
- **API functionality**: Review the local implementation
- **API keys**: Contact OpenAI/ElevenLabs support 