# YouTube Cookies Export Guide

If you're experiencing "Sign in to confirm you're not a bot" errors, you can export cookies from your browser to authenticate with YouTube. This is much safer than sharing login credentials.

## How to Export Cookies

### Chrome/Edge:
1. Install the "Get cookies.txt" extension from Chrome Web Store
2. Go to YouTube and make sure you're logged in
3. Click the extension icon and export cookies
4. Save the file as `youtube_cookies.txt`

### Firefox:
1. Install the "cookies.txt" extension
2. Go to YouTube and make sure you're logged in
3. Use the extension to export cookies
4. Save the file as `youtube_cookies.txt`

### Manual Method (Chrome):
1. Go to YouTube and log in
2. Open Developer Tools (F12)
3. Go to Application/Storage tab
4. Find Cookies under Storage
5. Right-click and export as text file

## Using Cookies with the API

Once you have the cookies file, you can:

1. **Upload it to Modal** (if using the cloud API)
2. **Place it in your project directory** (if running locally)
3. **Reference it in your API call**

## Security Notes:
- Never share your cookies file publicly
- Cookies expire periodically, so you may need to re-export
- This method is safer than sharing passwords
- Only use cookies from your own browser

## Example Usage:
```bash
# Local usage
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --cookies-file youtube_cookies.txt

# API usage (you'd need to modify the API to accept cookies file)
```

## Alternative Solutions:
1. Try different videos (some are more restricted)
2. Use the improved downloader with multiple fallback methods
3. Consider using videos from other platforms 