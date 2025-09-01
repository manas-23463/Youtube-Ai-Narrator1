#!/usr/bin/env python3
"""
Simple S3 Test Script
Quick test to verify S3 URL accessibility and basic functionality
"""

import requests
import os
import sys
from urllib.parse import urlparse

def test_s3_url_accessibility(s3_url: str) -> bool:
    """Test if S3 URL is accessible and returns video content"""
    try:
        print(f"🔍 Testing S3 URL accessibility: {s3_url}")
        
        # Test HEAD request first (faster)
        print("   📡 Testing HEAD request...")
        head_response = requests.head(s3_url, timeout=10)
        
        if head_response.status_code == 200:
            content_type = head_response.headers.get('content-type', 'unknown')
            content_length = head_response.headers.get('content-length', 'unknown')
            
            print(f"   ✅ HEAD request successful!")
            print(f"      Content-Type: {content_type}")
            print(f"      Content-Length: {content_length} bytes")
            
            # Check if it's a video file
            if 'video' in content_type.lower() or any(ext in s3_url.lower() for ext in ['.mp4', '.webm', '.mov', '.avi', '.mkv']):
                print(f"   ✅ Content appears to be a video file")
                return True
            else:
                print(f"   ⚠️  Content-Type doesn't indicate video: {content_type}")
                return False
        else:
            print(f"   ❌ HEAD request failed - Status: {head_response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ HEAD request failed - Error: {e}")
        return False

def test_s3_video_download(s3_url: str, max_size_mb: int = 10) -> bool:
    """Test downloading a small portion of the S3 video"""
    try:
        print(f"   📥 Testing video download (max {max_size_mb}MB)...")
        
        # Download first few MB to test
        max_bytes = max_size_mb * 1024 * 1024
        
        response = requests.get(s3_url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Read first few MB
        downloaded_bytes = 0
        chunk_size = 8192
        
        for chunk in response.iter_content(chunk_size=chunk_size):
            downloaded_bytes += len(chunk)
            if downloaded_bytes >= max_bytes:
                break
        
        print(f"   ✅ Successfully downloaded {downloaded_bytes / (1024*1024):.1f}MB")
        return True
        
    except Exception as e:
        print(f"   ❌ Download test failed - Error: {e}")
        return False

def test_s3_url_format(s3_url: str) -> bool:
    """Test if S3 URL format is valid"""
    try:
        print(f"🔍 Testing S3 URL format: {s3_url}")
        
        # Parse URL
        parsed = urlparse(s3_url)
        
        # Check basic structure
        if parsed.scheme != 'https':
            print(f"   ❌ Invalid scheme: {parsed.scheme} (expected https)")
            return False
        
        if not parsed.netloc:
            print(f"   ❌ No hostname found")
            return False
        
        # Check if it looks like S3
        if 's3.amazonaws.com' in parsed.netloc or '.s3.' in parsed.netloc:
            print(f"   ✅ URL appears to be S3 format")
        else:
            print(f"   ⚠️  URL doesn't match typical S3 pattern")
        
        # Check file extension
        path = parsed.path.lower()
        video_extensions = ['.mp4', '.webm', '.mov', '.avi', '.mkv']
        
        has_video_extension = any(path.endswith(ext) for ext in video_extensions)
        if has_video_extension:
            print(f"   ✅ URL has video file extension")
        else:
            print(f"   ⚠️  URL doesn't have common video file extension")
        
        return True
        
    except Exception as e:
        print(f"   ❌ URL format test failed - Error: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Simple S3 URL Test")
    print("=" * 50)
    
    # Get S3 URL from user
    s3_url = input("🔗 Enter your S3 video URL: ").strip()
    
    if not s3_url:
        print("❌ No URL provided. Exiting.")
        return
    
    if not s3_url.startswith("https://"):
        print("❌ Invalid URL. Must start with https://")
        return
    
    print(f"\n🚀 Starting S3 URL tests...")
    print("=" * 50)
    
    # Test 1: URL Format
    format_ok = test_s3_url_format(s3_url)
    
    # Test 2: URL Accessibility
    accessibility_ok = test_s3_url_accessibility(s3_url)
    
    # Test 3: Video Download (small test)
    download_ok = False
    if accessibility_ok:
        download_ok = test_s3_video_download(s3_url, max_size_mb=5)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    
    tests = [
        ("URL Format", format_ok),
        ("URL Accessibility", accessibility_ok),
        ("Video Download", download_ok)
    ]
    
    passed = 0
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Your S3 URL is ready for use.")
        print("\n💡 Next steps:")
        print("   1. Start your API server: python interactive_api.py")
        print("   2. Run the full integration test: python test_s3_integration.py")
        print("   3. Use the S3 URL in your interactive player")
    elif passed >= 2:
        print("⚠️  Most tests passed. Your S3 URL should work with some limitations.")
    else:
        print("❌ Multiple tests failed. Please check your S3 URL and permissions.")

if __name__ == "__main__":
    main()
