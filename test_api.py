import sys
import os
import requests

# Usage:
#   python test_api.py http://localhost:8000 test_image.jpg

api_url   = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
img_path  = sys.argv[2] if len(sys.argv) > 2 else "test_image.jpg"

if not os.path.exists(img_path):
    print(f"Error: image not found at '{img_path}'")
    sys.exit(1)

endpoint = f"{api_url.rstrip('/')}/detect"
print(f"Sending request to: {endpoint}")

with open(img_path, "rb") as f:
    response = requests.post(endpoint, files={"file": f}, timeout=30)

print(f"Status: {response.status_code}")

if response.status_code == 200:
    data = response.json()
    detections = data.get("detections", [])
    print(f"\nDetections found: {len(detections)}")
    for i, d in enumerate(detections, 1):
        print(f"  {i}. {d['class']}  confidence: {d['confidence']}  box: {d['box']}")
else:
    print(response.text)