#!/usr/bin/env python3
"""
Simple mock backend server for Lunar GeoDetect
This is a lightweight alternative that doesn't require heavy ML dependencies
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import random
from datetime import datetime

class CORSRequestHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "message": "Lunar GeoDetect API is running (Mock Mode)",
                "version": "1.0.0",
                "status": "ready"
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path.startswith('/detect/'):
            # Generate mock detection results
            detection_type = self.path.split('/')[-1]
            
            # Read the request body (we won't process it, just acknowledge it)
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                self.rfile.read(content_length)
            
            # Generate mock results
            results = self.generate_mock_results(detection_type)
            
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(results).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def generate_mock_results(self, detection_type):
        """Generate mock detection results"""
        results = {
            "boulders": [],
            "landslides": [],
            "processing_time": round(random.uniform(2.0, 4.0), 2),
            "metadata": {
                "image_size": [800, 600],
                "detection_mode": detection_type,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        if detection_type in ['boulders', 'all']:
            # Generate random boulders
            num_boulders = random.randint(5, 12)
            for i in range(num_boulders):
                boulder = {
                    "id": f"B-{i+1:03d}",
                    "diameter": round(random.uniform(5.0, 20.0), 1),
                    "lat": round(23.45 + random.uniform(-0.01, 0.01), 6),
                    "lon": round(-45.67 + random.uniform(-0.01, 0.01), 6),
                    "confidence": random.randint(75, 98),
                    "bbox": None
                }
                results["boulders"].append(boulder)
        
        if detection_type in ['landslides', 'all']:
            # Generate random landslides
            num_landslides = random.randint(1, 3)
            for i in range(num_landslides):
                landslide = {
                    "id": f"L-{i+1:03d}",
                    "area_km2": round(random.uniform(0.5, 2.5), 2),
                    "center": [
                        round(23.45 + random.uniform(-0.01, 0.01), 6),
                        round(-45.67 + random.uniform(-0.01, 0.01), 6)
                    ],
                    "confidence": random.randint(80, 95),
                    "polygon": None
                }
                results["landslides"].append(landslide)
        
        return results

    def log_message(self, format, *args):
        # Override to reduce console spam
        if self.path != '/favicon.ico':
            super().log_message(format, *args)

def run_server(port=8000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, CORSRequestHandler)
    print(f"🚀 Mock Lunar GeoDetect Backend running on http://localhost:{port}")
    print("This is a lightweight mock server for testing purposes")
    print("Press Ctrl+C to stop")
    httpd.serve_forever()

if __name__ == '__main__':
    run_server() 