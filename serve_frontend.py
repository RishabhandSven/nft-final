#!/usr/bin/env python3
"""
Simple HTTP server for NFT Wash Trading Detection Frontend
Serves HTML/CSS/JS on port 3000
"""
import http.server
import socketserver
import os
from pathlib import Path

PORT = 3000
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow communication with backend
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.send_header('Cache-Control', 'no-store, must-revalidate')
        super().end_headers()

    def do_GET(self):
        # Serve index.html for root path
        if self.path == '/':
            self.path = '/index.html'
        return super().do_GET()

os.chdir(SCRIPT_DIR)

print(f"[*] Starting frontend server on http://localhost:{PORT}")
print(f"[*] Serving files from: {SCRIPT_DIR}")

with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
    print(f"[OK] Frontend server running. Press Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down...")
