#!/usr/bin/env python
"""
Start the NFT Wash Trading Detection API Server.
Uses subprocess for stable Windows compatibility.
Run with: python run_api_server.py
"""

import subprocess
import time
import sys
import os
import requests

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def start_server():
    """Start uvicorn in a subprocess and keep it running."""
    print("[*] Starting NFT Wash Trading Detection API Server...")
    print("[*] Server will be available at http://localhost:8000")
    print("[*] Press Ctrl+C to stop\n")
    
    try:
        # Start uvicorn server in subprocess
        cmd = [sys.executable, "-m", "uvicorn", "ai_engine.api:app", "--host", "127.0.0.1", "--port", "8000"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Give server time to start
        time.sleep(3)
        
        # Verify it's running
        try:
            resp = requests.get('http://127.0.0.1:8000/health', timeout=2)
            print(f"[OK] Server is running and responding!")
            print(f"[OK] Health check: {resp.json()}\n")
        except:
            print("[!] Server started but health check failed")
        
        # Keep process alive
        while True:
            time.sleep(1)
            if process.poll() is not None:
                print("[!] Server process exited unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n[*] Stopping server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("[*] Server stopped")
        sys.exit(0)
    except Exception as e:
        print(f"[!] Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    start_server()

