#!/usr/bin/env python3
"""
Persistent NFT Wash Trading Detection Server Launcher
Keeps the uvicorn server running in background for frontend integration
"""
import subprocess
import time
import sys
import os
import signal

def main():
    port = sys.argv[1] if len(sys.argv) > 1 else '8000'
    
    print(f"[*] Starting NFT Wash Trading Detection Server on port {port}...")
    print("[*] Server will run indefinitely. Press Ctrl+C to stop.")
    print()
    
    # Start uvicorn server in background
    proc = subprocess.Popen(
        [sys.executable, '-m', 'uvicorn', 'ai_engine.api:app', '--port', port, '--host', '0.0.0.0'],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    try:
        # Stream output from the server
        for line in proc.stdout:
            print(line.rstrip())
            
    except KeyboardInterrupt:
        print("\n[*] Shutting down server...")
        proc.terminate()
        proc.wait(timeout=5)
        print("[OK] Server stopped")
    except Exception as e:
        print(f"[!] Error: {e}")
        proc.terminate()

if __name__ == '__main__':
    main()
