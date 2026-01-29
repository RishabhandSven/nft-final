#!/usr/bin/env python3
"""
Simple startup script for the wash trading API
Run without auto-reload to avoid shutdown issues
"""
import sys
import uvicorn

if __name__ == "__main__":
    print("[*] Starting Wash Trading API server...")
    try:
        uvicorn.run(
            "ai_engine.api:app",
            host="127.0.0.1",
            port=8001,
            log_level="info",
            reload=False  # Disable reload to prevent crashes
        )
    except Exception as e:
        print(f"[!] Error: {e}")
        sys.exit(1)
