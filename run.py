"""
AdaptiveEdge Voice AI — Application Runner.
Launches the FastAPI server and serves the on-device dashboard.
"""

import sys
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run AdaptiveEdge Voice AI On-Device Stack")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print("=" * 60)
    print("  AdaptiveEdge Voice AI — On-Device Conversational Stack")
    print("  CPU-Only • Cloud: OFF • Tiered Local Routing")
    print(f"  Dashboard available at: http://{args.host}:{args.port}")
    print("=" * 60)

    uvicorn.run("backend.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
