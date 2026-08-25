import os
import sys
import webbrowser
import http.server
import socketserver
from pathlib import Path

PORT = int(os.getenv("PORT", 8080))
UI_DIR = Path(__file__).parent / "ui"

class VisionBoardHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(UI_DIR), **kwargs)

    def log_message(self, format, *args):
        # Clean logging
        sys.stdout.write(f"[VisionBoard UI] {self.address_string()} - {format % args}\n")

def run_server(port=PORT, auto_open=True):
    os.chdir(str(UI_DIR))
    url = f"http://localhost:{port}"
    print(f"\n{'='*55}")
    print(f"       VisionBoard AI — Signboard Detection Studio")
    print(f"{'='*55}")
    print(f"  [+] Serving UI at: {url}")
    print(f"  [+] Press Ctrl+C to stop the server")
    print(f"{'='*55}\n")

    if auto_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    with socketserver.TCPServer(("", port), VisionBoardHTTPRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down VisionBoard UI server...")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch VisionBoard AI Web Studio")
    parser.add_argument("--port", type=int, default=PORT, help="Port to serve on (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open browser")
    args = parser.parse_args()

    run_server(port=args.port, auto_open=not args.no_browser)
