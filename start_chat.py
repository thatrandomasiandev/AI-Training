#!/usr/bin/env python3
"""
Quick launcher for the AI Engineering System Chat Interface
"""

import subprocess
import sys
import time
import webbrowser
import os


def main():
    """Launch the AI chat interface."""
    print("🚀 AI Engineering System - Chat Interface Launcher")
    print("="*60)
    
    # Check if the chat interface file exists
    chat_file = "ai_chat_interface.py"
    if not os.path.exists(chat_file):
        print(f"❌ Error: {chat_file} not found!")
        print("Make sure you're in the correct directory.")
        return
    
    print("🌐 Starting chat interface...")
    print("📱 The interface will open in your web browser")
    print("🔄 To stop the server, press Ctrl+C in the terminal")
    print("="*60)
    
    try:
        # Start the chat interface
        subprocess.run([sys.executable, chat_file])
    except KeyboardInterrupt:
        print("\n👋 Chat interface stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting chat interface: {e}")


if __name__ == "__main__":
    main()
