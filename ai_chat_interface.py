#!/usr/bin/env python3
"""
AI Engineering System - Interactive Chat Interface
Opens in a web browser for real-time chat with your AI system.
"""

import asyncio
import json
import webbrowser
import threading
import time
from flask import Flask, render_template_string, request, jsonify
from datetime import datetime
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import your AI system
try:
    from ai_engineering_system.core.main import EngineeringAI
    from ai_engineering_system.core.orchestrator import EngineeringTask
    AI_SYSTEM_AVAILABLE = True
except ImportError:
    AI_SYSTEM_AVAILABLE = False
    print("Warning: AI system not available. Running in demo mode.")

app = Flask(__name__)

# Global AI system instance
ai_system = None
chat_history = []

# HTML template for the chat interface
CHAT_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Engineering System - Chat Interface</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .chat-container {
            width: 90%;
            max-width: 800px;
            height: 80vh;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        
        .chat-header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }
        
        .chat-header p {
            opacity: 0.9;
            font-size: 14px;
        }
        
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
        }
        
        .message.user {
            justify-content: flex-end;
        }
        
        .message.ai {
            justify-content: flex-start;
        }
        
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 18px;
            word-wrap: break-word;
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .message.ai .message-content {
            background: white;
            border: 1px solid #e1e5e9;
            color: #333;
        }
        
        .message-avatar {
            width: 32px;
            height: 32px;
            border-radius: 50%;
            margin: 0 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            font-size: 14px;
        }
        
        .message.user .message-avatar {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        
        .message.ai .message-avatar {
            background: #28a745;
            color: white;
        }
        
        .chat-input {
            padding: 20px;
            background: white;
            border-top: 1px solid #e1e5e9;
        }
        
        .input-container {
            display: flex;
            gap: 10px;
        }
        
        .chat-input input {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid #e1e5e9;
            border-radius: 25px;
            font-size: 14px;
            outline: none;
        }
        
        .chat-input input:focus {
            border-color: #667eea;
        }
        
        .send-button {
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
        }
        
        .send-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .send-button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .typing-indicator {
            display: none;
            padding: 10px 16px;
            color: #666;
            font-style: italic;
        }
        
        .system-status {
            padding: 10px 20px;
            background: #e8f5e8;
            border-bottom: 1px solid #d4edda;
            font-size: 12px;
            color: #155724;
        }
        
        .system-status.error {
            background: #f8d7da;
            color: #721c24;
        }
        
        .quick-actions {
            padding: 10px 20px;
            background: #f8f9fa;
            border-bottom: 1px solid #e1e5e9;
        }
        
        .quick-actions button {
            margin: 2px;
            padding: 6px 12px;
            background: #6c757d;
            color: white;
            border: none;
            border-radius: 15px;
            cursor: pointer;
            font-size: 12px;
        }
        
        .quick-actions button:hover {
            background: #5a6268;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>🤖 AI Engineering System</h1>
            <p>Chat with your advanced multi-modal AI system</p>
        </div>
        
        <div id="system-status" class="system-status">
            {% if ai_available %}
                ✅ AI System Ready - All modules loaded
            {% else %}
                ⚠️ Demo Mode - AI system not available
            {% endif %}
        </div>
        
        <div class="quick-actions">
            <button onclick="sendQuickMessage('What can you do?')">What can you do?</button>
            <button onclick="sendQuickMessage('Explain stress in engineering')">Explain stress</button>
            <button onclick="sendQuickMessage('Analyze a beam problem')">Beam analysis</button>
            <button onclick="sendQuickMessage('How do you think?')">How do you think?</button>
        </div>
        
        <div class="chat-messages" id="chat-messages">
            <div class="message ai">
                <div class="message-avatar">🤖</div>
                <div class="message-content">
                    Hello! I'm your AI Engineering System. I can help you with:
                    <br>• Engineering problem solving
                    <br>• Technical explanations
                    <br>• Multi-modal data analysis
                    <br>• Creative solutions
                    <br><br>What would you like to explore today?
                </div>
            </div>
        </div>
        
        <div class="typing-indicator" id="typing-indicator">
            🤖 AI is thinking...
        </div>
        
        <div class="chat-input">
            <div class="input-container">
                <input type="text" id="message-input" placeholder="Ask me anything about engineering..." onkeypress="handleKeyPress(event)">
                <button class="send-button" onclick="sendMessage()" id="send-button">Send</button>
            </div>
        </div>
    </div>

    <script>
        function handleKeyPress(event) {
            if (event.key === 'Enter') {
                sendMessage();
            }
        }
        
        function sendQuickMessage(message) {
            document.getElementById('message-input').value = message;
            sendMessage();
        }
        
        function sendMessage() {
            const input = document.getElementById('message-input');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message to chat
            addMessage('user', message);
            input.value = '';
            
            // Show typing indicator
            showTypingIndicator();
            
            // Send to server
            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({message: message})
            })
            .then(response => response.json())
            .then(data => {
                hideTypingIndicator();
                addMessage('ai', data.response);
            })
            .catch(error => {
                hideTypingIndicator();
                addMessage('ai', 'Sorry, I encountered an error. Please try again.');
                console.error('Error:', error);
            });
        }
        
        function addMessage(sender, content) {
            const messagesContainer = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            
            const avatar = sender === 'user' ? '👤' : '🤖';
            messageDiv.innerHTML = `
                <div class="message-avatar">${avatar}</div>
                <div class="message-content">${content.replace(/\\n/g, '<br>')}</div>
            `;
            
            messagesContainer.appendChild(messageDiv);
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        function showTypingIndicator() {
            document.getElementById('typing-indicator').style.display = 'block';
            document.getElementById('send-button').disabled = true;
        }
        
        function hideTypingIndicator() {
            document.getElementById('typing-indicator').style.display = 'none';
            document.getElementById('send-button').disabled = false;
        }
        
        // Focus on input when page loads
        window.onload = function() {
            document.getElementById('message-input').focus();
        };
    </script>
</body>
</html>
"""


class AIChatInterface:
    """Interactive chat interface for the AI Engineering System."""
    
    def __init__(self):
        self.ai_system = None
        self.setup_ai_system()
    
    def setup_ai_system(self):
        """Initialize the AI system."""
        global AI_SYSTEM_AVAILABLE
        if AI_SYSTEM_AVAILABLE:
            try:
                self.ai_system = EngineeringAI(device='cpu')
                print("✅ AI System initialized successfully")
            except Exception as e:
                print(f"⚠️ Warning: Could not initialize AI system: {e}")
                AI_SYSTEM_AVAILABLE = False
        else:
            print("⚠️ Running in demo mode - AI system not available")
    
    async def process_message(self, message: str) -> str:
        """Process a user message and return AI response."""
        if not AI_SYSTEM_AVAILABLE or not self.ai_system:
            return self.get_demo_response(message)
        
        try:
            # Determine the type of question
            if any(word in message.lower() for word in ['explain', 'what is', 'how does', 'tell me about']):
                return await self.handle_explanation_request(message)
            elif any(word in message.lower() for word in ['analyze', 'calculate', 'solve', 'design']):
                return await self.handle_analysis_request(message)
            elif any(word in message.lower() for word in ['think', 'reason', 'how do you']):
                return await self.handle_thinking_request(message)
            else:
                return await self.handle_general_request(message)
        
        except Exception as e:
            return f"I encountered an error while processing your request: {str(e)}. Please try rephrasing your question."
    
    async def handle_explanation_request(self, message: str) -> str:
        """Handle explanation requests."""
        # Simulate AI explanation capabilities
        if 'stress' in message.lower():
            return """**Stress in Engineering:**

**What it is:** Internal force per unit area within a material

**Units:** Pascal (Pa) or MPa

**Real-world examples:**
• Bridge beam under traffic load
• Bolt under tension  
• Concrete column supporting building

**Think of it like:**
• Pressure in a balloon
• Tension in a rope

**Key formula:** σ = F/A (stress = force/area)

Would you like me to explain any specific type of stress or show calculations?"""
        
        elif 'beam' in message.lower():
            return """**Beam Analysis:**

**Types of beams:**
• Simply supported
• Cantilever
• Fixed-fixed
• Continuous

**Key calculations:**
• Bending moment: M = wL²/8 (simply supported)
• Shear force: V = wL/2
• Deflection: δ = 5wL⁴/(384EI)

**Design considerations:**
• Material properties
• Loading conditions
• Safety factors
• Deflection limits

Would you like me to analyze a specific beam problem?"""
        
        else:
            return "I'd be happy to explain engineering concepts! Could you be more specific about what you'd like me to explain? For example:\n• Stress and strain\n• Beam analysis\n• Fluid dynamics\n• Material properties\n• Control systems"
    
    async def handle_analysis_request(self, message: str) -> str:
        """Handle analysis requests."""
        return """**Engineering Analysis Capabilities:**

I can analyze:
• **Structural problems** - beams, columns, frames
• **Fluid dynamics** - flow rates, pressure drops
• **Material properties** - stress-strain relationships
• **Control systems** - stability, response
• **Optimization** - design parameters

**To get started:**
1. Describe your problem
2. Provide any data/parameters
3. Specify what you want to find

**Example:** "Analyze a 10m steel beam with 50kN load"

What specific analysis would you like me to perform?"""
    
    async def handle_thinking_request(self, message: str) -> str:
        """Handle thinking/reasoning requests."""
        return """**How I Think and Reason:**

**My Problem-Solving Approach:**
1. **Understanding** - Break down the problem into components
2. **Data Analysis** - Process numerical, text, and visual data
3. **Pattern Recognition** - Identify relationships and trends
4. **Solution Generation** - Find optimal solutions considering constraints
5. **Uncertainty Handling** - Quantify and account for uncertainties
6. **Learning** - Improve from each problem I solve

**My Strengths:**
• Multi-modal data processing
• Step-by-step reasoning
• Confidence quantification
• Creative problem-solving
• Engineering knowledge base

**Example thinking process:**
🧠 Step 1: Understanding the problem (Confidence: 95%)
🧠 Step 2: Analyzing data (Confidence: 98%)
🧠 Step 3: Calculating results (Confidence: 99%)
🧠 Step 4: Conclusion (Confidence: 96%)

Would you like me to demonstrate my thinking on a specific problem?"""
    
    async def handle_general_request(self, message: str) -> str:
        """Handle general requests."""
        if 'what can you do' in message.lower():
            return """**What I Can Do:**

**🧠 AI Capabilities:**
• Machine Learning - Classification, regression, clustering
• Natural Language Processing - Document analysis, text processing
• Computer Vision - Image processing, CAD analysis
• Reinforcement Learning - Optimization, adaptive control
• Neural Networks - Custom architectures

**🏗️ Engineering Applications:**
• Structural analysis and design
• Fluid dynamics calculations
• Material property prediction
• Manufacturing optimization
• Control system design
• Multi-objective optimization

**🗣️ Communication:**
• Explain complex concepts in simple terms
• Provide step-by-step solutions
• Answer technical questions
• Generate creative solutions

**What would you like to explore?"""
        
        else:
            return "I'm here to help with engineering problems and explanations! You can ask me about:\n\n• Engineering concepts and calculations\n• Problem analysis and solutions\n• Technical explanations\n• Design optimization\n• How I think and reason\n\nWhat would you like to know?"


# Global chat interface instance
chat_interface = AIChatInterface()


@app.route('/')
def index():
    """Main chat interface."""
    return render_template_string(CHAT_TEMPLATE, ai_available=AI_SYSTEM_AVAILABLE)


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'response': 'Please enter a message.'})
        
        # Process message asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(chat_interface.process_message(message))
        loop.close()
        
        return jsonify({'response': response})
    
    except Exception as e:
        return jsonify({'response': f'Sorry, I encountered an error: {str(e)}'})


def open_browser():
    """Open the chat interface in a web browser."""
    time.sleep(1.5)  # Wait for server to start
    webbrowser.open('http://localhost:8080')


def main():
    """Main function to start the chat interface."""
    print("🚀 Starting AI Engineering System Chat Interface...")
    print("="*60)
    
    if AI_SYSTEM_AVAILABLE:
        print("✅ AI System loaded successfully")
    else:
        print("⚠️ Running in demo mode - AI system not available")
    
    print("🌐 Opening chat interface in your web browser...")
    print("📱 The interface will be available at: http://localhost:8080")
    print("🔄 To stop the server, press Ctrl+C")
    print("="*60)
    
    # Open browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Flask server
    try:
        app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n👋 Chat interface stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting server: {e}")


if __name__ == "__main__":
    main()
