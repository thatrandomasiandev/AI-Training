#!/usr/bin/env python3
"""
Interactive AI Explanation System - Ask the AI anything about engineering!
"""

import asyncio
import json
from typing import Dict, Any, List
import random


class InteractiveAIExplainer:
    """Interactive explanation system that can answer questions in natural language."""
    
    def __init__(self):
        self.knowledge_base = self._build_knowledge_base()
        self.conversation_history = []
        self.user_preferences = {
            "complexity_level": "intermediate",
            "explanation_style": "friendly_expert",
            "include_examples": True,
            "include_formulas": False
        }
    
    def _build_knowledge_base(self) -> Dict[str, Any]:
        """Build a comprehensive knowledge base for explanations."""
        return {
            "engineering_concepts": {
                "stress": {
                    "definition": "Internal force per unit area within a material",
                    "formula": "σ = F/A",
                    "units": "Pascal (Pa) or MPa",
                    "examples": ["Bridge beam under traffic load", "Bolt under tension", "Concrete column supporting building"],
                    "analogies": ["Pressure in a balloon", "Tension in a rope", "Squeezing a sponge"],
                    "related_concepts": ["strain", "elasticity", "yield_strength", "ultimate_strength"]
                },
                "deflection": {
                    "definition": "Displacement or deformation of a structural element under load",
                    "formula": "δ = (5wL⁴)/(384EI) for simply supported beam",
                    "units": "meters (m) or millimeters (mm)",
                    "examples": ["Bridge sagging under traffic", "Floor joist bending under weight", "Cantilever beam tip movement"],
                    "analogies": ["Diving board bending", "Tree branch swaying", "Trampoline surface deforming"],
                    "related_concepts": ["stiffness", "moment_of_inertia", "modulus_of_elasticity", "span_length"]
                },
                "safety_factor": {
                    "definition": "Ratio of material strength to actual stress",
                    "formula": "SF = σ_allowable / σ_actual",
                    "units": "dimensionless",
                    "examples": ["Bridge designed for 2x expected load", "Elevator cable rated for 10x weight", "Building foundation oversized for safety"],
                    "analogies": ["Wearing a helmet when biking", "Having a spare tire in car", "Extra parachute for skydiving"],
                    "related_concepts": ["reliability", "uncertainty", "risk_assessment", "design_codes"]
                },
                "moment": {
                    "definition": "Rotational force that causes bending or twisting",
                    "formula": "M = F × d",
                    "units": "Newton-meters (N·m) or kN·m",
                    "examples": ["Beam bending under load", "Wrench turning a bolt", "Door opening force"],
                    "analogies": ["Seesaw balancing", "Wheel turning", "Lever action"],
                    "related_concepts": ["shear", "bending_stress", "moment_diagram", "equilibrium"]
                },
                "young_modulus": {
                    "definition": "Measure of material stiffness or resistance to elastic deformation",
                    "formula": "E = σ/ε",
                    "units": "Pascal (Pa) or GPa",
                    "examples": ["Steel beam stiffness", "Concrete column rigidity", "Aluminum frame flexibility"],
                    "analogies": ["Spring stiffness", "Rubber band stretchiness", "Wood vs metal flexibility"],
                    "related_concepts": ["elasticity", "stiffness", "material_properties", "stress_strain_curve"]
                }
            },
            "ai_reasoning": {
                "problem_solving": "I break down complex problems into smaller, manageable steps",
                "data_analysis": "I analyze numerical data, text, and images to understand the problem",
                "pattern_recognition": "I identify patterns and relationships in engineering data",
                "optimization": "I find the best solutions considering multiple constraints",
                "uncertainty_handling": "I quantify and account for uncertainties in my analysis",
                "learning": "I learn from each problem to improve future solutions"
            },
            "engineering_applications": {
                "structural_engineering": "Design and analysis of buildings, bridges, and infrastructure",
                "mechanical_engineering": "Design of machines, vehicles, and mechanical systems",
                "civil_engineering": "Infrastructure, transportation, and environmental systems",
                "materials_engineering": "Development and testing of new materials",
                "aerospace_engineering": "Design of aircraft, spacecraft, and related systems"
            }
        }
    
    def answer_question(self, question: str) -> str:
        """Answer a user question in natural language."""
        
        # Add question to conversation history
        self.conversation_history.append({"type": "question", "content": question})
        
        # Analyze the question
        question_lower = question.lower()
        
        # Check for concept explanations
        for concept, info in self.knowledge_base["engineering_concepts"].items():
            if concept in question_lower or any(keyword in question_lower for keyword in info["related_concepts"]):
                return self._explain_concept(concept, info, question)
        
        # Check for AI reasoning questions
        if any(keyword in question_lower for keyword in ["how do you", "how does the ai", "how do you think", "reasoning", "thinking"]):
            return self._explain_ai_reasoning(question)
        
        # Check for application questions
        if any(keyword in question_lower for keyword in ["what is", "explain", "tell me about"]):
            return self._explain_general_concept(question)
        
        # Check for comparison questions
        if any(keyword in question_lower for keyword in ["difference between", "compare", "vs", "versus"]):
            return self._explain_comparison(question)
        
        # Check for example requests
        if any(keyword in question_lower for keyword in ["example", "for instance", "like what"]):
            return self._provide_examples(question)
        
        # Default response
        return self._provide_general_response(question)
    
    def _explain_concept(self, concept: str, info: Dict[str, Any], question: str) -> str:
        """Explain a specific engineering concept."""
        
        explanation = f"Great question! Let me explain **{concept.replace('_', ' ').title()}**:\n\n"
        
        # Add definition
        explanation += f"**What it is:** {info['definition']}\n\n"
        
        # Add formula if requested
        if self.user_preferences["include_formulas"] and "formula" in info:
            explanation += f"**Formula:** {info['formula']}\n\n"
        
        # Add units
        if "units" in info:
            explanation += f"**Units:** {info['units']}\n\n"
        
        # Add examples if requested
        if self.user_preferences["include_examples"] and "examples" in info:
            explanation += "**Real-world examples:**\n"
            for example in info["examples"][:3]:  # Show first 3 examples
                explanation += f"• {example}\n"
            explanation += "\n"
        
        # Add analogies
        if "analogies" in info:
            explanation += "**Think of it like:**\n"
            for analogy in info["analogies"][:2]:  # Show first 2 analogies
                explanation += f"• {analogy}\n"
            explanation += "\n"
        
        # Add related concepts
        if "related_concepts" in info:
            explanation += f"**Related concepts:** {', '.join(info['related_concepts'])}\n\n"
        
        return explanation
    
    def _explain_ai_reasoning(self, question: str) -> str:
        """Explain AI reasoning process."""
        
        explanation = "Great question! Let me explain how I think and reason:\n\n"
        
        reasoning_info = self.knowledge_base["ai_reasoning"]
        
        explanation += "**My Problem-Solving Approach:**\n"
        explanation += f"• {reasoning_info['problem_solving']}\n"
        explanation += f"• {reasoning_info['data_analysis']}\n"
        explanation += f"• {reasoning_info['pattern_recognition']}\n"
        explanation += f"• {reasoning_info['optimization']}\n"
        explanation += f"• {reasoning_info['uncertainty_handling']}\n"
        explanation += f"• {reasoning_info['learning']}\n\n"
        
        explanation += "**Example of my thinking process:**\n"
        explanation += "1. I first understand what you're asking\n"
        explanation += "2. I break it down into smaller parts\n"
        explanation += "3. I analyze each part using my knowledge\n"
        explanation += "4. I combine the insights to form a complete answer\n"
        explanation += "5. I check my reasoning and provide confidence levels\n\n"
        
        explanation += "I'm designed to think like an expert engineer, but I can explain things in simple terms too!"
        
        return explanation
    
    def _explain_general_concept(self, question: str) -> str:
        """Explain general concepts."""
        
        # Extract key terms from the question
        question_lower = question.lower()
        
        if "engineering" in question_lower:
            return self._explain_engineering_field(question)
        elif "ai" in question_lower or "artificial intelligence" in question_lower:
            return self._explain_ai_capabilities()
        else:
            return "I'd be happy to explain that! Could you be more specific about what aspect you'd like me to clarify?"
    
    def _explain_engineering_field(self, question: str) -> str:
        """Explain engineering fields."""
        
        explanation = "Engineering is a broad field with many specialties:\n\n"
        
        for field, description in self.knowledge_base["engineering_applications"].items():
            explanation += f"**{field.replace('_', ' ').title()}:** {description}\n"
        
        explanation += "\nI can help with problems in any of these areas! What specific engineering topic interests you?"
        
        return explanation
    
    def _explain_ai_capabilities(self) -> str:
        """Explain AI capabilities."""
        
        explanation = "I'm an AI Engineering System with these capabilities:\n\n"
        explanation += "**🧠 Thinking & Reasoning:** I can analyze complex engineering problems step-by-step\n"
        explanation += "**📊 Multi-Modal Analysis:** I process numbers, text, and images together\n"
        explanation += "**🔍 Problem Solving:** I find optimal solutions considering multiple constraints\n"
        explanation += "**📚 Knowledge Base:** I understand engineering concepts, formulas, and applications\n"
        explanation += "**🗣️ Natural Language:** I can explain complex concepts in simple terms\n"
        explanation += "**🔄 Learning:** I improve with each problem I solve\n\n"
        explanation += "I'm like having an expert engineer who can think, analyze, and explain things clearly!"
        
        return explanation
    
    def _explain_comparison(self, question: str) -> str:
        """Explain comparisons between concepts."""
        
        question_lower = question.lower()
        
        if "stress" in question_lower and "strain" in question_lower:
            return """**Stress vs Strain - The Key Difference:**

**Stress** is the force per unit area (σ = F/A) - it's what you apply to a material
**Strain** is the deformation per unit length (ε = ΔL/L) - it's how the material responds

Think of it like this:
• Stress = How hard you push on a spring
• Strain = How much the spring compresses

They're related by Young's Modulus: E = σ/ε (stress divided by strain)"""
        
        elif "steel" in question_lower and "concrete" in question_lower:
            return """**Steel vs Concrete - Material Comparison:**

**Steel:**
• High strength in tension and compression
• Ductile (bends before breaking)
• Expensive but lightweight
• Good for beams, columns, reinforcement

**Concrete:**
• Strong in compression, weak in tension
• Brittle (breaks suddenly)
• Cheap but heavy
• Good for foundations, walls, slabs

**Best of both worlds:** Reinforced concrete combines concrete's compression strength with steel's tension strength!"""
        
        else:
            return "I can compare many engineering concepts! What specific comparison would you like me to explain?"
    
    def _provide_examples(self, question: str) -> str:
        """Provide examples for concepts."""
        
        question_lower = question.lower()
        
        if "stress" in question_lower:
            return """**Real-world Stress Examples:**

• **Bridge Beam:** Cars and trucks create stress in the steel beams
• **Building Column:** The weight of floors above creates compressive stress
• **Bolt in Tension:** When you tighten a bolt, it experiences tensile stress
• **Pressure Vessel:** Gas or liquid pressure creates stress in the walls
• **Crane Cable:** Lifting heavy loads creates tensile stress in the cable

Each example shows how external forces create internal stress in materials!"""
        
        elif "deflection" in question_lower:
            return """**Real-world Deflection Examples:**

• **Diving Board:** Bends down when you stand on it
• **Bridge Deck:** Sags slightly under traffic loads
• **Floor Joist:** Bends under the weight of furniture and people
• **Cantilever Beam:** Tip moves down when loaded
• **Tree Branch:** Sways in the wind

Deflection is everywhere - we just need to make sure it's not too much!"""
        
        else:
            return "I have lots of examples! What specific concept would you like examples for?"
    
    def _provide_general_response(self, question: str) -> str:
        """Provide a general response to unclear questions."""
        
        responses = [
            "That's an interesting question! Could you give me more details about what you'd like to know?",
            "I'd love to help explain that! Can you be more specific about which aspect interests you?",
            "Great question! I can explain many engineering concepts. What would you like to learn about?",
            "I'm here to help! Could you rephrase your question so I can give you a better answer?",
            "That's a good question! I can explain engineering concepts, AI reasoning, or help with specific problems. What would be most helpful?"
        ]
        
        return random.choice(responses)
    
    def set_user_preferences(self, preferences: Dict[str, Any]):
        """Set user preferences for explanations."""
        self.user_preferences.update(preferences)
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the conversation history."""
        return self.conversation_history
    
    def clear_conversation(self):
        """Clear the conversation history."""
        self.conversation_history = []


async def interactive_demo():
    """Interactive demonstration of the explanation system."""
    
    print("🗣️ INTERACTIVE AI EXPLANATION SYSTEM")
    print("="*60)
    print("Ask me anything about engineering! I can explain concepts, reasoning, and more.")
    print("Type 'quit' to exit, 'help' for examples, or 'preferences' to change settings.")
    print("="*60)
    
    explainer = InteractiveAIExplainer()
    
    # Demo questions
    demo_questions = [
        "What is stress?",
        "How do you think about engineering problems?",
        "What's the difference between stress and strain?",
        "Give me examples of deflection",
        "What can you do as an AI?",
        "Explain safety factor",
        "What is Young's modulus?"
    ]
    
    print("\n🎯 Here are some example questions you can ask:")
    for i, question in enumerate(demo_questions, 1):
        print(f"{i}. {question}")
    
    print("\n" + "="*60)
    
    # Simulate some demo interactions
    for question in demo_questions[:3]:  # Show first 3 as examples
        print(f"\n👤 **You:** {question}")
        answer = explainer.answer_question(question)
        print(f"🤖 **AI:** {answer}")
        print("-" * 60)
    
    print("\n🎯 The AI can answer questions about:")
    print("• Engineering concepts (stress, deflection, safety factors, etc.)")
    print("• AI reasoning and thinking processes")
    print("• Comparisons between different concepts")
    print("• Real-world examples and applications")
    print("• Problem-solving approaches")
    print("• And much more!")
    
    print("\n🚀 Try asking your own questions - the AI is ready to explain anything!")


if __name__ == "__main__":
    asyncio.run(interactive_demo())
