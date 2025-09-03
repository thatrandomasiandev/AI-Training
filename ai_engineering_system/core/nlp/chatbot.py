"""
Engineering chatbot and technical assistant.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
import openai
from langchain.llms import OpenAI
from langchain.chains import LLMChain, ConversationChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import json
import re


class EngineeringChatbot:
    """
    Engineering-focused chatbot for technical assistance.
    """
    
    def __init__(self, config):
        """
        Initialize the engineering chatbot.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI
        self.openai_client = None
        self.langchain_llm = None
        
        # Conversation memory
        self.memory = ConversationBufferMemory(return_messages=True)
        self.conversation_history = []
        
        # Engineering knowledge base
        self.engineering_knowledge = self._load_engineering_knowledge()
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize language models."""
        try:
            # Initialize OpenAI client
            self.openai_client = openai.OpenAI()
            
            # Initialize LangChain LLM
            self.langchain_llm = OpenAI(
                temperature=0.7,
                max_tokens=1000,
                model_name="gpt-3.5-turbo"
            )
            
            self.logger.info("Language models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing language models: {e}")
            self.openai_client = None
            self.langchain_llm = None
    
    def _load_engineering_knowledge(self) -> Dict[str, Any]:
        """Load engineering knowledge base."""
        return {
            "domains": {
                "structural": {
                    "keywords": ["beam", "column", "load", "stress", "strain", "deflection", "moment", "shear"],
                    "concepts": ["load analysis", "stress analysis", "deflection calculation", "safety factors"],
                    "formulas": ["stress = force/area", "deflection = (5*w*l^4)/(384*E*I)"]
                },
                "mechanical": {
                    "keywords": ["machine", "mechanism", "gear", "bearing", "motor", "engine"],
                    "concepts": ["kinematics", "dynamics", "thermodynamics", "fluid mechanics"],
                    "formulas": ["power = torque * angular_velocity", "efficiency = output/input"]
                },
                "electrical": {
                    "keywords": ["circuit", "voltage", "current", "power", "resistance", "capacitance"],
                    "concepts": ["Ohm's law", "Kirchhoff's laws", "power analysis", "circuit design"],
                    "formulas": ["V = I*R", "P = V*I", "P = I^2*R"]
                },
                "materials": {
                    "keywords": ["steel", "aluminum", "concrete", "composite", "polymer", "ceramic"],
                    "concepts": ["material properties", "stress-strain relationship", "fatigue", "creep"],
                    "formulas": ["Young's modulus = stress/strain", "yield strength", "ultimate strength"]
                }
            },
            "standards": {
                "ASTM": "American Society for Testing and Materials",
                "ISO": "International Organization for Standardization",
                "ANSI": "American National Standards Institute",
                "BS": "British Standards",
                "DIN": "Deutsches Institut für Normung",
                "JIS": "Japanese Industrial Standards"
            },
            "units": {
                "force": ["N", "kN", "MN", "lbf", "kip"],
                "stress": ["Pa", "kPa", "MPa", "GPa", "psi", "ksi"],
                "length": ["mm", "cm", "m", "in", "ft"],
                "mass": ["g", "kg", "lb", "slug"],
                "temperature": ["°C", "°F", "K"],
                "power": ["W", "kW", "MW", "hp"],
                "frequency": ["Hz", "kHz", "MHz", "GHz"]
            }
        }
    
    async def chat(self, message: str, context: Optional[str] = None) -> str:
        """
        Chat with the engineering assistant.
        
        Args:
            message: User message
            context: Optional context for the conversation
            
        Returns:
            Assistant response
        """
        self.logger.info(f"Processing chat message: {message[:100]}...")
        
        # Analyze the message
        analysis = self._analyze_message(message)
        
        # Generate response based on analysis
        if analysis["type"] == "question":
            response = await self._answer_question(message, analysis, context)
        elif analysis["type"] == "calculation":
            response = await self._perform_calculation(message, analysis)
        elif analysis["type"] == "explanation":
            response = await self._provide_explanation(message, analysis)
        elif analysis["type"] == "design":
            response = await self._assist_design(message, analysis)
        else:
            response = await self._general_response(message, analysis)
        
        # Update conversation history
        self._update_conversation_history(message, response)
        
        return response
    
    def _analyze_message(self, message: str) -> Dict[str, Any]:
        """Analyze the user message to determine intent and domain."""
        message_lower = message.lower()
        
        # Determine message type
        message_type = "general"
        if any(word in message_lower for word in ["calculate", "compute", "find", "determine"]):
            message_type = "calculation"
        elif any(word in message_lower for word in ["what", "how", "why", "explain", "describe"]):
            message_type = "question"
        elif any(word in message_lower for word in ["design", "create", "develop", "build"]):
            message_type = "design"
        elif any(word in message_lower for word in ["explain", "describe", "tell me about"]):
            message_type = "explanation"
        
        # Determine engineering domain
        domain = self._identify_domain(message)
        
        # Extract technical terms
        technical_terms = self._extract_technical_terms(message)
        
        # Check for formulas or calculations
        has_formulas = bool(re.search(r'[=<>≤≥\+\-\*\/\^\(\)]', message))
        
        return {
            "type": message_type,
            "domain": domain,
            "technical_terms": technical_terms,
            "has_formulas": has_formulas,
            "complexity": self._assess_complexity(message)
        }
    
    def _identify_domain(self, message: str) -> str:
        """Identify the engineering domain from the message."""
        message_lower = message.lower()
        
        domain_scores = {}
        for domain, info in self.engineering_knowledge["domains"].items():
            score = sum(1 for keyword in info["keywords"] if keyword in message_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        else:
            return "general"
    
    def _extract_technical_terms(self, message: str) -> List[str]:
        """Extract technical terms from the message."""
        technical_terms = []
        
        # Extract measurements
        measurement_pattern = r'\d+(?:\.\d+)?\s*(?:mm|cm|m|in|ft|MPa|GPa|psi|ksi|N|kN|MN|kg|g|V|A|W|kW|MW|°C|°F|K|rpm|Hz|kHz|MHz|GHz)'
        measurements = re.findall(measurement_pattern, message, re.IGNORECASE)
        technical_terms.extend(measurements)
        
        # Extract engineering keywords
        for domain, info in self.engineering_knowledge["domains"].items():
            for keyword in info["keywords"]:
                if keyword in message.lower():
                    technical_terms.append(keyword)
        
        return list(set(technical_terms))
    
    def _assess_complexity(self, message: str) -> str:
        """Assess the complexity of the message."""
        technical_terms = self._extract_technical_terms(message)
        has_formulas = bool(re.search(r'[=<>≤≥\+\-\*\/\^\(\)]', message))
        
        if len(technical_terms) > 5 or has_formulas:
            return "high"
        elif len(technical_terms) > 2:
            return "medium"
        else:
            return "low"
    
    async def _answer_question(self, message: str, analysis: Dict[str, Any], context: Optional[str] = None) -> str:
        """Answer engineering questions."""
        domain = analysis["domain"]
        technical_terms = analysis["technical_terms"]
        
        # Build context for the response
        response_context = f"You are an expert engineering assistant specializing in {domain} engineering. "
        
        if context:
            response_context += f"Context: {context}. "
        
        if technical_terms:
            response_context += f"Technical terms mentioned: {', '.join(technical_terms)}. "
        
        # Get domain-specific knowledge
        if domain in self.engineering_knowledge["domains"]:
            domain_info = self.engineering_knowledge["domains"][domain]
            response_context += f"Key concepts in {domain} engineering: {', '.join(domain_info['concepts'])}. "
        
        # Generate response using OpenAI
        if self.openai_client:
            try:
                response = await self._generate_openai_response(message, response_context)
                return response
            except Exception as e:
                self.logger.error(f"Error generating OpenAI response: {e}")
        
        # Fallback to rule-based response
        return self._generate_rule_based_response(message, analysis)
    
    async def _perform_calculation(self, message: str, analysis: Dict[str, Any]) -> str:
        """Perform engineering calculations."""
        domain = analysis["domain"]
        
        # Extract numerical values and units
        values = self._extract_values_and_units(message)
        
        if not values:
            return "I couldn't identify the values and units needed for the calculation. Please provide specific numerical values with their units."
        
        # Try to identify the calculation type
        calculation_type = self._identify_calculation_type(message, domain)
        
        if calculation_type:
            result = self._perform_specific_calculation(calculation_type, values, domain)
            if result:
                return f"Based on the {calculation_type} calculation:\n{result}"
        
        # General calculation assistance
        return f"I can help with {domain} calculations. I found these values: {values}. Could you specify what calculation you need to perform?"
    
    def _extract_values_and_units(self, message: str) -> List[Dict[str, Any]]:
        """Extract numerical values and units from the message."""
        pattern = r'(\d+(?:\.\d+)?)\s*(mm|cm|m|in|ft|MPa|GPa|psi|ksi|N|kN|MN|kg|g|V|A|W|kW|MW|°C|°F|K|rpm|Hz|kHz|MHz|GHz)'
        matches = re.findall(pattern, message, re.IGNORECASE)
        
        values = []
        for match in matches:
            values.append({
                "value": float(match[0]),
                "unit": match[1]
            })
        
        return values
    
    def _identify_calculation_type(self, message: str, domain: str) -> Optional[str]:
        """Identify the type of calculation needed."""
        message_lower = message.lower()
        
        calculation_types = {
            "stress": ["stress", "force", "area"],
            "deflection": ["deflection", "displacement", "bending"],
            "power": ["power", "torque", "speed", "rpm"],
            "efficiency": ["efficiency", "output", "input"],
            "safety_factor": ["safety", "factor", "allowable", "ultimate"]
        }
        
        for calc_type, keywords in calculation_types.items():
            if sum(1 for keyword in keywords if keyword in message_lower) >= 2:
                return calc_type
        
        return None
    
    def _perform_specific_calculation(self, calculation_type: str, values: List[Dict[str, Any]], domain: str) -> Optional[str]:
        """Perform specific engineering calculations."""
        if calculation_type == "stress" and len(values) >= 2:
            # Find force and area values
            force = next((v for v in values if v["unit"] in ["N", "kN", "MN", "lbf"]), None)
            area = next((v for v in values if v["unit"] in ["mm²", "cm²", "m²", "in²"]), None)
            
            if force and area:
                stress = force["value"] / area["value"]
                return f"Stress = Force / Area = {force['value']} {force['unit']} / {area['value']} {area['unit']} = {stress:.2f} {force['unit']}/{area['unit']}"
        
        elif calculation_type == "power" and len(values) >= 2:
            # Find torque and speed values
            torque = next((v for v in values if v["unit"] in ["N⋅m", "lb⋅ft"]), None)
            speed = next((v for v in values if v["unit"] in ["rpm", "rad/s"]), None)
            
            if torque and speed:
                if speed["unit"] == "rpm":
                    angular_velocity = speed["value"] * 2 * 3.14159 / 60  # Convert to rad/s
                else:
                    angular_velocity = speed["value"]
                
                power = torque["value"] * angular_velocity
                return f"Power = Torque × Angular Velocity = {torque['value']} {torque['unit']} × {angular_velocity:.2f} rad/s = {power:.2f} W"
        
        return None
    
    async def _provide_explanation(self, message: str, analysis: Dict[str, Any]) -> str:
        """Provide explanations of engineering concepts."""
        domain = analysis["domain"]
        technical_terms = analysis["technical_terms"]
        
        # Build explanation context
        explanation_context = f"Explain the following {domain} engineering concept in simple terms: {message}"
        
        if technical_terms:
            explanation_context += f" Focus on these technical terms: {', '.join(technical_terms)}"
        
        # Generate explanation using OpenAI
        if self.openai_client:
            try:
                response = await self._generate_openai_response(message, explanation_context)
                return response
            except Exception as e:
                self.logger.error(f"Error generating explanation: {e}")
        
        # Fallback to rule-based explanation
        return self._generate_rule_based_explanation(technical_terms, domain)
    
    async def _assist_design(self, message: str, analysis: Dict[str, Any]) -> str:
        """Assist with engineering design tasks."""
        domain = analysis["domain"]
        
        design_context = f"You are helping with {domain} engineering design. Provide practical design guidance for: {message}"
        
        if self.openai_client:
            try:
                response = await self._generate_openai_response(message, design_context)
                return response
            except Exception as e:
                self.logger.error(f"Error generating design assistance: {e}")
        
        return f"I can help with {domain} design. Please provide more specific requirements for your design project."
    
    async def _general_response(self, message: str, analysis: Dict[str, Any]) -> str:
        """Generate general engineering responses."""
        domain = analysis["domain"]
        
        general_context = f"You are an engineering assistant. Respond helpfully to: {message}"
        
        if self.openai_client:
            try:
                response = await self._generate_openai_response(message, general_context)
                return response
            except Exception as e:
                self.logger.error(f"Error generating general response: {e}")
        
        return "I'm here to help with engineering questions and problems. How can I assist you today?"
    
    async def _generate_openai_response(self, message: str, context: str) -> str:
        """Generate response using OpenAI API."""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": context},
                    {"role": "user", "content": message}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise
    
    def _generate_rule_based_response(self, message: str, analysis: Dict[str, Any]) -> str:
        """Generate rule-based response as fallback."""
        domain = analysis["domain"]
        technical_terms = analysis["technical_terms"]
        
        if domain in self.engineering_knowledge["domains"]:
            domain_info = self.engineering_knowledge["domains"][domain]
            concepts = domain_info["concepts"]
            
            response = f"In {domain} engineering, key concepts include: {', '.join(concepts[:3])}. "
            
            if technical_terms:
                response += f"I noticed you mentioned: {', '.join(technical_terms[:3])}. "
            
            response += "Could you provide more specific details about what you'd like to know?"
            
            return response
        
        return "I can help with various engineering topics. Please provide more specific information about your question."
    
    def _generate_rule_based_explanation(self, technical_terms: List[str], domain: str) -> str:
        """Generate rule-based explanation as fallback."""
        if not technical_terms:
            return f"I can explain {domain} engineering concepts. Please specify which concept you'd like me to explain."
        
        explanations = {
            "stress": "Stress is the internal force per unit area within a material when subjected to external loads.",
            "strain": "Strain is the deformation of a material relative to its original dimensions.",
            "deflection": "Deflection is the displacement of a structural element under load.",
            "moment": "Moment is the tendency of a force to cause rotation about a point or axis.",
            "shear": "Shear is a force that causes one part of a material to slide past another part."
        }
        
        for term in technical_terms:
            if term in explanations:
                return explanations[term]
        
        return f"I can explain {domain} engineering concepts. The terms you mentioned ({', '.join(technical_terms)}) are important in engineering analysis."
    
    def _update_conversation_history(self, user_message: str, assistant_response: str):
        """Update conversation history."""
        self.conversation_history.append({
            "user": user_message,
            "assistant": assistant_response,
            "timestamp": asyncio.get_event_loop().time()
        })
        
        # Keep only last 10 exchanges
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history."""
        return self.conversation_history
    
    def clear_conversation_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.memory.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """Get chatbot status."""
        return {
            "openai_available": self.openai_client is not None,
            "langchain_available": self.langchain_llm is not None,
            "conversation_length": len(self.conversation_history),
            "domains_available": list(self.engineering_knowledge["domains"].keys()),
            "standards_available": list(self.engineering_knowledge["standards"].keys())
        }


class TechnicalAssistant(EngineeringChatbot):
    """
    Specialized technical assistant with enhanced capabilities.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.specialized_knowledge = self._load_specialized_knowledge()
    
    def _load_specialized_knowledge(self) -> Dict[str, Any]:
        """Load specialized technical knowledge."""
        return {
            "formulas": {
                "structural": {
                    "beam_deflection": "δ = (5*w*l^4)/(384*E*I)",
                    "stress": "σ = F/A",
                    "moment": "M = F*d",
                    "shear": "V = F"
                },
                "mechanical": {
                    "power": "P = T*ω",
                    "efficiency": "η = P_out/P_in",
                    "kinetic_energy": "KE = 0.5*m*v^2",
                    "potential_energy": "PE = m*g*h"
                },
                "electrical": {
                    "ohm_law": "V = I*R",
                    "power": "P = V*I",
                    "resistance_series": "R_total = R1 + R2 + ...",
                    "resistance_parallel": "1/R_total = 1/R1 + 1/R2 + ..."
                }
            },
            "material_properties": {
                "steel": {
                    "density": "7850 kg/m³",
                    "young_modulus": "200 GPa",
                    "yield_strength": "250 MPa",
                    "ultimate_strength": "400 MPa"
                },
                "aluminum": {
                    "density": "2700 kg/m³",
                    "young_modulus": "70 GPa",
                    "yield_strength": "95 MPa",
                    "ultimate_strength": "186 MPa"
                },
                "concrete": {
                    "density": "2400 kg/m³",
                    "compressive_strength": "25 MPa",
                    "tensile_strength": "2.5 MPa"
                }
            }
        }
    
    async def provide_material_properties(self, material: str) -> str:
        """Provide material properties information."""
        material_lower = material.lower()
        
        for mat_name, properties in self.specialized_knowledge["material_properties"].items():
            if mat_name in material_lower:
                response = f"Properties of {mat_name}:\n"
                for prop, value in properties.items():
                    response += f"- {prop.replace('_', ' ').title()}: {value}\n"
                return response
        
        return f"I don't have specific properties for {material}. I have data for: {', '.join(self.specialized_knowledge['material_properties'].keys())}"
    
    async def provide_formula(self, domain: str, formula_type: str) -> str:
        """Provide engineering formulas."""
        if domain in self.specialized_knowledge["formulas"]:
            formulas = self.specialized_knowledge["formulas"][domain]
            if formula_type in formulas:
                return f"{formula_type.replace('_', ' ').title()}: {formulas[formula_type]}"
            else:
                available = list(formulas.keys())
                return f"Available formulas for {domain}: {', '.join(available)}"
        
        return f"I don't have formulas for {domain}. Available domains: {', '.join(self.specialized_knowledge['formulas'].keys())}"
