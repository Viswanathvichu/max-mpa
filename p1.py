
import streamlit as st
import google.generativeai as genai
from datetime import datetime
import json
import re
from typing import Dict, List, Optional, Tuple, Union
import os
from dotenv import load_dotenv
import logging
from dataclasses import dataclass, asdict
import numpy as np
from pathlib import Path
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv("C:/Users/viswa/Desktop/Chatbot/Shridevi/p2.env")
print(os.environ) 
# Constants
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Enhanced data models using dataclasses
@dataclass
class NutritionalInfo:
    calories: int
    protein: float
    carbs: float
    fats: float
    fiber: float

@dataclass
class UserInfo:
    height: Optional[float] = None
    weight: Optional[float] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    activity_level: Optional[str] = None
    goal: Optional[str] = None
    restrictions: List[str] = None
    preferences: List[str] = None
    meals_per_day: Optional[int] = None
    diet_type: Optional[str] = None
    cooking_time: Optional[float] = None
    
    def __post_init__(self):
        self.restrictions = self.restrictions or []
        self.preferences = self.preferences or []
    
    def to_dict(self) -> Dict:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def calculate_bmr(self) -> Optional[float]:
        """Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation"""
        try:
            if all(v is not None for v in [self.weight, self.height, self.age, self.gender]):
                # Convert height to cm if needed
                height_cm = self.height if self.height > 3 else self.height * 100
                
                if self.gender.lower() == 'male':
                    bmr = (10 * self.weight) + (6.25 * height_cm) - (5 * self.age) + 5
                else:
                    bmr = (10 * self.weight) + (6.25 * height_cm) - (5 * self.age) - 161
                
                return round(bmr, 2)
            return None
        except Exception as e:
            logger.error(f"Error calculating BMR: {str(e)}")
            return None
    
    def calculate_tdee(self) -> Optional[float]:
        """Calculate Total Daily Energy Expenditure"""
        activity_multipliers = {
            'sedentary': 1.2,
            'lightly active': 1.375,
            'moderately active': 1.55,
            'very active': 1.725,
            'extra active': 1.9
        }
        
        try:
            bmr = self.calculate_bmr()
            if bmr and self.activity_level:
                multiplier = activity_multipliers.get(self.activity_level.lower(), 1.2)
                return round(bmr * multiplier, 2)
            return None
        except Exception as e:
            logger.error(f"Error calculating TDEE: {str(e)}")
            return None

@dataclass
class ChatMemory:
    name: Optional[str] = None
    preferences: Dict = None
    context: List[Dict] = None
    max_context_length: int = 10
    
    def __post_init__(self):
        self.preferences = self.preferences or {}
        self.context = self.context or []
    
    def add_context(self, user_input: str, bot_response: str):
        """Add a new conversation to the context"""
        try:
            self.context.append({
                'user_input': user_input,
                'bot_response': bot_response,
                'timestamp': str(datetime.now())
            })
            if len(self.context) > self.max_context_length:
                self.context = self.context[-self.max_context_length:]
        except Exception as e:
            logger.error(f"Error adding context: {str(e)}")
    
    def get_recent_context(self, num_messages: int = 3) -> str:
        """Get recent conversation context"""
        try:
            context = ""
            if self.name:
                context += f"The user's name is {self.name}. "
            
            if self.context:
                context += "Recent conversations:\n"
                for conv in self.context[-num_messages:]:
                    context += f"User: {conv['user_input']}\nAssistant: {conv['bot_response']}\n"
            
            return context
        except Exception as e:
            logger.error(f"Error getting recent context: {str(e)}")
            return ""

# Enhanced questionnaire with more detailed options
DIET_QUESTIONS = [
    {
        "question": "What is your height in centimeters? (or enter in feet and inches as 5'10\")",
        "validation": lambda x: validate_height(x),
        "error_message": "Please enter a valid height between 100 and 250 cm or in feet and inches format (e.g., 5'10\")",
        "transform": lambda x: convert_height_to_cm(x)
    },
    {
        "question": "What is your weight in kilograms? (or enter in pounds followed by 'lbs')",
        "validation": lambda x: validate_weight(x),
        "error_message": "Please enter a valid weight between 30 and 300 kg or in pounds (e.g., 150 lbs)",
        "transform": lambda x: convert_weight_to_kg(x)
    },
    {
        "question": "What is your age?",
        "validation": lambda x: 12 <= int(x) <= 120,
        "error_message": "Please enter a valid age between 12 and 120 years.",
        "transform": lambda x: int(x)
    },
    {
        "question": "What is your gender? (male/female)",
        "validation": lambda x: x.lower() in ['male', 'female'],
        "error_message": "Please enter either 'male' or 'female'.",
        "transform": lambda x: x.lower()
    },
    {
        "question": "What is your activity level?\n- Sedentary (little or no exercise)\n- Lightly active (exercise 1-3 times/week)\n- Moderately active (exercise 3-5 times/week)\n- Very active (exercise 6-7 times/week)\n- Extra active (very intense exercise daily)",
        "validation": lambda x: x.lower() in ['sedentary', 'lightly active', 'moderately active', 'very active', 'extra active'],
        "error_message": "Please choose one of the listed activity levels.",
        "transform": lambda x: x.lower()
    },
    {
        "question": "What is your goal? (lose weight/gain weight/maintain weight)",
        "validation": lambda x: x.lower() in ["lose weight", "gain weight", "maintain weight"],
        "error_message": "Please choose one of: lose weight, gain weight, maintain weight",
        "transform": lambda x: x.lower()
    },
    {
        "question": "Do you have any dietary restrictions or allergies? (Type 'none' if none)",
        "validation": lambda x: True,
        "error_message": "",
        "transform": lambda x: [r.strip() for r in x.split(',') if r.strip().lower() != 'none']
    },
    {
        "question": "What are your favorite foods or cuisines?",
        "validation": lambda x: len(x.strip()) > 0,
        "error_message": "Please enter at least one food or cuisine preference.",
        "transform": lambda x: [p.strip() for p in x.split(',')]
    },
    {
        "question": "How many meals do you prefer per day? (2-6)",
        "validation": lambda x: 2 <= int(x) <= 6,
        "error_message": "Please enter a number between 2 and 6.",
        "transform": lambda x: int(x)
    },
    {
        "question": "Do you prefer vegetarian, non-vegetarian, or both types of food?",
        "validation": lambda x: x.lower() in ["vegetarian", "non-vegetarian", "both"],
        "error_message": "Please choose one of: vegetarian, non-vegetarian, both",
        "transform": lambda x: x.lower()
    },
    {
        "question": "How many hours can you spend on cooking per meal? (0.25-3)",
        "validation": lambda x: 0.25 <= float(x) <= 3,
        "error_message": "Please enter a time between 0.25 (15 minutes) and 3 hours.",
        "transform": lambda x: float(x)
    }
]

# Utility functions
def validate_height(height_str: str) -> bool:
    """Validate height input in either cm or feet'inches\" format"""
    try:
        if "'" in height_str:  # feet and inches format
            feet_str, inches_str = height_str.split("'")
            inches_str = inches_str.replace('"', '')
            feet = float(feet_str)
            inches = float(inches_str) if inches_str else 0
            cm = (feet * 30.48) + (inches * 2.54)
            return 100 <= cm <= 250
        else:  # centimeter format
            return 100 <= float(height_str) <= 250
    except ValueError:
        return False

def validate_weight(weight_str: str) -> bool:
    """Validate weight input in either kg or lbs format"""
    try:
        if 'lbs' in weight_str.lower():  # pounds format
            lbs = float(weight_str.lower().replace('lbs', '').strip())
            kg = lbs * 0.45359237
            return 30 <= kg <= 300
        else:  # kilogram format
            return 30 <= float(weight_str) <= 300
    except ValueError:
        return False

def convert_height_to_cm(height_str: str) -> float:
    """Convert height input to centimeters"""
    try:
        if "'" in height_str:  # feet and inches format
            feet_str, inches_str = height_str.split("'")
            inches_str = inches_str.replace('"', '')
            feet = float(feet_str)
            inches = float(inches_str) if inches_str else 0
            return round((feet * 30.48) + (inches * 2.54), 2)
        else:  # centimeter format
            return float(height_str)
    except ValueError as e:
        logger.error(f"Error converting height: {str(e)}")
        raise

def convert_weight_to_kg(weight_str: str) -> float:
    """Convert weight input to kilograms"""
    try:
        if 'lbs' in weight_str.lower():  # pounds format
            lbs = float(weight_str.lower().replace('lbs', '').strip())
            return round(lbs * 0.45359237, 2)
        else:  # kilogram format
            return float(weight_str)
    except ValueError as e:
        logger.error(f"Error converting weight: {str(e)}")
        raise

def calculate_bmi(height_cm: float, weight_kg: float) -> Optional[float]:
    """Calculate BMI with input validation"""
    try:
        if not (100 <= height_cm <= 250 and 30 <= weight_kg <= 300):
            return None
        height_m = height_cm / 100
        bmi = weight_kg / (height_m * height_m)
        return round(bmi, 2)
    except Exception as e:
        logger.error(f"Error calculating BMI: {str(e)}")
        return None

def get_bmi_category(bmi: float) -> str:
    """Get BMI category with health recommendations"""
    if bmi < 18.5:
        return "Underweight", "Consider consulting with a healthcare provider about healthy weight gain strategies."
    elif 18.5 <= bmi < 25:
        return "Normal weight", "Maintain your healthy lifestyle with balanced nutrition and regular exercise."
    elif 25 <= bmi < 30:
        return "Overweight", "Focus on portion control and increasing physical activity for gradual weight loss."
    else:
        return "Obese", "Consult with healthcare providers about developing a comprehensive weight management plan."

def generate_diet_plan() -> str:
    """Generate a personalized diet plan based on user information"""
    try:
        user_info = st.session_state.user_info
        bmi = calculate_bmi(user_info.height, user_info.weight)
        
        if bmi is None:
            return "Error: Invalid height or weight values."
        
        bmi_category, health_recommendation = get_bmi_category(bmi)
        tdee = user_info.calculate_tdee()
        
        if tdee is None:
            return "Error: Unable to calculate daily energy needs."
        
        # Adjust calories based on goal
        if user_info.goal == "lose weight":
            target_calories = tdee - 500  # 500 calorie deficit for weight loss
        elif user_info.goal == "gain weight":
            target_calories = tdee + 500  # 500 calorie surplus for weight gain
        else:
            target_calories = tdee
        
        context = f"""
User Profile:
- BMI: {bmi} ({bmi_category})
- TDEE: {tdee} calories/day
- Target calories: {target_calories} calories/day
- Age: {user_info.age}
- Gender: {user_info.gender}
- Activity level: {user_info.activity_level}
- Goal: {user_info.goal}
- Dietary restrictions: {', '.join(user_info.restrictions) or 'None'}
- Food preferences: {', '.join(user_info.preferences)}
- Meals per day: {user_info.meals_per_day}
- Diet type: {user_info.diet_type}
- Cooking time per meal: {user_info.cooking_time} hours

Health Recommendation: {health_recommendation}

Previous context:
{st.session_state.memory.get_recent_context()}

Based on this information, please provide:
1. A summary of their nutritional needs
2. A detailed 7-day diet plan with {user_info.meals_per_day} meals that:
   - Provides {target_calories} calories per day
   - Aligns with their food preferences and restrictions
   - Supports their {user_info.goal} goal
   - Includes specific portion sizes and macronutrient breakdowns
   - Fits within their {user_info.cooking_time}-hour cooking time constraint
   - Provides variety and balanced nutrition
3. Tips for meal prep and planning
4. Recommended supplements if needed
5. Exercise recommendations that complement their diet plan

Format the response clearly with sections and bullet points for easy reading.
"""
        
        response = model.generate_content(context)
        return response.text
    except Exception as e:
        logger.error(f"Error generating diet plan: {str(e)}")
        return f"Error generating diet plan: {str(e)}"

def validate_answer(question_index: int, answer: str) -> Tuple[bool, str]:
    """Validate user answers to diet questions"""
    try:
        question = DIET_QUESTIONS[question_index]
        is_valid = question["validation"](answer)
        return is_valid, "" if is_valid else question["error_message"]
    except (ValueError, IndexError) as e:
        logger.error(f"Error validating answer: {str(e)}")
        return False, "Invalid input. Please try again."

def handle_user_input(user_input: str) -> None:
    """Process user input and generate appropriate responses"""
    if not user_input.strip():
        return

    try:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        if st.session_state.questioning_mode:
            is_valid, error_message = validate_answer(st.session_state.current_question_index, user_input)
            
            if not is_valid:
                response = error_message
            else:
                # Transform and store answer
                current_question = DIET_QUESTIONS[st.session_state.current_question_index]
                transformed_answer = current_question["transform"](user_input)
                
                # Update user info based on question index
                if st.session_state.current_question_index == 0:
                    st.session_state.user_info.height = transformed_answer
                elif st.session_state.current_question_index == 1:
                    st.session_state.user_info.weight = transformed_answer
                elif st.session_state.current_question_index == 2:
                    st.session_state.user_info.age = transformed_answer
                elif st.session_state.current_question_index == 3:
                    st.session_state.user_info.gender = transformed_answer
                elif st.session_state.current_question_index == 4:
                    st.session_state.user_info.activity_level = transformed_answer
                elif st.session_state.current_question_index == 5:
                    st.session_state.user_info.goal = transformed_answer
                elif st.session_state.current_question_index == 6:
                    st.session_state.user_info.restrictions = transformed_answer
                elif st.session_state.current_question_index == 7:
                    st.session_state.user_info.preferences = transformed_answer
                elif st.session_state.current_question_index == 8:
                    st.session_state.user_info.meals_per_day = transformed_answer
                elif st.session_state.current_question_index == 9:
                    st.session_state.user_info.diet_type = transformed_answer
                elif st.session_state.current_question_index == 10:
                    st.session_state.user_info.cooking_time = transformed_answer
                
                st.session_state.current_question_index += 1
                
                if st.session_state.current_question_index < len(DIET_QUESTIONS):
                    response = DIET_QUESTIONS[st.session_state.current_question_index]["question"]
                else:
                    st.session_state.questioning_mode = False
                    response = "Great! Let me generate your personalized diet plan..."
                    response += "\n\n" + generate_diet_plan()
        else:
            # Regular chat mode
            if "diet plan" in user_input.lower():
                st.session_state.questioning_mode = True
                st.session_state.current_question_index = 0
                response = "Let's create your personalized diet plan! " + DIET_QUESTIONS[0]["question"]
            else:
                try:
                    context = f"""You are MAX, a professional diet planning assistant. Previous context:
{st.session_state.memory.get_recent_context()}

User: {user_input}
Assistant: """
                    response = model.generate_content(context).text
                except Exception as e:
                    response = f"I apologize, but I encountered an error: {str(e)}"
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.session_state.memory.add_context(user_input, response)
    except Exception as e:
        logger.error(f"Error handling user input: {str(e)}")
        st.error("An error occurred while processing your request. Please try again.")

# Page Configuration and Styling
st.set_page_config(
    page_title="MAX - Diet Assistant",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Apply custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize Gemini API
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
except Exception as e:
    st.error(f"Error initializing Gemini API: {str(e)}")
    st.stop()

# Initialize session states if not exists
def init_session_state():
    """Initialize all session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'user_info' not in st.session_state:
        st.session_state.user_info = UserInfo()
    if 'memory' not in st.session_state:
        st.session_state.memory = ChatMemory()
    if 'current_question_index' not in st.session_state:
        st.session_state.current_question_index = 0
    if 'questioning_mode' not in st.session_state:
        st.session_state.questioning_mode = False

# Main application
def main():
    """Main application function"""
    try:
        # Initialize session state
        init_session_state()
        
        # Display header
        st.markdown('<h1 class="main-header">MAX - Your Personal Diet Planning Assistant 🥗</h1>', 
                   unsafe_allow_html=True)
        
        # Welcome message for new sessions
        if not st.session_state.chat_history:
            st.markdown("""
            <div class="chat-message assistant-message">
            👋 Hello! I'm MAX, your personal diet planning assistant. I can help you:
            
            • Create a personalized diet plan
            • Calculate your BMI and caloric needs
            • Provide evidence-based nutritional advice
            • Suggest healthy recipes and meal ideas
            
            Type "diet plan" to start creating your personalized plan!
            </div>
            """, unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            message_type = "user-message" if message["role"] == "user" else "assistant-message"
            st.markdown(
                f'<div class="chat-message {message_type}">'
                f'{"You" if message["role"] == "user" else "MAX"}: {message["content"]}'
                '</div>',
                unsafe_allow_html=True
            )
        
        # Input area
        with st.container():
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            
            # Create a form for input
            with st.form(key='chat_form', clear_on_submit=True):
                # Create two columns for input field and submit button
                col1, col2 = st.columns([6, 1])
                
                with col1:
                    user_input = st.text_input(
                        "Type your message here...",
                        key="user_message",
                        label_visibility="collapsed"
                    )
                
                with col2:
                    submit_button = st.form_submit_button(
                        "Send",
                        use_container_width=True,
                        type="primary"
                    )
                
                # Handle form submission
                if submit_button and user_input:
                    handle_user_input(user_input)
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Add bottom padding
        st.markdown("<div style='padding-bottom: 100px'></div>", unsafe_allow_html=True)
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An error occurred. Please refresh the page and try again.")

if __name__ == "__main__":
    main()