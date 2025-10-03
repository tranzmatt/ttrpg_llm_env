# TTRPG LLM Inference Script
# Use your trained TTRPG GM model for interactive conversations

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from typing import List, Dict
import os

class TTRPGGameMaster:
    """A class to handle TTRPG GM conversations using the trained model"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the TTRPG Game Master
        
        Args:
            model_path: Path to the trained model directory
            device: Device to run on ("cuda", "cpu", or "auto")
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        
        self._load_model()
    
    def _setup_device(self, device: str) -> str:
        """Setup the appropriate device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self):
        """Load the trained model and tokenizer"""
        print(f"Loading TTRPG GM model from: {self.model_path}")
        print(f"Using device: {self.device}")
        
        try:
            # Try loading with AutoModelForCausalLM first
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                low_cpu_mem_usage=True
            )
            
            # Set pad token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure you've run the training script first and the model path is correct.")
            raise
    
    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation for the model"""
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback formatting
            formatted = ""
            for msg in messages:
                if msg["role"] == "system":
                    formatted += f"System: {msg['content']}\n\n"
                elif msg["role"] == "user":
                    formatted += f"User: {msg['content']}\n\n"
                elif msg["role"] == "assistant":
                    formatted += f"Assistant: {msg['content']}\n\n"
            formatted += "Assistant: "
            return formatted
    
    def ask_gm(self, question: str, context: str = "", use_history: bool = True) -> str:
        """
        Ask the GM a question
        
        Args:
            question: The question to ask
            context: Additional context for the question
            use_history: Whether to include conversation history
            
        Returns:
            The GM's response
        """
        # Prepare the conversation
        messages = [
            {
                "role": "system", 
                "content": "You are an expert Game Master for tabletop RPGs. You have extensive knowledge of game rules, mechanics, and storytelling. Provide helpful, creative, and practical advice to enhance the gaming experience."
            }
        ]
        
        # Add conversation history if requested
        if use_history and self.conversation_history:
            messages.extend(self.conversation_history[-6:])  # Last 6 messages
        
        # Add current question
        user_message = question
        if context:
            user_message = f"{question}\n\nContext: {context}"
        
        messages.append({"role": "user", "content": user_message})
        
        # Format for the model
        prompt = self._format_conversation(messages)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024  # Leave room for generation
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the new response
        response = full_response[len(prompt):].strip()
        
        # Update conversation history
        if use_history:
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def ask_gm_streaming(self, question: str, context: str = "") -> None:
        """
        Ask the GM a question with streaming response
        
        Args:
            question: The question to ask
            context: Additional context for the question
        """
        messages = [
            {
                "role": "system", 
                "content": "You are an expert Game Master for tabletop RPGs. Provide helpful, creative, and practical advice."
            },
            {"role": "user", "content": f"{question}\n\nContext: {context}" if context else question}
        ]
        
        prompt = self._format_conversation(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        streamer = TextStreamer(self.tokenizer, skip_prompt=True)
        
        print(f"\nGM Response:")
        print("-" * 40)
        
        with torch.no_grad():
            _ = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                streamer=streamer,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        print("\n" + "-" * 40)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared.")
    
    def save_conversation(self, filename: str):
        """Save conversation history to a file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
        print(f"Conversation saved to: {filename}")
    
    def load_conversation(self, filename: str):
        """Load conversation history from a file"""
        with open(filename, 'r', encoding='utf-8') as f:
            self.conversation_history = json.load(f)
        print(f"Conversation loaded from: {filename}")

def interactive_session(gm_model: TTRPGGameMaster):
    """Run an interactive session with the GM"""
    print("\n" + "="*60)
    print("🎲 WELCOME TO YOUR TTRPG GAME MASTER AI! 🎲")
    print("="*60)
    print("Ask me anything about rules, mechanics, or GM advice!")
    print("Commands:")
    print("  'quit' or 'exit' - End the session")
    print("  'clear' - Clear conversation history")
    print("  'save <filename>' - Save conversation")
    print("  'load <filename>' - Load conversation")
    print("  'stream <question>' - Get streaming response")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n🎮 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("\n👋 Thanks for playing! Good luck with your campaigns!")
                break
            
            elif user_input.lower() == 'clear':
                gm_model.clear_history()
                continue
            
            elif user_input.lower().startswith('save '):
                filename = user_input[5:].strip() or "conversation.json"
                gm_model.save_conversation(filename)
                continue
            
            elif user_input.lower().startswith('load '):
                filename = user_input[5:].strip()
                if os.path.exists(filename):
                    gm_model.load_conversation(filename)
                else:
                    print(f"File {filename} not found.")
                continue
            
            elif user_input.lower().startswith('stream '):
                question = user_input[7:].strip()
                if question:
                    gm_model.ask_gm_streaming(question)
                continue
            
            elif not user_input:
                continue
            
            # Get GM response
            print("\n🎲 GM: ", end="", flush=True)
            response = gm_model.ask_gm(user_input)
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 Session interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def test_model_examples(gm_model: TTRPGGameMaster):
    """Test the model with example questions"""
    test_questions = [
        {
            "question": "How do I handle a player who wants to attempt something not covered by the rules?",
            "context": ""
        },
        {
            "question": "What's the best way to balance combat encounters?",
            "context": "I have 4 level 3 players in D&D 5e"
        },
        {
            "question": "How should I interpret this spell?",
            "context": "A player wants to use Misty Step to teleport through a wall they can't see through"
        },
        {
            "question": "What do I do when players break the game with creative interpretations?",
            "context": "Players are trying to use cantrips in ways that seem overpowered"
        }
    ]
    
    print("\n" + "="*50)
    print("🧪 TESTING MODEL WITH EXAMPLE QUESTIONS")
    print("="*50)
    
    for i, example in enumerate(test_questions, 1):
        print(f"\n📝 Test {i}: {example['question']}")
        if example['context']:
            print(f"Context: {example['context']}")
        print("-" * 40)
        
        response = gm_model.ask_gm(example['question'], example['context'], use_history=False)
        print(f"🎲 GM Response: {response}")
        print("-" * 40)

def main():
    """Main function to run the inference script"""
    # Configuration
    model_path = "./trained_ttrpg_gm"  # Path to your trained model
    
    print("TTRPG GM Model Inference")
    print("=" * 40)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found at: {model_path}")
        print("Please train the model first using train_ttrpg_llm.py")
        return
    
    try:
        # Load the model
        gm_model = TTRPGGameMaster(model_path)
        
        # Check GPU memory if available
        if torch.cuda.is_available():
            print(f"🔥 GPU Memory: {torch.cuda.get_device_name()}")
            print(f"   Allocated: {torch.cuda.memory_allocated()/1024**3:.1f}GB")
        
        # Ask user what they want to do
        print("\nWhat would you like to do?")
        print("1. Interactive session")
        print("2. Test with examples")
        print("3. Single question")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            interactive_session(gm_model)
        elif choice == "2":
            test_model_examples(gm_model)
        elif choice == "3":
            question = input("\nEnter your question: ").strip()
            if question:
                print("\n🎲 GM Response:")
                print("-" * 40)
                response = gm_model.ask_gm(question)
                print(response)
        else:
            print("Invalid choice. Running interactive session by default.")
            interactive_session(gm_model)
    
    except Exception as e:
        print(f"❌ Error initializing GM model: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure the model path is correct")
        print("2. Check that you have enough GPU/RAM")
        print("3. Verify all dependencies are installed")

if __name__ == "__main__":
    main()