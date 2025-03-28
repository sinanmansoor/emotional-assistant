from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class ResponseGenerator:
    def __init__(self, 
                 model_name="microsoft/DialoGPT-small", 
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize conversational AI model
        
        Args:
            model_name (str): Hugging Face model identifier
            device (str): Computing device (CPU/GPU)
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
    
    def generate_empathetic_response(self, emotion):
        """
        Generate an empathetic response based on detected emotion
        
        Args:
            emotion (str): Detected emotion
        
        Returns:
            str: Generated conversational response
        """
        empathy_prompts = {
            'Happy': "I'm glad you're feeling happy! What's making you smile today?",
            'Sad': "I'm here for you. Would you like to talk about what's troubling you?",
            'Angry': "I sense you're feeling frustrated. Let's take a deep breath together.",
            'Surprise': "Wow, something interesting just happened! Tell me more.",
            'Fear': "It's okay to feel anxious. I'm here to support you.",
            'Disgust': "Sounds like something is really bothering you. Want to discuss it?",
            'Neutral': "How are you feeling right now? I'm here to listen."
        }
        
        prompt = empathy_prompts.get(emotion, "How can I help you today?")
        
        # Encode input and generate response
        input_ids = self.tokenizer.encode(prompt + self.tokenizer.eos_token, return_tensors='pt').to(self.device)
        
        output = self.model.generate(
            input_ids, 
            max_length=100, 
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            top_k=50,
            top_p=0.95,
            temperature=0.7
        )
        
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response