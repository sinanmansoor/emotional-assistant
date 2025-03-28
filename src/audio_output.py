from gtts import gTTS
import os
import playsound
import logging

class AudioOutput:
    @staticmethod
    def text_to_speech(text, language='en', slow=False):
        """
        Convert text to speech and play audio
        
        Args:
            text (str): Text to convert
            language (str): Language code
            slow (bool): Speak slowly if True
        """
        try:
            # Create temporary audio file
            tts = gTTS(text=text, lang=language, slow=slow)
            audio_file = "temp_response.mp3"
            tts.save(audio_file)
            
            # Play audio
            playsound.playsound(audio_file)
            
            # Remove temporary file
            os.remove(audio_file)
        
        except Exception as e:
            logging.error(f"Audio generation error: {e}")