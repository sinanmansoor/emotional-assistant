import cv2
import time
import logging
from src.emotion_recognition import EmotionRecognizer
from src.response_generator import ResponseGenerator
from src.audio_output import AudioOutput

class EmotionalAssistant:
    def __init__(self):
        """
        Initialize emotional assistant components
        """
        self.emotion_recognizer = EmotionRecognizer()
        self.response_generator = ResponseGenerator()
        self.audio_output = AudioOutput()
        
        # Configuration
        self.response_interval = 5  # Seconds between responses
        self.last_response_time = 0
    
    def run(self):
        """
        Main application loop for real-time emotion detection
        """
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        # Configure video capture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        try:
            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    logging.error("Failed to capture frame")
                    break
                
                # Detect faces
                faces = self.emotion_recognizer.detect_faces(frame)
                
                for (x, y, w, h) in faces:
                    # Extract face region
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Preprocess and classify emotion
                    processed_face = self.emotion_recognizer.preprocess_face(face_img)
                    emotion = self.emotion_recognizer.classify_emotion(processed_face)
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, emotion, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    # Generate response at intervals
                    current_time = time.time()
                    if current_time - self.last_response_time > self.response_interval:
                        response = self.response_generator.generate_empathetic_response(emotion)
                        print(f"AI Response ({emotion}): {response}")
                        
                        # Generate audio response
                        self.audio_output.text_to_speech(response)
                        
                        # Update last response time
                        self.last_response_time = current_time
                
                # Display frame
                cv2.imshow('Emotional Assistant', frame)
                
                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except Exception as e:
            logging.error(f"An error occurred: {e}")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()

def main():
    """
    Entry point for the Emotional Assistant
    """
    logging.basicConfig(level=logging.INFO)
    assistant = EmotionalAssistant()
    assistant.run()

if __name__ == "__main__":
    main()