import cv2
from deepface import DeepFace
import pygame
import speech_recognition as sr

# Initialize pygame mixer for playing audio
pygame.mixer.init()

# Define the emotion-to-song mapping
emotion_song_map = {
    "happy": ["happy1.mp3", "happy2.mp3", "happy3.mp3"],
    "angry": ["angry1.mp3", "angry2.mp3"],
    "sad": ["sad1.mp3", "sad2.mp3"],
    "surprise": ["surprise1.mp3", "surprise2.mp3"],
    "fear": ["fear1.mp3", "fear2.mp3"],
    "neutral": ["neutral1.mp3", "neutral2.mp3"],
    "disgust": ["disgust1.mp3", "disgust2.mp3"],
}

# Define the emotion-to-activity suggestion mapping
emotion_activity_map = {
    "happy": "🎉 Feeling joyful? Try dancing, celebrating with friends, or solving a fun puzzle!",
    "angry": "🎯 Feeling frustrated? Let off steam with a workout or creative expression.",
    "sad": "🧩 Feeling blue? Solve a puzzle or watch a cheerful movie.",
    "surprise": "📖 Feeling surprised? Dive into trivia or explore a new hobby!",
    "fear": "🛡️ Feeling uneasy? Practice meditation or calming breathing exercises.",
    "neutral": "☕ Feeling calm? Enjoy journaling or savor a cup of tea.",
    "disgust": "🍴 Feeling off? Tidy up your space or cook something fresh.",
}

# Initialize variables for the program state
last_emotion = None  # Store the last detected emotion
emotion_index = {emotion: 0 for emotion in emotion_song_map}  # Initialize song indices


def play_song(emotion):
    """Play a song based on detected emotion."""
    if emotion in emotion_song_map:
        songs = emotion_song_map[emotion]
        song_file = songs[emotion_index[emotion]]

        try:
            pygame.mixer.music.load(song_file)
            pygame.mixer.music.play()
            print(f"🎵 Now playing {emotion} song: {song_file}")
        except pygame.error as e:
            print(f"❌ Error playing song: {e}")
    else:
        print(f"❌ No songs found for emotion: {emotion}")


def draw_emotions_on_frame(frame, emotions, dominant_emotion):
    """Annotate the frame with emotion results."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Display the dominant emotion
    cv2.putText(frame, f'Dominant Emotion: {dominant_emotion}', (10, 40), font, 1, (0, 255, 0), 2)

    # Display other emotions and their confidence scores
    y_offset = 70
    for emotion, score in emotions.items():
        text = f"{emotion.capitalize()}: {score:.2f}%"
        cv2.putText(frame, text, (10, y_offset), font, 0.8, (0, 0, 0), 2)
        y_offset += 30

    return frame


def process_voice_command():
    """Process voice commands for playback and additional features."""
    global last_emotion
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("🎤 Listening for voice commands...")
        try:
            audio = recognizer.listen(source, timeout=10)
            command = recognizer.recognize_google(audio).lower()
            print(f"🎙️ You said: {command}")

            if "stop" in command or "halt" in command:
                pygame.mixer.music.stop()
                print("🛑 Music stopped.")
            elif ("play" in command or "resume" in command) and last_emotion:
                play_song(last_emotion)
            elif "pause" in command or "hold" in command:
                pygame.mixer.music.pause()
                print("⏸️ Music paused.")
            elif "next" in command and last_emotion:
                emotion_index[last_emotion] = (emotion_index[last_emotion] + 1) % len(emotion_song_map[last_emotion])
                play_song(last_emotion)
            elif "what's playing" in command and last_emotion:
                print(f"🎵 Currently playing: {emotion_song_map[last_emotion][emotion_index[last_emotion]]}")
            elif "recommend something" in command and last_emotion:
                print(f"💡 Suggestion: {emotion_activity_map[last_emotion]}")
            else:
                print("❌ Command not recognized.")
        except sr.UnknownValueError:
            print("❌ Could not understand your voice.")
        except sr.RequestError as e:
            print(f"❌ Error with speech recognition service: {e}")


# Webcam Initialization
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Unable to access webcam.")
    exit()

print("✅ Webcam ready. Press SPACE for emotion analysis, V for voice command, or Q to quit.")

cv2.namedWindow("Emotion Detection")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Webcam feed lost.")
        break

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # SPACE for Emotion Analysis
        temp_img = "temp_capture.jpg"
        cv2.imwrite(temp_img, frame)
        try:
            result = DeepFace.analyze(img_path=temp_img, actions=['emotion'])
            emotions = result[0]['emotion']
            dominant_emotion = result[0]['dominant_emotion']
            last_emotion = dominant_emotion
            print(f"😊 Detected Emotion: {dominant_emotion}")

            # Annotate the frame with emotion details
            frame = draw_emotions_on_frame(frame, emotions, dominant_emotion)

            # Play a song based on emotion
            play_song(dominant_emotion)

            # Show the updated frame
            cv2.imshow("Emotion Detection", frame)
        except Exception as e:
            print(f"❌ Emotion analysis failed: {e}")

    elif key == ord('v'):  # V for Voice Commands
        process_voice_command()

    elif key == ord('q'):  # Q to Quit
        print("👋 Exiting.")
        break

pygame.mixer.music.stop()
cap.release()
cv2.destroyAllWindows()