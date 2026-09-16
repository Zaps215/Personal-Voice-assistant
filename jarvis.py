import speech_recognition as sr
import pyttsx3
import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq

# Load environment variables from .env file
load_dotenv()

# ----------------------------
# 1. Choose your AI model
# ----------------------------
# Valid Groq model choices: "llama-3.3-70b-versatile" or "llama3-70b-8192"
model = Groq(id="openai/gpt-oss-20b")


# ----------------------------
# 2. Text-to-Speech (voice output)
# ----------------------------
# Initialize globally so garbage collection doesn't destroy callbacks while speaking
engine = pyttsx3.init()
engine.setProperty('rate', 175)
engine.setProperty('volume', 0.9)

def speak(text):
    """Convert text to speech"""
    try:
        print(f"JARVIS: {text}")
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"TTS Error: {e}")
        print(f"JARVIS would have said: {text}")


# ----------------------------
# 3. Speech-to-Text using sounddevice (NO PyAudio!)
# ----------------------------
def listen():
    """Listen for voice input and convert to text"""
    recognizer = sr.Recognizer()
    fs = 44100  # Sample rate (Hz)
    duration = 5  # Max seconds to listen

    try:
        import sounddevice as sd
    except (ImportError, OSError) as e:
        print(f"⚠️ Audio backend unavailable: {e}")
        print("Install PortAudio (for example, the libportaudio2 system package) and try again.")
        return None

    print("\n🎤 Listening... (speak now)")

    try:
        # Record audio using sounddevice
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()  # Wait until recording is done

        # Convert to AudioData for speech_recognition
        audio_bytes = recording.tobytes()
        audio_data = sr.AudioData(audio_bytes, sample_rate=fs, sample_width=2)

        # Recognize speech
        text = recognizer.recognize_google(audio_data)
        print(f"🗣️ You: {text}")
        return text.lower()

    except sd.PortAudioError as e:
        print(f"⚠️ Microphone error: {e}")
        print("Make sure your microphone is connected and not being used by another app.")
        return None
    except sr.UnknownValueError:
        print("❌ Sorry, I didn't catch that. Please repeat.")
        return None
    except sr.RequestError:
        print("🌐 Network error. Check your internet connection.")
        return None
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
        return None


# ----------------------------
# 4. Create the JARVIS Agent
# ----------------------------
def create_jarvis():
    """Create the JARVIS agent with personality"""
    return Agent(
        model=model,
        description="""
        You are JARVIS, Tony Stark's witty and helpful AI assistant.
        You are sophisticated, intelligent, and have a British charm.
        You assist the user with various tasks and answer questions.
        """,
        instructions=[
            "Be helpful, clever, and conversational.",
            "Keep responses concise and natural (under 3 sentences when possible).",
            "Use a professional but friendly tone.",
            "If you don't know something, say so rather than making it up.",
            "Address the user as 'sir' or by their name if you know it."
        ],
    )


# ----------------------------
# 5. Main Loop
# ----------------------------
def main():
    print("=" * 50)
    print("🚀 J.A.R.V.I.S. Initializing...")
    print("=" * 50)
    print("Say 'exit', 'goodbye', or 'shutdown' to quit.")
    print("=" * 50)

    # Greet the user
    speak("Good day, sir. JARVIS at your service.")

    # Create the agent
    agent = create_jarvis()

    while True:
        command = listen()

        if command is None:
            continue

        # Check for exit commands
        if command in ["exit", "quit", "goodbye", "shutdown", "bye", "see you"]:
            speak("Goodbye, sir. It was a pleasure serving you.")
            break

        # Process the command
        print("🤔 Thinking...")
        try:
            response = agent.run(command)
            reply = response.content if hasattr(response, 'content') else str(response)
            speak(reply)
        except Exception as e:
            print(f"⚠️ Agent error: {e}")
            speak("I'm having some trouble processing that. Could you repeat?")


if __name__ == "__main__":
    main()