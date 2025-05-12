import whisper
from gtts import gTTS
from tempfile import NamedTemporaryFile

whisper_model = whisper.load_model("base")

def transcribe(audio_bytes):
    with NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name

    result = whisper_model.transcribe(audio_path)
    return result["text"], result.get("language", "fr")

def synthesize(text, lang):
    try:
        tts = gTTS(text=text, lang=lang)
        with NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts.save(f.name)
        with open(f.name, "rb") as audio_file:
            return audio_file.read()
    except Exception:
        return b""
