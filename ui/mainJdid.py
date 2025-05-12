import os
import io
import sys
import shutil
import uuid
import wave
import numpy as np
import httpx
import asyncio
from tempfile import NamedTemporaryFile
from gtts import gTTS
import chainlit as cl
import traceback
import gc

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from moonshine.moonshine.transcribe import transcribe as moonshine_transcribe
from app.services.text_extractors import (
    extract_text_from_pdf, extract_text_from_docx,
    extract_text_from_image, extract_text_from_txt,
    extract_text_from_csv
)
from app.chains.file_chain import create_chain_from_text_file

# === Cleanup temp folder
async def delete_dir_later(path):
    try:
        await asyncio.sleep(15)
        for _ in range(3):
            try:
                shutil.rmtree(path)
                break
            except PermissionError:
                await asyncio.sleep(5)
    except Exception:
        pass

# === STT
async def moonshine_speech_to_text(audio_bytes):
    with NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        audio_path = f.name
    result = moonshine_transcribe(audio_path, model="moonshine/base")
    return result[0] if result else ""

# === TTS
async def text_to_speech(text: str, lang_code: str):
    try:
        if len(text.strip()) < 2:
            return "too_short.mp3", b""
        tts = gTTS(text=text, lang=lang_code)
        with NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tts.save(f.name)
            f.seek(0)
            return f.name, f.read()
    except Exception as e:
        print("[TTS ERROR]", e)
        return "error.mp3", b""

# === Start session
@cl.on_chat_start
async def start():
    cl.user_session.set("audio_chunks", [])
    cl.user_session.set("tts_enabled", True)
    cl.user_session.set("web_search_enabled", False)
    cl.user_session.set("language", "en")
    cl.user_session.set("last_user_input", "")

    await cl.Message(
        content="Bienvenue ! Posez une question ou activez une option ci-dessous.",
        actions=[
            cl.Action(name="toggle_web_search", label="🌐 Web Search: ❌ Off", value="off", payload={"enabled": False}),
            cl.Action(name="toggle_tts", label="🔊 Activer/Désactiver la voix", value="toggle", payload={"value": "toggle"})
        ]
    ).send()

# === TTS toggle
@cl.action_callback("toggle_tts")
async def toggle_tts(action):
    current = cl.user_session.get("tts_enabled", True)
    cl.user_session.set("tts_enabled", not current)
    label = "🟢 Synthèse vocale activée" if not current else "🔇 Synthèse vocale désactivée"
    await cl.Message(content=label).send()
    await action.remove()

# === Web search toggle
@cl.action_callback("toggle_web_search")
async def toggle_web_search(action):
    current = cl.user_session.get("web_search_enabled", False)
    new_state = not current
    cl.user_session.set("web_search_enabled", new_state)
    label = "🌐 Web Search: ✅ On" if new_state else "🌐 Web Search: ❌ Off"

    await cl.Message(
        content=f"🔁 Web Search {'activé' if new_state else 'désactivé'}",
        actions=[
            cl.Action(name="toggle_web_search", label=label, value="on" if new_state else "off", payload={"enabled": new_state})
        ]
    ).send()

# === Audio handlers
@cl.on_audio_start
async def on_audio_start():
    cl.user_session.set("audio_chunks", [])
    return True

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.InputAudioChunk):
    chunks = cl.user_session.get("audio_chunks", [])
    chunks.append(np.frombuffer(chunk.data, dtype=np.int16))
    cl.user_session.set("audio_chunks", chunks)

@cl.on_audio_end
async def on_audio_end():
    if not (chunks := cl.user_session.get("audio_chunks")):
        return
    wav_buffer = io.BytesIO()
    concatenated = np.concatenate(chunks)
    with wave.open(wav_buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
        wav_file.writeframes(concatenated.tobytes())
    wav_buffer.seek(0)
    audio_bytes = wav_buffer.getvalue()
    cl.user_session.set("audio_chunks", [])
    transcription = await moonshine_speech_to_text(audio_bytes)
    cl.user_session.set("last_user_input", transcription)
    await handle_message_text(transcription)

# === User message
@cl.on_message
async def handle_message(msg: cl.Message):
    cl.user_session.set("last_user_input", msg.content.strip())
    await handle_message_text(msg.content.strip(), msg.elements)

# === Handle message based on toggle
async def handle_message_text(user_input, elements=None):
    if cl.user_session.get("web_search_enabled", False):
        await run_web_search(user_input)
    else:
        await handle_question(user_input, elements or [])

# === Web search logic
async def run_web_search(query: str):
    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post("http://localhost:8000/websummary", json={"query": query})
            data = response.json()

            # ✅ Correct nested extraction
            response_data = data if isinstance(data, dict) else {}
            summary_data = response_data.get("summary", {})  # now a dict

            # ✅ Extract actual summary and sources
            summary_text = summary_data.get("summary", "Aucun résumé.")
            sources = summary_data.get("sources", [])

            # ✅ Render sources safely
            if not sources:
                source_lines = "- Aucune source trouvée."
            else:
                source_lines = "\n".join([
                    f"- [{src.get('title', 'Source inconnue')}]({src.get('url', '#' )})" for src in sources
                ])

            await cl.Message(
                content=f"🧠 **Web Summary**\n\n{summary_text}\n\n🔗 **Sources :**\n{source_lines}"
            ).send()
    except Exception as e:
        traceback.print_exc()
        await cl.Message(content=f"❌ Erreur backend : {e}").send()

# === Local KB or /rag
async def handle_question(user_input, elements):
    tts_enabled = cl.user_session.get("tts_enabled", True)
    lang = cl.user_session.get("language", "fr")

    actions = [
        cl.Action(name="toggle_tts", label="🔊 Activer/Désactiver la voix", value="toggle", payload={"value": "toggle"}),
        cl.Action(name="toggle_web_search", label="🌐 Web Search: ✅ On" if cl.user_session.get("web_search_enabled") else "🌐 Web Search: ❌ Off", value="toggle", payload={})
    ]

    if elements:
        for file in elements:
            try:
                path = file.path
                name = file.name.lower()
                base = os.path.splitext(name)[0]

                if name.endswith(".pdf"):
                    text = extract_text_from_pdf(path)
                elif name.endswith(".docx"):
                    text = extract_text_from_docx(path)
                elif name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")):
                    text = extract_text_from_image(path)
                elif name.endswith(".txt"):
                    text = extract_text_from_txt(path)
                elif name.endswith(".csv"):
                    text = extract_text_from_csv(path)
                else:
                    await cl.Message(f"❌ Fichier non supporté : {file.name}", actions=actions).send()
                    continue

                txt_path = f"{base}_temp.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)

                kb_path = f"KB/temp_{base}_{uuid.uuid4().hex[:8]}"
                if os.path.exists(kb_path):
                    shutil.rmtree(kb_path)

                chain, kb_path = create_chain_from_text_file(txt_path, kb_path)
                result = chain.invoke({"query": user_input})
                answer = result["result"]
                del chain
                gc.collect()

                if not answer or len(answer.strip()) < 5:
                    answer = "❌ Je n'ai pas trouvé d'information pertinente pour cette question."

                if tts_enabled:
                    filename, audio_data = await text_to_speech(answer, lang)
                    if audio_data:
                        await cl.Message(content=answer, elements=[cl.Audio(mime="audio/mpeg", auto_play=True, content=audio_data)], actions=actions).send()
                    else:
                        await cl.Message(content=f"{answer}\n\n❌ Synthèse vocale indisponible.", actions=actions).send()
                else:
                    await cl.Message(content=answer, actions=actions).send()

                asyncio.create_task(delete_dir_later(kb_path))
            except Exception as e:
                traceback.print_exc()
                await cl.Message(f"❌ Erreur avec `{file.name}` : {e}", actions=actions).send()
    else:
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post("http://localhost:8000/rag", json={"question": user_input})
                response.raise_for_status()
                json_data = response.json()
                answer = json_data.get("response", "Aucune réponse.")

                # ✅ Handle case where answer is a dict
                if isinstance(answer, dict):
                    answer = answer.get("result") or answer.get("answer") or str(answer)

                # ✅ Guard against empty or irrelevant output
                if not answer or len(answer.strip()) < 5 or "aucun résultat" in answer.lower():
                    answer = "❌ Je n'ai pas trouvé d'information pertinente pour cette question."

                print("[DEBUG] Réponse de l'agent :", answer)

                if tts_enabled:
                    filename, audio_data = await text_to_speech(answer, lang)
                    if audio_data:
                        await cl.Message(content=answer, elements=[cl.Audio(mime="audio/mpeg", auto_play=True, content=audio_data)], actions=actions).send()
                    else:
                        await cl.Message(content=f"{answer}\n\n❌ Synthèse vocale indisponible.", actions=actions).send()
                else:
                    await cl.Message(content=answer, actions=actions).send()
        except Exception as e:
            traceback.print_exc()
            await cl.Message(f"❌ Erreur backend : {e}", actions=actions).send()
