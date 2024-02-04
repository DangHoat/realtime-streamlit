import logging
import os
import logging.handlers
import queue
import threading
import uuid
import time
import urllib.request
import os
import whisper
from collections import deque
from pathlib import Path
from typing import List

import av
import numpy as np
import pydub
import streamlit as st
from twilio.rest import Client
from custom.stable_whisper import stabilize_timestamps
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from test import AudioRecorder
HERE = Path(__file__).parent

logger = logging.getLogger(__name__)
model = whisper.load_model('tiny')

# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501


# This code is based on https://github.com/whitphx/streamlit-webrtc/blob/c1fe3c783c9e8042ce0c95d789e833233fd82e74/sample_utils/turn.py
@st.cache_data  # type: ignore
def get_ice_servers():
    return [{"urls": ["stun:stun.l.google.com:19302"]}]



def detect_audio_full(path):
    audio = whisper.load_audio(path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    detected_language = max(probs, key=probs.get)
    logger.debug(f"Detected language: {detected_language} file: {path}")

    results = model.transcribe(path, language=detected_language)
    return results, detected_language
    

def main():
    st.header("RealTime Speech-to-Text")
    st.markdown(
        """
        """
    )

    sound_only_page = "Ghi âm"
    with_video_page = "Ghi Hình"
    app_mode = st.selectbox("Chọn kiểu xử lý", [sound_only_page, with_video_page])

    if app_mode == sound_only_page:
        app_sst()
    elif app_mode == with_video_page:
        app_sst_with_video()


def app_sst():
    webrtc_ctx = webrtc_streamer(
        key="speech-to-text",
        mode=WebRtcMode.SENDONLY,
        audio_receiver_size=1024,
        rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": False, "audio": True},
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading...")
    lang_output = st.empty()
    text_output = st.empty()
    stream = None

    while True:
        if webrtc_ctx.audio_receiver:

            sound_chunk = pydub.AudioSegment.empty()
            try:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
            except queue.Empty:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Đang chạy. Vui lòng nói gì đó!")
            text = ''
            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                # item = f'chunk/{str(uuid.uuid4())}.wav'

                sound_chunk += sound
            file = f'chunk/{str(uuid.uuid4())}.wav'
            try:
                sound_chunk.export(file, format='wav')
                if len(sound_chunk) > 0:
                        print('abcd')
                        rs, lang = detect_audio_full(file)
                        print(rs.get('text'))
                        if rs.get('text'):
                            text += rs.get('text')
                            text_output.markdown(f"**Output :** {text}")
                            lang_output.markdown(f"**Nhận diện ngôn ngữ :** {lang or 'unknown'}")
            except Exception as e:
                print(e)
            finally :
                if os.path.exists(file):
                    os.remove(file)
                
        else:
            status_indicator.write("AudioReciver is not set. Abort.")
            break


def app_sst_with_video():
    frames_deque_lock = threading.Lock()
    frames_deque: deque = deque([])

    async def queued_audio_frames_callback(
        frames: List[av.AudioFrame],
    ) -> av.AudioFrame:
        with frames_deque_lock:
            frames_deque.extend(frames)

        # Return empty frames to be silent.
        new_frames = []
        for frame in frames:
            input_array = frame.to_ndarray()
            new_frame = av.AudioFrame.from_ndarray(
                np.zeros(input_array.shape, dtype=input_array.dtype),
                layout=frame.layout.name,
            )
            new_frame.sample_rate = frame.sample_rate
            new_frames.append(new_frame)

        return new_frames

    webrtc_ctx = webrtc_streamer(
        key="speech-to-text-w-video",
        mode=WebRtcMode.SENDRECV,
        queued_audio_frames_callback=queued_audio_frames_callback,
        rtc_configuration={"iceServers": get_ice_servers()},
        media_stream_constraints={"video": True, "audio": True},
    )

    status_indicator = st.empty()

    if not webrtc_ctx.state.playing:
        return

    status_indicator.write("Loading...")
    lang_output = st.empty()
    text_output = st.empty()
    stream = None

    while True:
        if webrtc_ctx.state.playing:

            sound_chunk = pydub.AudioSegment.empty()

            audio_frames = []
            with frames_deque_lock:
                while len(frames_deque) > 0:
                    frame = frames_deque.popleft()
                    audio_frames.append(frame)

            if len(audio_frames) == 0:
                time.sleep(0.1)
                status_indicator.write("No frame arrived.")
                continue

            status_indicator.write("Đang chạy. Vui lòng nói gì đó!")

            for audio_frame in audio_frames:
                sound = pydub.AudioSegment(
                    data=audio_frame.to_ndarray().tobytes(),
                    sample_width=audio_frame.format.bytes,
                    frame_rate=audio_frame.sample_rate,
                    channels=len(audio_frame.layout.channels),
                )
                sound_chunk += sound
            file = f'chunk/{str(uuid.uuid4())}.wav'
            try:
                sound_chunk.export(file, format='wav')
                if len(sound_chunk) > 0:
                        print('abcd')
                        rs, lang = detect_audio_full(file)
                        print(rs.get('text'))
                        if rs.get('text'):
                            text += rs.get('text')
                        text_output.markdown(f"**Output :** {text}")
                        lang_output.markdown(f"**Nhận diện ngôn ngữ :** {lang or 'unknown'}")
            except Exception as e:
                pass
            finally:
                pass
        else:
            status_indicator.write("Stopped.")
            break


if __name__ == "__main__":
    import os

    DEBUG = os.environ.get("DEBUG", "false").lower() not in ["false", "no", "0"]

    logging.basicConfig(
        format="[%(asctime)s] %(levelname)7s from %(name)s in %(pathname)s:%(lineno)d: "
        "%(message)s",
        force=True,
    )

    logger.setLevel(level=logging.DEBUG if DEBUG else logging.INFO)

    st_webrtc_logger = logging.getLogger("streamlit_webrtc")
    st_webrtc_logger.setLevel(logging.DEBUG)

    fsevents_logger = logging.getLogger("fsevents")
    fsevents_logger.setLevel(logging.WARNING)

    main()
