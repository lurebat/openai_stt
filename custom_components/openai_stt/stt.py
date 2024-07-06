"""
Support for OpenAI STT.
"""
import logging
from typing import AsyncIterable
import async_timeout
import voluptuous as vol
from homeassistant.components.tts import CONF_LANG
from homeassistant.components.stt import (
    AudioBitRates,
    AudioChannels,
    AudioCodecs,
    AudioFormats,
    AudioSampleRates,
    Provider,
    SpeechMetadata,
    SpeechResult,
    SpeechResultState,
)
import homeassistant.helpers.config_validation as cv
import wave
import io
import openai

_LOGGER = logging.getLogger(__name__)

CONF_API_KEY = 'api_key'
DEFAULT_LANG = 'en-US'
OPENAI_STT_URL = "https://api.openai.com/v1/audio/transcriptions"
CONF_MODEL = 'model'
CONF_URL = 'url'
CONF_PROMPT = 'prompt'
CONF_TEMPERATURE = 'temperature'

PLATFORM_SCHEMA = cv.PLATFORM_SCHEMA.extend({
    vol.Required(CONF_API_KEY): cv.string,
    vol.Optional(CONF_LANG, default=DEFAULT_LANG): cv.string,
    vol.Optional(CONF_MODEL, default='whisper'): cv.string,
    vol.Optional(CONF_URL, default=None): cv.string,
    vol.Optional(CONF_PROMPT, default=None): cv.string,
    vol.Optional(CONF_TEMPERATURE, default=0.1): cv.positive_float,
})

async def async_get_engine(hass, config, discovery_info=None):
    """Set up OpenAI STT speech component."""
    api_key = config[CONF_API_KEY]
    languages = config.get(CONF_LANG, DEFAULT_LANG)
    model = config.get(CONF_MODEL)
    url = config.get('url')
    prompt = config.get('prompt')
    temperature = config.get('temperature')
    return OpenAISTTProvider(hass, api_key, languages, model, url, prompt, temperature)

class OpenAISTTProvider(Provider):
    """The OpenAI STT API provider."""

    def __init__(self, hass, api_key, lang, model, url, prompt, temperature):
        """Initialize OpenAI STT provider."""
        self.hass = hass
        self._api_key = api_key
        self._language = lang
        self._model = model
        self._url = url
        self._prompt = prompt or f"You are transcribing a command to a virtual assistant. The possible languages are {self.supported_languages}." 
        self._temperature = temperature

    @property
    def default_language(self) -> str:
        """Return the default language."""
        return self._language.split(',')[0]

    @property
    def supported_languages(self) -> list[str]:
        """Return the list of supported languages."""
        # Ideally, this list should be dynamically fetched from OpenAI, if supported.
        return self._language.split(',')

    @property
    def supported_formats(self) -> list[AudioFormats]:
        """Return a list of supported formats."""
        return [AudioFormats.WAV]

    @property
    def supported_codecs(self) -> list[AudioCodecs]:
        """Return a list of supported codecs."""
        return [AudioCodecs.PCM]

    @property
    def supported_bit_rates(self) -> list[AudioBitRates]:
        """Return a list of supported bitrates."""
        return [AudioBitRates.BITRATE_16]

    @property
    def supported_sample_rates(self) -> list[AudioSampleRates]:
        """Return a list of supported samplerates."""
        return [AudioSampleRates.SAMPLERATE_16000]

    @property
    def supported_channels(self) -> list[AudioChannels]:
        """Return a list of supported channels."""
        return [AudioChannels.CHANNEL_MONO]

    async def async_process_audio_stream(
        self, metadata: SpeechMetadata, stream: AsyncIterable[bytes]
    ) -> SpeechResult:
        if metadata.format != AudioFormats.WAV:
            _LOGGER.error("Unsupported audio format: %s", metadata.format)
            return SpeechResult("", SpeechResultState.ERROR)
        
        data = b''

        async for chunk in stream:
            data += chunk

        wav_stream = io.BytesIO()

        with wave.open(wav_stream, 'w') as wav_file:
            wav_file.setnchannels(metadata.channel)
            wav_file.setsampwidth(metadata.bit_rate // 8)
            wav_file.setframerate(metadata.sample_rate)

            wav_file.writeframes(data)
            
        wav_stream.seek(0)

        async with openai.AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._url
        ) as openai_client:
            try:
                res = await openai_client.audio.transcriptions.create(
                    file=("audio.wav", wav_stream),
                    model=self._model,
                    prompt=self._prompt,
                    temperature=self._temperature,
                    response_format="text",
                    timeout=20
                )

                if res is None:
                    return SpeechResult("Couldn't transcribe text", SpeechResultState.ERROR)
                
                return SpeechResult(res.text, SpeechResultState.SUCCESS)
            
            except Exception as e:
                _LOGGER.error("Failed to transcribe audio: %s", repr(e))
                return SpeechResult(str(e), SpeechResultState.ERROR)
