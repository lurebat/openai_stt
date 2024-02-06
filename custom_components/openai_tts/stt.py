"""
Support for OpenAI STT.
"""
import logging
from typing import AsyncIterable
import httpx
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
from homeassistant.core import HomeAssistant
import homeassistant.helpers.config_validation as cv
import wave
import io

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
    vol.Optional(CONF_MODEL, default='whisper-1'): cv.string,
    vol.Optional(CONF_URL, default=None): cv.string,
    vol.Optional(CONF_PROMPT, default=None): cv.string,
    vol.Optional(CONF_TEMPERATURE, default=0): cv.positive_int,
})

async def async_get_engine(hass, config, discovery_info=None):
    """Set up OpenAI STT speech component."""
    api_key = config[CONF_API_KEY]
    language = config.get(CONF_LANG, DEFAULT_LANG)
    model = config.get(CONF_MODEL)
    url = config.get('url')
    prompt = config.get('prompt')
    temperature = config.get('temperature')
    return OpenAISTTProvider(hass, api_key, language, model, url, prompt, temperature)

class OpenAISTTProvider(Provider):
    """The OpenAI STT API provider."""

    def __init__(self, hass, api_key, lang, model, url, prompt, temperature):
        """Initialize OpenAI STT provider."""
        self.hass = hass
        self._api_key = api_key
        self._language = lang
        self._model = model
        self._url = url
        self._prompt = prompt
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
            wav_file.setsampwidth(2)
            wav_file.setframerate(metadata.sample_rate)

            wav_file.writeframes(data)
            

        wav =  wav_stream.getvalue() # wav file object 


        # OpenAI API endpoint for audio transcription
        url = self._url or OPENAI_STT_URL
        # Your OpenAI API key
        headers = {
            'Authorization': f'Bearer {self._api_key}',
            'api-key': self._api_key,
            'Content-Type': 'multipart/form-data'
        }
        # Prepare the data for the POST request
        files = {
            'file': ('filename.wav', wav, 'audio/wav')
        }
        data = {
            'model': self._model,
            'language': self._language,
            'temperature': self._temperature,
            'prompt': self._prompt
        }
        async with async_timeout.timeout(10):
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, files=files, data=data)
                if response.status_code == 200:
                    return SpeechResult(
                        response.text,
                        SpeechResultState.SUCCESS,
                    )
                else:
                    _LOGGER.error("%s", response.text)
                    return SpeechResult("", SpeechResultState.ERROR)
