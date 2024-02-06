"""
Support for OpenAI STT.
"""
import logging
import requests
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

_LOGGER = logging.getLogger(__name__)

CONF_API_KEY = 'api_key'
DEFAULT_LANG = 'en-US'
OPENAI_STT_URL = "https://api.openai.com/v1/audio/transcriptions"
CONF_MODEL = 'model'

PLATFORM_SCHEMA = cv.PLATFORM_SCHEMA.extend({
    vol.Required(CONF_API_KEY): cv.string,
    vol.Optional(CONF_LANG, default=DEFAULT_LANG): cv.string,
    vol.Optional(CONF_MODEL, default='whisper-1'): cv.string
    vol.Optional('url', default=None): cv.string
})

async def async_get_engine(hass, config, discovery_info=None):
    """Set up OpenAI STT speech component."""
    api_key = config[CONF_API_KEY]
    language = config.get(CONF_LANG, DEFAULT_LANG)
    model = config.get(CONF_MODEL)
    url = config.get('url')
    return OpenAISTTProvider(hass, api_key, language, model, url)

class OpenAISTTProvider(Provider):
    """The OpenAI STT API provider."""

    def __init__(self, hass, api_key, lang, model, url):
        """Initialize OpenAI STT provider."""
        self.hass = hass
        self._api_key = api_key
        self._language = lang
        self._model = model
        self._url = url

    @property
    def default_language(self) -> str:
        """Return the default language."""
        return self._language

    @property
    def supported_languages(self) -> list[str]:
        """Return the list of supported languages."""
        # Ideally, this list should be dynamically fetched from OpenAI, if supported.
        return [self._language]

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
        self, metadata: SpeechMetadata, stream
    ) -> SpeechResult:
        # Collect data
        audio_data = b""
        async for chunk in stream:
            audio_data += chunk

        # OpenAI API endpoint for audio transcription
        url = self._url or OPENAI_STT_URL
        # Your OpenAI API key
        headers = {
            'Authorization': f'Bearer {self._api_key}',
            'Content-Type': 'multipart/form-data'
        }
        # Prepare the data for the POST request
        files = {
            'file': ('filename.wav', audio_data, 'audio/wav')
        }
        data = {
            'model': self._model,
            'language': self._language,
            # 'prompt': 'Your prompt here',  # Optional
            # 'response_format': 'json',  # Optional, defaults to json
            # 'temperature': 0  # Optional
        }
        def job():
            response = requests.post(url, headers=headers, files=files, data=data)
            if response.status_code == 200:
                return response.text
            else:
                _LOGGER.error("%s", response.text)
                return ''

        async with async_timeout.timeout(10):
            assert self.hass
            response = await self.hass.async_add_executor_job(job)
            if len(response) > 0:
                return SpeechResult(
                    response.text,
                    SpeechResultState.SUCCESS,
                )
            return SpeechResult("", SpeechResultState.ERROR)