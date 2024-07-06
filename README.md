# openai_stt

5. Configure the component in your `configuration.yaml` file:

```yaml
stt:
  - platform: openai_stt
    api_key: "your_api_key_here"
    # Optional parameters:
    language: "en-US"
    model: "whisper"
    prompt: "You are transcribing a command to a virtual assistant."
