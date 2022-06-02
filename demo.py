from email.mime import audio
import os
from google.cloud import texttospeech
from google.cloud import texttospeech_v1

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'plucky-sector-346202-445f467170a3.json'

client = texttospeech_v1.TextToSpeechClient()

text = "bonjour"
# this is the text that is being converted to speech


synthesis_input = texttospeech_v1.SynthesisInput(text=text)

print(client.list_voices())
# this will print all language codes (languages/accents)

voice1 = texttospeech_v1.VoiceSelectionParams(

    language_code = 'en-in',
    ssml_gender = texttospeech_v1.SsmlVoiceGender.MALE
)

voice2 = texttospeech_v1.VoiceSelectionParams(
    name = 'fr-FR-Wavenet-E', 
    language_code = 'fr-FR'
)

# I believe I can change the emotion here
# or possibly in voice
audio_config = texttospeech_v1.AudioConfig(
    audio_encoding = texttospeech_v1.AudioEncoding.MP3
)

response1 = client.synthesize_speech(
    input = synthesis_input, 
    voice = voice1,
    audio_config = audio_config
)

response2 = client.synthesize_speech(
    input = synthesis_input, 
    voice = voice2,
    audio_config = audio_config
)


with open('audio file1.mp3', 'wb') as output1:
    output1.write(response1.audio_content)


with open('audio file2.mp3', 'wb') as output2:
    output2.write(response2.audio_content)




