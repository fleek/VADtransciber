# VADtransciber

This app was created specifically for my use case and not optimized.

It combines processing into several AI models into one app to automate the transcription of .wav and .mp4 files.

1.  Convert audio into the best format the model was trained on.
2.  Perform VAD using silero-vad (https://github.com/snakers4/silero-vad) I find as the most accurate so far and dump the slice info into json.
3.  Use pyannote.audio to diariatize the audio and annotate the json slice info from the previous step.
4.  Slice the audio into real audio chunk files.
5.  feed each audio chunk file into whisper (I find that whisper gives most accurate performance without history.
6.  combine audo chunk transcripts into a full transcript file.

A .json file will be created for each step, this is for my own troubleshooting and possible to restart from any step

This program has been created to be modular, not with efficiency in mind.

For installation please install the necessary dependencies in requirements.txt and also refer to https://pytorch.org/get-started/previous-versions/ for installation of the necessary cuda support.

Please also obtain your own pyannote token

I have used python 3.8 as some of the libraries does not support 3.10 yet.

