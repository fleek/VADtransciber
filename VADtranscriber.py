import ffmpeg
import torch
from pysrt import SubRipFile, SubRipTime, SubRipItem
import whisper
from colorama import Fore
import os
import json
import subprocess
import PySimpleGUI as sg
from tqdm import tqdm
from pyannote.audio import Pipeline

SAMPLING_RATE = 16000


def convertWav(src):
    print("Converting MP4 to Wav")
    tempwav = "work/" + os.path.splitext(os.path.basename(src))[0] + "_temp.wav"
    print(tempwav)
    ffmpeg.input(src).output(
             tempwav,
             ar="16000",
             ac="1",
             acodec="pcm_s16le",
             map_metadata="-1",
             fflags="+bitexact",
    ).overwrite_output().run(quiet=True)



def performVAD(src):

    print(Fore.GREEN + f"Loading Silero VAD model")
    smodel, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                   model='silero_vad',
                                   force_reload=True,
                                   onnx=False)

    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    print("Read Wav file")
    tempwav = "work/"+os.path.splitext(os.path.basename(src))[0] + "_temp.wav"
    wav = read_audio(tempwav, sampling_rate=SAMPLING_RATE)

# get speech timestamps from full audio file

    print("getting speech timestamps")
    st = get_speech_timestamps(wav, smodel,
                                              threshold=0.65,                  #0.5,
                                              sampling_rate=16000,
                                              min_speech_duration_ms=5,      #250,
                                              min_silence_duration_ms=100,      #100,
                                              window_size_samples= 1536,      #this is fixed
                                              speech_pad_ms=10,                #30,
                                              return_seconds= False,
                                              visualize_probs=False
                                              )
    print("generate chunk list")
    chunklist = []
    for i,s in enumerate(st):
        fname = "vad/" + os.path.splitext(os.path.basename(src))[0] + f"_{i:05d}.wav"
        chunklist.append({'start':s['start']-int(120*SAMPLING_RATE/1000), 'end':s['end']+int(50*SAMPLING_RATE/1000), 'idx':i, 'text':"", 'fname': fname, 'speaker':""})

    tempchunk = "work/" + os.path.splitext(os.path.basename(src))[0] + "_chunk.json"
    with open(tempchunk, 'w', encoding='utf-8') as fp:
        json.dump(chunklist, fp)

    return wav, chunklist


def formSpeechSlices(st,wav):
    print(Fore.GREEN + f"Loading Silero VAD model")
    smodel, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                   model='silero_vad',
                                   force_reload=True,
                                   onnx=True)

    (get_speech_timestamps,
     save_audio,
     read_audio,
     VADIterator,
     collect_chunks) = utils

    print("Forming speech slices")
    for c in st:
        save_audio(
            c['fname'],
            collect_chunks([c], wav),
            sampling_rate=SAMPLING_RATE,
        )


def doTranscribe(src,ms,dv):
    tempfinal = "work/" + os.path.splitext(os.path.basename(src))[0] + "_final.json"
    if not os.path.exists(tempfinal):
        print(Fore.RED + f"File wasn't previously processed, skipping {src}")
        return None

    with open(tempfinal, encoding='utf-8') as json_file:
        st = json.load(json_file)
    print(Fore.GREEN + f"Loading Whisper {ms} model")
    wmodel = whisper.load_model(ms, download_root="models", device=dv)
    dest = os.path.splitext(src)[0] + "_"+ ms +".srt"
    fp16 = True if dv == "cuda" else False
    print(Fore.GREEN+ "Doing transcription and create srt")
    subs = SubRipFile()
    options = {"task":"transcribe","language":"English","fp16":fp16,'no_speech_threshold':0.1, "condition_on_previous_text": False, "logprob_threshold": -1.00, "without_timestamps":True}
    pbar = tqdm(st)
    for s in pbar:
        pbar.set_description_str(s['fname'])
        result = whisper.transcribe(wmodel, audio=s['fname'], verbose=False, **options)
        pbar.set_postfix_str(result['text'])
        s['text'] = result['text']
        i = SubRipItem()
        i.start = SubRipTime(milliseconds=s['start']/(SAMPLING_RATE/1000))
        i.end = SubRipTime(milliseconds=s['end']/(SAMPLING_RATE/1000))
        i.index = s['idx']
        i.text = f"{s['speaker'].strip()} {s['text'].strip()}"
        subs.append(i)

    subs.save(dest, encoding='utf-8')
    strfinal = "work/" + os.path.splitext(os.path.basename(src))[0] + "_" + ms + "_srt.json"
    with open(strfinal,"w", encoding='utf-8') as srt_file:
        json.dump(st,srt_file)

    print(Fore.GREEN+"-----------------Transcription Complete-------------------")
    return dest

def PopulateSpeakers(src):
    print("Populate Speakers")

    tempchunk = "work/" + os.path.splitext(os.path.basename(src))[0] + "_chunk.json"
    with open(tempchunk, encoding='utf-8') as fp:
        st = json.load(fp)

    tempdiarise = "work/" + os.path.splitext(os.path.basename(src))[0] + "_diarise.json"
    with open(tempdiarise,encoding='utf-8') as json_file:
        splist = json.load(json_file)

    for s in st:
        for sp in splist:
            speaker = "<Speaker?>"
            if s['start']/SAMPLING_RATE >= sp['start']-0.2 and s['end']/SAMPLING_RATE <= sp['end']+0.3:
                print(Fore.GREEN + f"{s['idx']=}:  {s['start']/SAMPLING_RATE=}  {sp['start']-0.2=}   | {s['end']/SAMPLING_RATE=}  {sp['end']+0.2=}")
                speaker = f"<Speaker{int(sp['speaker'].replace('SPEAKER_',''))+1}> "
                break
        s['speaker'] = speaker

    tempfinal = "work/" + os.path.splitext(os.path.basename(src))[0] + "_final.json"
    with open(tempfinal, 'w',encoding='utf-8') as fp:
        json.dump(st, fp)


def TranscriptionPipe(src, modelsize, device):
    #clearworkfolders()
    create_folder('./work')
    create_folder('./vad')
    convertWav(src)
    wav, st = performVAD(src)
    localdiarise("work/" + os.path.splitext(os.path.basename(src))[0] + "_temp.wav","work/" + os.path.splitext(os.path.basename(src))[0] + "_diarise.json")
    PopulateSpeakers(src)
    formSpeechSlices(st, wav)
    return doTranscribe(src,modelsize,device)

def diarise(src):
    tempwav = os.getcwd()+"/work/" + os.path.splitext(os.path.basename(src))[0] + "_temp.wav"
    tempdiarise = os.getcwd()+"/work/" + os.path.splitext(os.path.basename(src))[0] + "_diarise.json"
    cmd = f'diarise.bat "{tempwav}" "{tempdiarise}" '
    ret = subprocess.call(cmd, shell=True)


def localdiarise(src,dest):
    dlist = []
    print(Fore.GREEN+"Loading Diarization Model")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1",
                                        use_auth_token=<PyAnnote token>)

    print(Fore.GREEN+"Loading Audio")
    diarization = pipeline(src)

    print(Fore.GREEN+"Diarise")
    idx = 0
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        print(f"idx={idx} start={turn.start:.1f}s stop={turn.end:.1f}s {speaker}")
        dlist.append({'idx':idx, 'start':turn.start,'end':turn.end, 'speaker': speaker})
        idx+=1

    with open(dest,"w",encoding='utf-8') as fp:
        json.dump(dlist, fp)

    Subfile = SubRipFile()
    for idx, d in enumerate(dlist):
        i = SubRipItem()
        i.index = idx
        i.start = d['start']
        i.start = SubRipTime(milliseconds=int(d['start']*1000))
        i.end = SubRipTime(milliseconds=int(d['end']*1000))
        i.text = d['speaker']
        Subfile.append(i)
    Subfile.save("diarise.srt")

def clearworkfolders():
    for f in os.listdir("vad"):
        os.remove(f"vad/{f}")


def create_folder(folder_path):
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)


def Interactive():
    layout = [
        [sg.Text("Select Video File"),sg.In(size=(25, 1), enable_events=True, key="-File-"), sg.FileBrowse()],
        [sg.Text("_"*100)],
        [sg.Text("Select Model")],
        [sg.Radio('tiny.en','Model',key="tiny.en"),sg.Radio('base.en','Model',key="base.en"),sg.Radio('small.en','Model',key="small.en"),sg.Radio('medium.en','Model',key="medium.en",default=True), sg.Radio('large-v1','Model',key="large-v1"),sg.Radio('large-v2','Model',key="large-v2")],
        [sg.Text("Select Method")],
        [sg.Radio('cpu', 'Method', key="cpu"), sg.Radio('cuda', 'Method', key="cuda", default=True)],
        [sg.Text("_" * 100)],
        [sg.Button('Full Processing'),sg.Button('Transcribe Again'), sg.Button('Exit')]
    ]
    window = sg.Window(title="VAD+Transcribe", layout=layout)
    while True:
        event, values = window.read(timeout=0.5)
        if event == "OK" or event == sg.WIN_CLOSED:
            break
        if event == "-File-":
            pass
        if event == "Full Processing":
            ms = "large-v2"
            dv = "cpu"
            if values["-File-"] == "":
                continue
            if values["cuda"]:
                dv = 'cuda'
            if values["tiny.en"]:
                ms = "tiny.en"
            if values["base.en"]:
                ms = "base.en"
            if values["small.en"]:
                ms = "small.en"
            if values["medium.en"]:
                ms = "medium.en"
            if values["large-v1"]:
                ms = "large-v1"
            if values["large-v2"]:
                ms = "large-v2"
            TranscriptionPipe(values['-File-'],ms,dv)
        if event == "Transcribe Again":
            ms = "large-v2"
            dv = "cpu"
            if values["-File-"] == "":
                continue
            if values["cuda"]:
                dv = 'cuda'
            if values["tiny.en"]:
                ms = "tiny.en"
            if values["base.en"]:
                ms = "base.en"
            if values["small.en"]:
                ms = "small.en"
            if values["medium.en"]:
                ms = "medium.en"
            if values["large-v1"]:
                ms = "large-v1"
            if values["large-v2"]:
                ms = "large-v2"
            if not doTranscribe(values['-File-'], ms, dv):
                TranscriptionPipe(values['-File-'], ms, dv)
        if event in (None, 'Exit'):  # checks if user wants to
            exit
            break

def BatchProcess(srcfolder,ms,dv):
    for subdir, dirs, files in os.walk(srcfolder):
        for f in files:
            if os.path.splitext(f)[-1].upper() in (".WAV",".MP4"):
                print(f"Processing {os.path.join(subdir, f)}")
                TranscriptionPipe(os.path.join(subdir, f),ms, dv)

if __name__ == '__main__':
    #BatchProcess(r"Z:\testfiles","medium.en","cuda")
    Interactive()
