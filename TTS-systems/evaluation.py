import argparse
import os
import glob
import re
import jiwer
import speech_recognition as sr
from tqdm import tqdm

import Define


TAG_MAPPING = {
    "google": {  # https://stackoverflow.com/questions/14257598/what-are-language-codes-in-chromes-implementation-of-the-html5-speech-recogniti/14302134#14302134
        "en": "en",
        "zh": "zh",
        "ko": "ko",
        "jp": "ja",
        "fr": "fr",
        "de": "de",
        "es": "es",
        "ru": "ru",
    },
    "whisper": {  # https://github.com/openai/whisper/blob/main/whisper/tokenizer.py
        "en": "en-US",
        "zh": "zh-CN",
        "ko": "ko",
        "jp": "ja",
        "fr": "fr-FR",
        "de": "de-DE",
        "es": "es-ES",
        "ru": "ru",
    },
}

r = sr.Recognizer()
def whisper(wav_path, lang: str):
    with sr.AudioFile(wav_path) as source:
        audio = r.record(source)  # read the entire audio file

    # recognize speech using Whisper
    try:
        res = r.recognize_whisper(audio, model='large', language=TAG_MAPPING["whisper"][lang])
        # res = r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS, language="ko-KR")
        return res
    except sr.UnknownValueError:
        print("Whisper could not understand audio")
    except sr.RequestError as e:
        print("Whisper error; {0}".format(e))
    return ""


def google(wav_path, lang):
    with sr.AudioFile(wav_path) as source:
        audio = r.record(source)  # read the entire audio file

    # recognize speech using Google API
    try:
        res = r.recognize_google(audio, key=None, language=TAG_MAPPING["google"][lang])
        return res
    except sr.UnknownValueError:
        print("Google could not understand audio")
    except sr.RequestError as e:
        print("Google error; {0}".format(e))
    return ""


def cer(raw_text, pred_text, remove_whitespace=False):
    raw_text = re.sub(r'[^\w\s]', '', raw_text)
    pred_text = re.sub(r'[^\w\s]', '', pred_text)
    if remove_whitespace:
        raw_text = raw_text.replace(' ', '')
        pred_text = pred_text.replace(' ', '')
    raw_text = raw_text.upper()
    pred_text = pred_text.upper()
    cer = jiwer.cer(raw_text, pred_text)

    return cer

def calculate_wer(reference, hypothesis):
    print(reference)
    print(hypothesis)
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    # Counting the number of substitutions, deletions, and insertions
    substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp)
    deletions = len(ref_words) - len(hyp_words)
    insertions = len(hyp_words) - len(ref_words)
    # Total number of words in the reference text
    total_words = len(ref_words)
    # Calculating the Word Error Rate (WER)
    wer = (substitutions + deletions + insertions) / total_words
    return wer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--wav_dir", type=str, help="wav path", default="./output_temp/fastspeech2/wav/test.wav")
    parser.add_argument("-l", "--lang", type=str, help="language", default="en")
    parser.add_argument("-t", "--type", type=str, help="type (Google or Whisper)", default="Google")
    args = parser.parse_args()

    translate = ""
    # inputSentence = "Amidst the sprawling tapestry of an ancient city, where history and modernity intertwine in a captivating embrace, labyrinthine alleyways wind their way through vibrant markets alive with the colors and aromas of exotic spices, while the echoes of distant footsteps resonate against the ornate facades of centuries-old architecture that stand as testament to the passage of time and the stories of countless generations."
    # inputSentence = "Underneath the starlit sky whispers of forgotten tales dance on the gentle breeze weaving dreams of distant worlds"
    inputSentence = "フォーニバルの裁判の際砦を離れられないダリオが自身の代わりに無罪の証言をする証人として彼を遣わす"
    if(args.type == "Google"):
        translate = google(args.wav_dir, args.lang)
    elif(args.type == "Whisper"):
        translate = whisper(args.wav_dir, args.lang)
    else:
        raise NotImplementedError
    
    # print(translate)
    wer = calculate_wer(inputSentence.lower() , translate.lower())
    print(f"wer : {wer}")