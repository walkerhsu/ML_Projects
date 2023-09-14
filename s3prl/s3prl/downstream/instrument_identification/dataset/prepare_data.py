import os
import pandas as pd
import argparse
from pathlib import Path
import json

INSTRUMENTS = [
    "cel",
    "cla",
    "flu",
    "gac",
    "gel",
    "org",
    "pia",
    "sax",
    "tru",
    "vio",
    "voi",
]


def ins2idx():
    return {instrument: idx for idx, instrument in enumerate(INSTRUMENTS)}


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--root",
        type=str,
        default="data/IRMAS",
        help="IRMAS dataset root",
    )
    args = parser.parse_args()
    return args.root


def saveData(root):
    for dir in os.listdir(root):
        if "IRMAS-TestingData" in dir or "IRMAS-EvaluationData" in dir:
            saveTestingData(root, dir)
        elif "IRMAS-TrainingData" in dir:
            saveTrainingData(root, dir)
        # elif "Training" in dir:
        #     saveTrainingData(root, dir)
        # elif "Testing" in dir:
        #     saveTestingData(root, dir)
        else:
            continue


def saveTrainingData(root, dir):
    trainingAudioInstrument = {
        "instruments": ins2idx(),
        "meta_data": [],
    }
    dir = os.path.join(root, dir)
    if os.path.isdir(dir):
        for musicFile in Path(dir).rglob("**/*.wav"):
            assert os.path.isfile(musicFile)
            musicData = {
                "path": str(musicFile),
            }
            audioIns = (str(musicFile)).split("/")[-1].split("__")[-2].split("][")
            while audioIns[-1][-1] != "]":
                audioIns[-1] = audioIns[-1][:-1]
            audioIns[-1] = audioIns[-1][:-1]
            while audioIns[0][0] != "[":
                audioIns[0] = audioIns[0][1:]
            audioIns[0] = audioIns[0][1:]
            audioIns = [ins for ins in audioIns if ins in INSTRUMENTS]
            # if len(audioIns) > most_ins:
            #     most_ins = len(audioIns)
            musicData["label"] = audioIns
            trainingAudioInstrument["meta_data"].append(musicData)

        print(trainingAudioInstrument)
        with open(f"{dir}.json", "w") as f:
            json.dump(trainingAudioInstrument, f)


def saveTestingData(root, dir):
    testingAudioInstrument = {
        "instruments": ins2idx(),
        "meta_data": [],
    }
    file_name = dir.split("-")[-1]
    dir = os.path.join(root, dir)
    if os.path.isdir(dir):
        dataDir = os.path.join(dir, dir.split("-")[-1])
        # dataDir = dir
        wavFiles = Path(dataDir).glob("*.wav")
        txtFiles = Path(dataDir).glob("*.txt")
        sortedWavFiles = []
        sortedTxtFiles = []
        for wavFile in wavFiles:
            sortedWavFiles.append(str(wavFile))
        for txtFile in txtFiles:
            sortedTxtFiles.append(str(txtFile))
        sortedWavFiles.sort()
        sortedTxtFiles.sort()
        for sortedWavFile, sortedTxtFile in zip(sortedWavFiles, sortedTxtFiles):
            assert sortedWavFile.split(".").remove("wav") == sortedTxtFile.split(
                "."
            ).remove("txt")
            assert os.path.isfile(sortedWavFile) and os.path.isfile(sortedTxtFile)
            musicData = {
                "path": sortedWavFile,
            }
            with open(sortedTxtFile, "r") as f:
                audioIns = [line.strip("\r\n\t ") for line in f]
                audioIns = [ins for ins in audioIns if ins in INSTRUMENTS]
                musicData["label"] = audioIns
                testingAudioInstrument["meta_data"].append(musicData)
        with open(f"{dir}.json", "w") as f:
            json.dump(testingAudioInstrument, f)


def main():
    root = get_arguments()
    saveData(root)


if __name__ == "__main__":
    main()
