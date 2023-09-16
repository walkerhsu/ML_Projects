import os
import random
import pandas as pd
import argparse
from pathlib import Path
import json
from sklearn.model_selection import KFold

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

    parser.add_argument(
        "-m",
        "--metadata",
        type=str,
        default="downstream/instrument_identification/meta_data",
        help="metadata root",
    )

    parser.add_argument("-f", "--folds", type=int, default=5, help="number of folds")
    args = parser.parse_args()
    return args.root, args.metadata, args.folds


def saveData(root):
    meta_data = []
    for dir in os.listdir(root):
        if f"IRMAS-TestingData" in dir:
            meta_data += saveTestingData(root, dir)
        elif "IRMAS-TrainingData" in dir:
            meta_data += saveTrainingData(root, dir)
        # elif "Training" in dir:
        #     saveTrainingData(root, dir)
        # elif "Testing" in dir:
        #     saveTestingData(root, dir)
        else:
            continue

    return meta_data


def saveTrainingData(root, dir):
    meta_data = []
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
            meta_data.append(musicData)
    return meta_data
    # print(trainingAudioInstrument)
    # with open(f"{dir}.json", "w") as f:
    #     json.dump(trainingAudioInstrument, f)


def saveTestingData(root, dir):
    meta_data = []
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
                meta_data.append(musicData)

    return meta_data


def main():
    root, metadata_root, folds = get_arguments()
    meta_data = saveData(root)
    random.shuffle(meta_data)
    kf = KFold(n_splits=folds)
    for fold_idx, (train_index, test_index) in enumerate(kf.split(meta_data)):
        trainingData = {
            "instruments": ins2idx(),
            "meta_data": [meta_data[idx] for idx in train_index],
        }
        testingData = {
            "instruments": ins2idx(),
            "meta_data": [meta_data[idx] for idx in test_index],
        }
        session = "Session{}".format(fold_idx + 1)
        os.mkdir(f"{metadata_root}/{session}")
        TrainingPath = os.path.join(metadata_root, session, "Training.json")
        TestingPath = os.path.join(metadata_root, session, "Testing.json")
        with open(TrainingPath, "w") as f:
            json.dump(trainingData, f)
        with open(TestingPath, "w") as f:
            json.dump(testingData, f)
    # saveData(root, metadata)


if __name__ == "__main__":
    main()
