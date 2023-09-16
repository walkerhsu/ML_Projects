import os

not_download, not_preprocess = False, False
train = True
if __name__ == "__main__":
    # Download IRMAS dataset
    if not_download:
        os.chdir("data")
        os.system(
            "wget https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part1.zip?download=1 -O IRMAS-TestingData-Part1.zip"
        )
        os.system(
            "wget https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part2.zip?download=1 -O IRMAS-TestingData-Part2.zip"
        )
        os.system(
            "wget https://zenodo.org/record/1290750/files/IRMAS-TestingData-Part3.zip?download=1 -O IRMAS-TestingData-Part3.zip"
        )
        os.system(
            "wget https://zenodo.org/record/1290750/files/IRMAS-TrainingData.zip?download=1 -O IRMAS-TrainingData.zip"
        )

        os.system("unzip IRMAS-TestingData-Part1.zip && rm IRMAS-TestingData-Part1.zip")
        os.system("unzip IRMAS-TestingData-Part2.zip && rm IRMAS-TestingData-Part2.zip")
        os.system("unzip IRMAS-TestingData-Part3.zip && rm IRMAS-TestingData-Part3.zip")
        os.system("unzip IRMAS-TrainingData.zip && rm IRMAS-TrainingData.zip")

    # process IRMAS dataset
    if not_preprocess:
        os.system("python downstream/instrument_identification/dataset/prepare_data.py")

    # train
    if train:
        # os.system(
        #     "python run_downstream.py -m train -n hubert--instrument_identification--BCE \
        #         -u hubert -d instrument_identification \
        #         -e result/downstream/hubert--instrument_identification--BCE/dev-best.ckpt \
        #         -cd"
        # )
        os.system(
            "python run_downstream.py -m train -n hubert--instrument_identification \
                -u hubert -d instrument_identification \
                -cd"
        )

    os.system("echo 'Done!' ")
