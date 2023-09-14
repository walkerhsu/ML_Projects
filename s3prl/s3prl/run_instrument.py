import os

if __name__ == "__main__":
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

    os.system("echo 'Done!' ")
