import os

datasets = [
    "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "https://www.openslr.org/resources/12/test-other.tar.gz",
    "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "https://www.openslr.org/resources/12/train-other-500.tar.gz",
]

emotion_datasets = "https://sail.usc.edu/iemocap/"

if __name__ == "__main__":
    # install requirements

    # os.system("pip install -r requirements/all.txt")
    # os.system("pip install -r requirements/dev.txt")
    # os.system("pip install -r requirements/install.txt")

    # Download the data
    if not os.path.exists("data"):
        os.mkdir("data")
        os.chdir("data")
        for dataset in datasets:
            os.system(f"wget -O {dataset.split('/')[-1]} {dataset}")
            os.system(f"tar -xvf {dataset.split('/')[-1]} && rm -rf {dataset.split('/')[-1]}")
        os.chdir("..")

    # Preprocess the data
    os.system("python preprocess/generate_len_for_bucket.py -f ")
    
    os.system("echo 'Done!' ")