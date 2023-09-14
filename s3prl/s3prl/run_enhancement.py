import os

if __name__ == "__main__":
    # install requirements

    # os.system("pip install -r requirements/all.txt")
    # os.system("pip install -r requirements/dev.txt")
    # os.system("pip install -r requirements/install.txt")

    # Download the data

    # os.system(
    #     "python ./downstream/enhancement_stft2/scripts/Voicebank/data_prepare.py voicebank_data \
    #         ./downstream/enhancement_stft2/datasets/voicebank --part train"
    # )
    # os.system(
    #     "python ./downstream/enhancement_stft2/scripts/Voicebank/data_prepare.py voicebank_data \
    #         ./downstream/enhancement_stft2/datasets/voicebank --part dev"
    # )
    # os.system(
    #     "python ./downstream/enhancement_stft/scripts/Voicebank/data_prepare.py voicebank_data \
    #         ./downstream/enhancement_stft2/datasets/voicebank --part test"
    # )

    # train

    os.system(
        "python3 run_downstream.py -m train \
       -c downstream/enhancement_stft2/configs/cfg_voicebank.yaml \
       -d enhancement_stft2 \
       -u hubert \
       -n hubert--voicebank2"
    )

    os.system(
        "python3 run_downstream.py -m evaluate \
       -e result/downstream/hubert--voicebank2/best-states-dev.ckpt"
    )

    os.system("echo 'Done!' ")
