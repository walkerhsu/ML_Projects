import os

if __name__ == "__main__":
    # train
    os.system(
        "python3 run_downstream.py -m train -n wav2vec2--phone_linear -u wav2vec2 -d phone_linear"
    )

    # # evaluate
    # os.system(
    #     "python3 run_downstream.py -m evaluate -e result/downstream/baseline--phone_linear/best-states-dev.ckpt"
    # )

    # os.system("python3 utility/get_best_dev.py result/baseline--phone_linear/log.log")

    os.system("echo 'Done!' ")
