import os
import time
ckpt_steps = 50000
ckpt_idx = 5
batch_size = 128
seconds = 5
elapsed_time_avg = 0
(p_control, e_control, d_control) = (1, 2.0, 1)

ckpt_steps_max = 150000
# ckpt_path = f"./output/LJSpeech-1.1_fastspeech_sample_code/ckpt/batch_size={batch_size}-epoch={(4*ckpt_idx)-1}-step={ckpt_steps}.ckpt"
inputSentence = "Underneath the starlit sky, whispers of forgotten tales dance on the gentle breeze, weaving dreams of distant worlds."
# inputSentence = "Deep learning is fun."
# inputSentence = "Amidst the sprawling tapestry of an ancient city, where history and modernity intertwine in a captivating embrace, labyrinthine alleyways wind their way through vibrant markets alive with the colors and aromas of exotic spices, while the echoes of distant footsteps resonate against the ornate facades of centuries-old architecture that stand as testament to the passage of time and the stories of countless generations."
# inputSentence = "和煦的陽光映照著花海和綠意盎然的原野。"


if __name__ == "__main__":
        # train
    # if(os.path.exists(ckpt_path)):
    #     os.system(f"python fastspeech2_train.py -s train -d data_config/LJSpeech-1.1 -m config/FastSpeech2/model/base.yaml -t config/FastSpeech2/train/baseline.yaml -a config/FastSpeech2/algorithm/baseline.yaml -n LJSpeech-1.1_fastspeech_sample_code -pre {ckpt_path}")
    # else:
    #     ckpt_steps=0
    #     ckpt_idx=0
    #     os.system(f"python fastspeech2_train.py -s train -d data_config/LJSpeech-1.1 -m config/FastSpeech2/model/base.yaml -t config/FastSpeech2/train/baseline.yaml -a config/FastSpeech2/algorithm/baseline.yaml -n LJSpeech-1.1_fastspeech_sample_code")

    # modify ckpt_steps, ckpt_idx, and ckpt_path for next iteration
    # ckpt_steps+=50000
    # ckpt_idx+=5
    # ckpt_path = f"./output/LJSpeech-1.1_fastspeech_sample_code/ckpt/batch_size={batch_size}-epoch={(4*ckpt_idx)-1}-step={ckpt_steps}.ckpt"
    
    # # inference
    # os.system(f"python fastspeech2_inference.py -d data_config/LJSpeech-1.1 -m output_temp/fastspeech2/npy/steps={ckpt_steps}-batch_size={batch_size}.npy -w output_temp/fastspeech2/wav/steps={ckpt_steps}-batch_size={batch_size}.wav -pre {ckpt_path} -i '{inputSentence}' ")

    # start_time = time.time()
    # os.system(f"python fastspeech2_inference.py -m ./output_temp/fastspeech2/npy/steps={ckpt_steps}-{seconds}secs-ped={p_control},{e_control},{d_control}.npy -w ./output_temp/fastspeech2/wav/steps={ckpt_steps}-{seconds}secs-ped={p_control},{e_control},{d_control}.wav -pre ./output/LJSpeech-1.1_fastspeech_sample_code/ckpt/batch_size={batch_size}-epoch={4*ckpt_idx-1}-step={ckpt_steps}.ckpt -i '{inputSentence}' ")
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # os.system(f"echo 'Elapsed time: {elapsed_time}' ")
    # elapsed_time_avg += elapsed_time
    # ckpt_idx += 1
    # ckpt_steps += 10000
    # os.system("pip install tensorflow")
    os.chdir("output")
    for dir in os.listdir():
        exp_name = dir    
        os.system(f"echo {exp_name}")
        os.system(f"tensorboard --logdir=./output/{exp_name}/log/tb/ --bind_all --load_fast=false")
    # os.system(f"echo ' Averaged processing time for {seconds} seconds sentence using Fastspeech2 (batch_size={batch_size}): {elapsed_time_avg / 5}' ")
    os.system("echo 'Done!' ")