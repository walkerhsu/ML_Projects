import os
import time

ckpt_path = f"./output/LJSpeech-1.1_fastspeech_sample_code/ckpt/epoch=19-step=50000.ckpt"
inputSentence = "Underneath the starlit sky, whispers of forgotten tales dance on the gentle breeze, weaving dreams of distant worlds."
# inputSentence = "Deep learning is fun."
# inputSentence = "Amidst the sprawling tapestry of an ancient city, where history and modernity intertwine in a captivating embrace, labyrinthine alleyways wind their way through vibrant markets alive with the colors and aromas of exotic spices, while the echoes of distant footsteps resonate against the ornate facades of centuries-old architecture that stand as testament to the passage of time and the stories of countless generations."
inputSentence = "和煦的陽光映照著花海和綠意盎然的原野。"
steps = 10000
epoch = 3
type="phoneme"
seconds = 5
elapsed_time_avg = 0

if __name__ == "__main__" :

    # os.system("python tacotron2_train.py -s train -d data_config/LJSpeech-1.1 -m config/Tacotron2/model/base.yaml -t config/Tacotron2/train/baseline.yaml -a config/Tacotron2/algorithm/baseline.yaml -n LJSpeech-1.1_tacotron_sample_code_phoneme")
    # os.system(f"python tacotron2_inference.py -d data_config/LJSpeech-1.1 -m output_temp/tacotron2/npy/steps=50000_phoneme.npy -w output_temp/tacotron2/wav/steps=50000_phoneme.wav -pre ./output/LJSpeech-1.1_tacotron_sample_code_phoneme/ckpt/epoch=19-step=50000.ckpt -i '{inputSentence}' ")
    for idx in range(5):
        start_time = time.time()
        os.system(f"python tacotron2_inference.py -p ./output_temp/tacotron2/img/steps={steps}_input={type}_{seconds}secs.png -m ./output_temp/tacotron2/npy/steps={steps}_input={type}_{seconds}secs.npy -w ./output_temp/tacotron2/wav/steps={steps}_input={type}_{seconds}secs.wav -pre ./output/LJSpeech-1.1_tacotron_sample_code_{type}/ckpt/epoch={epoch}-step={steps}.ckpt -i '{inputSentence}' -t {type}")
        end_time = time.time()
        elapsed_time = end_time - start_time
        os.system(f'echo "Elapsed time: {elapsed_time}"') 
        elapsed_time_avg += elapsed_time

        epoch += 4
        steps += 10000

    os.system(f"echo ' Averaged processing time for {seconds} seconds sentence using Tacotron2 ({type}): {elapsed_time_avg / 5}' ")
    os.system("echo 'Done!' ")