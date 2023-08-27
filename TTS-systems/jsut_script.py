import os
import time
ckpt_steps = 50000
ckpt_idx = 5
batch_size = 128
seconds = 20
elapsed_time_avg = 0

inputSentence = "フォーニバルの裁判の際、砦を離れられないダリオが自身の代わりに、無罪の証言をする証人として彼を遣わす。"
speaker = "JSUT"

if __name__ == "__main__":
    # download and preprocess
    os.chdir("raw_data")
    os.system("wget -O jsut_ver1.1.zip http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip")
    os.system("unzip jsut_ver1.1.zip")
    os.chdir("../")

    os.system("python -m scripts.jsut_hts2textgrid")
    os.system("python preprocess.py raw_data/jsut_ver1.1 preprocessed_data/JSUT --dataset JSUT --parse_raw --preprocess --force")
    os.system("python clean.py preprocessed_data/JSUT data_config/JSUT/clean.json")
    os.system("python preprocess.py raw_data/jsut_ver1.1 preprocessed_data/JSUT --dataset JSUT --create_dataset data_config/JSUT/clean.json --force")
    
    # train
    os.system("python -m scripts.collect_phonemes")
    os.system("python fastspeech2_train.py -s train -d data_config/JSUT -m config/FastSpeech2/model/base.yaml -t config/FastSpeech2/train/baseline.yaml -a config/FastSpeech2/algorithm/baseline.yaml -n JSUT_fastspeech_sample_code_JSUT")
    
    # inference
    os.system(f"python fastspeech2_inference.py -s jsut -d data_config/{speaker} -m ./output_temp/fastspeech2/npy/steps={ckpt_steps}-{seconds}secs-{speaker}.npy -w ./output_temp/fastspeech2/wav/steps={ckpt_steps}-{seconds}secs-{speaker}.wav -pre ./output/JSUT_fastspeech_sample_code/ckpt/epoch={4*ckpt_idx-1}-step={ckpt_steps}.ckpt -i '{inputSentence}' ")
    
    os.system("echo 'Done!' ")