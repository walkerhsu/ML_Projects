import os
ckpt_steps = 50000
ckpt_steps_max = 150000
path_to_fairseq = "/work/u5671908/ML_projects/wav2vec-s2p/s2p/fairseq"
path_to_kenlm = "/work/u5671908/ML_projects/wav2vec-s2p/s2p/kenlm"
# ckpt_path = 
# cd ML_projects/wav2vec-s2p/s2p && python w2vu_script.py

if __name__ == "__main__":
    # Initialize conda dependencies
    # os.system("sudo apt update")
    # os.system("sudo apt-get install -y libfftw3-dev libsndfile1-dev libgoogle-glog-dev libopenmpi-dev libboost-all-dev")
    
    # create environment
    # os.system("cd ../../")
    # os.system("conda env create --file wav2vec-s2p/environment.yml -n wav2vec-s2p")
    # os.system("echo 'Environment create!' ")
    
    # activate environment
    os.system("conda activate wav2vec-s2p")
    os.system("echo 'Env activated!' ")
    
    # Install the specific version of fairseq
    os.chdir("../../wav2vec-s2p")
    os.system("pwd")
    # os.system("rm -r fairseq && unzip fairseq.zip && mv fairseq-* fairseq")
    os.chdir("fairseq")
    os.system("pwd")
    # os.system("pip install --upgrade pip")
    os.system("python -m pip install . ")
    os.system("pip install --editable ./")
    os.system("python setup.py build_ext --inplace")
    os.system("echo 'fairseq done!!!' ")
    
    # Install KenLM
    os.chdir("../s2p")
    # os.system("git clone https://github.com/kpu/kenlm.git")
    os.chdir("kenlm/build")
    os.system("cmake .. && make -j 4")
    
    # Install some dependancies
    os.system("pip install kenlm && pip install editdistance")
    
    # Export some paths
    os.system(f"export FAIRSEQ_ROOT={path_to_fairseq}")
    os.system(f"export KENLM_ROOT={path_to_kenlm}")
    # os.system(". ~/.bashrc")
    
    # Install flashlight python bindings
    os.system("pwd")
    # os.chdir("s2p")
    # os.system("git clone --branch v0.3.2 https://github.com/flashlight/flashlight")
    os.chdir("flashlight/bindings/python")
    os.system("pip install -e .")

    os.system("echo 'finished installation!' ")
    
    
    
    # os.chdir("output")
    # for dir in os.listdir():
    #     exp_name = dir    
    #     os.system(f"echo {exp_name}")
    #     os.system(f"tensorboard --logdir=./output/{exp_name}/log/tb/ --bind_all --load_fast=false")
    os.system("echo 'Done!' ")