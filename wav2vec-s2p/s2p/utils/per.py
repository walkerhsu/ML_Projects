import edit_distance
import os
from tqdm import tqdm

ref_f = "/home/yukuanfu88/iven/wav2vecu-s2p/s2p/utils/asr_test.phones.txt"
hyp_f = "/home/yukuanfu88/iven/wav2vecu-s2p/s2p/multirun/2023-03-07/00-26-29/0/asr_test/asr_test.txt"

def read_phn_file(file_path):
    data = []
    with open(file_path, 'r') as fr:
        for line in fr:
            line = line.strip().split()
            data.append(line)
    return data
        
def filter(line):
    return True
    line = line.strip().split('\t')
    years = ['2016']
    return line[0][0:4] in years

def main():
    ref_data = read_phn_file(ref_f)
    hyp_data = read_phn_file(hyp_f)

    S, D, I, N = (0, 0, 0, 0)
    count = 0
    for ref, hyp in tqdm(zip(ref_data, hyp_data)):
        sm = edit_distance.SequenceMatcher(a=ref, b=hyp)
        opcodes = sm.get_opcodes()
        
        # Substitution
        s = sum([(max(x[2] - x[1], x[4] - x[3]) if x[0] == 'replace' else 0) for x in opcodes])
        # Deletion
        d = sum([(max(x[2] - x[1], x[4] - x[3]) if x[0] == 'delete' else 0) for x in opcodes])
        # Insertion
        i = sum([(max(x[2] - x[1], x[4] - x[3]) if x[0] == 'insert' else 0) for x in opcodes])
        n = len(ref)
        
        S += s
        D += d
        I += i
        N += n
        count += 1
    print(f"PER: {(S+D+I)/N *100} %")
    print(f"DEL RATE: {(D)/N *100} %")
    print(f"INS RATE: {(I)/N *100} %")
    print(f"SUB RATE: {(S)/N *100} %")
    print(count)

if __name__ == "__main__":
    main()