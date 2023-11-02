from fairseq.models.transformer_lm import TransformerLanguageModel
import numpy as np
from tqdm import tqdm
import os


output_dir = 'output'
model_dir = '7.5B'
test_lang = 'en'
output_name = "zero_shot_{}.txt".format(test_lang)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_file = open(os.path.join(output_dir, output_name), 'w')

lm = TransformerLanguageModel.from_pretrained(model_dir, bpe='sentencepiece')
lm = lm.eval()
lm = lm.half()
lm = lm.cuda()

def get_logprobs(prompt, verbose=False):
    import re
    prompt = re.sub('\n+' , '\n', prompt)  # collapse repeated newlines, which indicate separate documents
    if verbose:
        return prompt, lm.score(prompt, replace_newlines_with_eos=True)['positional_scores']
    return lm.score(prompt, replace_newlines_with_eos=True)['positional_scores']
    

# to 
pred_label = np.array(['entailment', 'contradiction', 'neutral'])
def XNLI_eval(premise, hypothesis, verbose=False):
    if verbose:
        prompt1, lprob1 = get_logprobs(premise + " , right? Yes, " + hypothesis, verbose)
        prompt2, lprob2 = get_logprobs(premise + " , right? No, " + hypothesis, verbose)
        prompt3, lprob3 = get_logprobs(premise + " , right? Also, " + hypothesis, verbose)
        lprob1 = lprob1.sum().cpu()
        lprob2 = lprob2.sum().cpu()
        lprob3 = lprob3.sum().cpu()
        output_file.write(prompt1+"\nScore: "+str(lprob1.item())+"\n")
        output_file.write(prompt2+"\nScore: "+str(lprob2.item())+"\n")
        output_file.write(prompt3+"\nScore: "+str(lprob3.item())+"\n\n")
    else:
        lprob1 = get_logprobs(premise + " , right? Yes, " + hypothesis, verbose).sum().cpu()
        lprob2 = get_logprobs(premise + " , right? No, " + hypothesis, verbose).sum().cpu()
        lprob3 = get_logprobs(premise + " , right? Also, " + hypothesis, verbose).sum().cpu()
    return pred_label[np.argmax([lprob1, lprob2, lprob3])]
    
def val_all(premise, hypothesis):
    lprob1 = get_logprobs(premise + " , right? Yes, " + hypothesis).sum().cpu()
    lprob2 = get_logprobs(premise + " , right? No, " + hypothesis).sum().cpu()
    lprob3 = get_logprobs(premise + " , right? Also, " + hypothesis).sum().cpu()
    return [lprob1, lprob2, lprob3]

# load xnli
lang = list()
label = list()
premise = list()
hypothesis = list()
for i, line in enumerate(open('data-bin/XNLI-1.0/xnli.test.tsv').readlines()):
    if i == 0:
        continue
    line = line.split('\t')
    lang.append(line[0])
    label.append(line[1])
    premise.append(line[6])
    hypothesis.append(line[7])


lang = np.array(lang)
label = np.array(label)
premise = np.array(premise)
hypothesis = np.array(hypothesis)

test_ind = np.where(lang == test_lang)
test_label = label[test_ind]
test_premise = premise[test_ind]
test_hypothesis = hypothesis[test_ind]

acc = 0.0
for i in tqdm(range(len(test_label))):
    predict = XNLI_eval(test_premise[i], test_hypothesis[i], verbose=(i%1000==0))
    if predict == test_label[i]:
        acc += 1.0


print('accuracy of zero-shot on {}: '.format(test_lang), acc/float(len(test_label)))
output_file.write('accuracy of zero-shot on {}: '.format(test_lang)+str(acc/float(len(test_label))))
output_file.write('\n')
output_file.close()


