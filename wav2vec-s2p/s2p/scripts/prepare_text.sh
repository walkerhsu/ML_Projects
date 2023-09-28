#!/usr/bin/env zsh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
KENLM_ROOT='/home/b07502072/kenlm/build/bin'
lg=$1
text_path=$2
target_dir=$3
min_phones=$4
phonemizer=$5
lid_path=$6
skip_prep=$7

if [ -z "$lid_path" ]; then
  lid_path="lid.187.bin"
fi

if [ -z "$skip_prep" ]; then
  skip_prep="false"
fi

ph_lg=${lg:l}
if test "$lg" = 'fr'; then
  ph_lg='fr-fr'
elif test "$lg" = 'en'; then
  ph_lg='en-us'
elif test "$lg" = 'pt'; then
  ph_lg='pt-br'
fi

USE_ESPEAK=''
if test "$phonemizer" = 'espeak'; then
  ESPEAK_PATH='YES'
elif test "$phonemizer" = 'espeak-ng'; then
  ESPEAK_PATH=$(which espeak-ng)
elif test "$phonemizer" = 'G2P'; then
  ESPEAK_PATH=''
else
  echo "Unknown phonemizer $phonemizer. Valid options are espeak, espean-ng and G2P"
  exit 1
fi

echo "lang: $lg"
echo "phone lang: $ph_lg"
echo "text path: $text_path"
echo "target dir: $target_dir"
echo "use espeak: $ESPEAK_PATH"
echo "skip prep: $skip_prep"
echo "min phone seen threshold is $min_phones"

if test "$skip_prep" = "false"; then
  mkdir -p $target_dir
  python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/normalize_and_filter_text.py --lang $lg --fasttext-model $lid_path < $text_path | grep -v '\-\-\-' > $target_dir/lm.upper.lid.txt
  ##python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/normalize_and_filter_text.py --lang $lg --fasttext-model $lid_path < $text_path | grep -v '\-\-\-'
  python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $target_dir/lm.upper.lid.txt --only-source --destdir $target_dir --thresholdsrc 2 --padding-factor 1 --dict-only
  cut -f1 -d' ' $target_dir/dict.txt | grep -v -x '[[:punct:]]*' | grep -Pv '\d\d\d\d\d+' > $target_dir/words.txt

  echo "complete preprocess and create words.txt"

  if [ -z "$ESPEAK_PATH" ]; then
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/g2p_wrd_to_phn.py --compact < $target_dir/words.txt > $target_dir/phones.txt
  else
    # echoing 1 into corpus will prevent the mismatch lines between lexicon and phones in case the phonemizer fails
    # one=$(echo "1" | PHONEMIZER_ESPEAK_PATH=$ESPEAK_PATH phonemize -p ' ' -w '' -l $ph_lg --language-switch remove-flags)
    # sed 's/$/ 1/' $target_dir/words.txt | PHONEMIZER_ESPEAK_PATH=$ESPEAK_PATH phonemize -o $target_dir/phones.txt -p ' ' -w '' -l $ph_lg -j 70 --language-switch remove-flags
    # echo "one is ${one}"
    # sed -i "s/${one}$//" $target_dir/phones.txt
    python /home/b07502072/u-speech2speech/s2p/scripts/phonemize_text.py $target_dir --lang $ph_lg
    paste $target_dir/words.txt $target_dir/phones.txt > $target_dir/lexicon.lst
  fi
else
  echo "-----------------skip preprocess---------------"
fi


python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $target_dir/phones.txt --only-source --destdir $target_dir/phones --thresholdsrc $min_phones --padding-factor 1 --dict-only

python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/filter_lexicon.py -d $target_dir/phones/dict.txt < $target_dir/lexicon.lst > $target_dir/lexicon_filtered.lst
python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/phonemize_with_sil.py -s 0.25 --surround --lexicon $target_dir/lexicon_filtered.lst < $target_dir/lm.upper.lid.txt > $target_dir/phones/lm.phones.filtered.txt
cp $target_dir/phones/dict.txt $target_dir/phones/dict.phn.txt
echo "<SIL> 0" >> $target_dir/phones/dict.phn.txt
python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $target_dir/phones/lm.phones.filtered.txt --workers 70 --only-source --destdir $target_dir/phones --srcdict $target_dir/phones/dict.phn.txt

$KENLM_ROOT/lmplz -o 4 -S 50G -T ./tmp < $target_dir/lm.upper.lid.txt --discount_fallback --prune 0 0 0 3 > $target_dir/kenlm.wrd.o40003.arpa
$KENLM_ROOT/build_binary $target_dir/kenlm.wrd.o40003.arpa $target_dir/kenlm.wrd.o40003.bin

$KENLM_ROOT/lmplz -o 4 -S 50G -T ./tmp < $target_dir/phones/lm.phones.filtered.txt --discount_fallback > $target_dir/phones/lm.phones.filtered.04.arpa
$KENLM_ROOT/build_binary $target_dir/phones/lm.phones.filtered.04.arpa $target_dir/phones/lm.phones.filtered.04.bin
$KENLM_ROOT/lmplz -o 6 -S 50G -T ./tmp < $target_dir/phones/lm.phones.filtered.txt --discount_fallback > $target_dir/phones/lm.phones.filtered.06.arpa
$KENLM_ROOT/build_binary $target_dir/phones/lm.phones.filtered.06.arpa $target_dir/phones/lm.phones.filtered.06.bin

# skip using kaldi
# lg=$lg python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_words_sil lm_arpa=$target_dir/kenlm.wrd.o40003.arpa wav2letter_lexicon=$target_dir/lexicon_filtered.lst data_dir=$target_dir/phones in_labels=phn "blank_symbol='<SIL>'"
# lg=$lg python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_words lm_arpa=$target_dir/kenlm.wrd.o40003.arpa wav2letter_lexicon=$target_dir/lexicon_filtered.lst data_dir=$target_dir/phones in_labels=phn
# lg=$lg python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_phn_sil lm_arpa=$target_dir/phones/lm.phones.filtered.06.arpa data_dir=$target_dir/phones in_labels=phn "blank_symbol='<SIL>'"