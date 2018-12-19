# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -e

#
# Data preprocessing configuration
#

N_MONO=10000000  # number of monolingual sentences for each language
CODES=60000      # number of BPE codes
N_THREADS=48     # number of threads in data preprocessing
N_EPOCHS=10      # number of fastText epochs


#
# Initialize tools and data paths
#

# main paths
UMT_PATH=$PWD
TOOLS_PATH=$PWD/tools
DATA_PATH=$PWD/data
MONO_PATH=$DATA_PATH/mono
PARA_PATH=$DATA_PATH/para

# create paths
mkdir -p $TOOLS_PATH
mkdir -p $DATA_PATH
mkdir -p $MONO_PATH
mkdir -p $PARA_PATH

# moses
MOSES=$TOOLS_PATH/mosesdecoder
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl

# fastBPE
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$FASTBPE_DIR/fast

# fastText
FASTTEXT_DIR=$TOOLS_PATH/fastText
FASTTEXT=$FASTTEXT_DIR/fasttext

# files full paths
SRC_RAW=$MONO_PATH/all.en
TGT_RAW=$MONO_PATH/all.de
SRC_TOK=$MONO_PATH/all.en.tok
TGT_TOK=$MONO_PATH/all.de.tok
BPE_CODES=$MONO_PATH/bpe_codes
CONCAT_BPE=$MONO_PATH/all.en-de.$CODES
SRC_VOCAB=$MONO_PATH/vocab.en.$CODES
TGT_VOCAB=$MONO_PATH/vocab.de.$CODES
FULL_VOCAB=$MONO_PATH/vocab.en-de.$CODES
SRC_VALID=$PARA_PATH/dev/newstest2015-deen-ref.en
TGT_VALID=$PARA_PATH/dev/newstest2015-deen-src.de
SRC_TEST=$PARA_PATH/dev/newstest2016-deen-ref.en
TGT_TEST=$PARA_PATH/dev/newstest2016-deen-src.de


#
# Download and install tools
#

# Download Moses
cd $TOOLS_PATH
if [ ! -d "$MOSES" ]; then
  echo "Cloning Moses from GitHub repository..."
  git clone https://github.com/moses-smt/mosesdecoder.git
fi
echo "Moses found in: $MOSES"

# Download fastBPE
cd $TOOLS_PATH
if [ ! -d "$FASTBPE_DIR" ]; then
  echo "Cloning fastBPE from GitHub repository..."
  git clone https://github.com/glample/fastBPE
fi
echo "fastBPE found in: $FASTBPE_DIR"

# Compile fastBPE
cd $TOOLS_PATH
if [ ! -f "$FASTBPE" ]; then
  echo "Compiling fastBPE..."
  cd $FASTBPE_DIR
  g++ -std=c++11 -pthread -O3 fast.cc -o fast
fi
echo "fastBPE compiled in: $FASTBPE"

# Download fastText
cd $TOOLS_PATH
if [ ! -d "$FASTTEXT_DIR" ]; then
  echo "Cloning fastText from GitHub repository..."
  git clone https://github.com/facebookresearch/fastText.git
fi
echo "fastText found in: $FASTTEXT_DIR"

# Compile fastText
cd $TOOLS_PATH
if [ ! -f "$FASTTEXT" ]; then
  echo "Compiling fastText..."
  cd $FASTTEXT_DIR
  make
fi
echo "fastText compiled in: $FASTTEXT"


#
# Download monolingual data
#

cd $MONO_PATH

echo "Downloading English files..."
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.en.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.en.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.en.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.en.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.en.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.en.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.en.shuffled.gz
# wget -c http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.en.shuffled.v2.gz
# wget -c http://data.statmt.org/wmt16/translation-task/news.2015.en.shuffled.gz
# wget -c http://data.statmt.org/wmt17/translation-task/news.2016.en.shuffled.gz
# wget -c http://data.statmt.org/wmt18/translation-task/news.2017.en.shuffled.deduped.gz

echo "Downloading German files..."
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.de.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.de.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.de.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.de.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.de.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.de.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.de.shuffled.gz
# wget -c http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.de.shuffled.v2.gz
# wget -c http://data.statmt.org/wmt16/translation-task/news.2015.de.shuffled.gz
# wget -c http://data.statmt.org/wmt17/translation-task/news.2016.de.shuffled.gz
# wget -c http://data.statmt.org/wmt18/translation-task/news.2017.de.shuffled.deduped.gz

# decompress monolingual data
for FILENAME in news*gz; do
  OUTPUT="${FILENAME::-3}"
  if [ ! -f "$OUTPUT" ]; then
    echo "Decompressing $FILENAME..."
    gunzip -k $FILENAME
  else
    echo "$OUTPUT already decompressed."
  fi
done

# concatenate monolingual data files
if ! [[ -f "$SRC_RAW" && -f "$TGT_RAW" ]]; then
  echo "Concatenating monolingual data..."
  cat $(ls news*en* | grep -v gz) | head -n $N_MONO > $SRC_RAW
  cat $(ls news*de* | grep -v gz) | head -n $N_MONO > $TGT_RAW
fi
echo "EN monolingual data concatenated in: $SRC_RAW"
echo "DE monolingual data concatenated in: $TGT_RAW"

# check number of lines
if ! [[ "$(wc -l < $SRC_RAW)" -eq "$N_MONO" ]]; then echo "ERROR: Number of lines doesn't match! Be sure you have $N_MONO sentences in your EN monolingual data."; exit; fi
if ! [[ "$(wc -l < $TGT_RAW)" -eq "$N_MONO" ]]; then echo "ERROR: Number of lines doesn't match! Be sure you have $N_MONO sentences in your DE monolingual data."; exit; fi

# tokenize data
if ! [[ -f "$SRC_TOK" && -f "$TGT_TOK" ]]; then
  echo "Tokenize monolingual data..."
  cat $SRC_RAW | $NORM_PUNC -l en | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_TOK
  cat $TGT_RAW | $NORM_PUNC -l de | $TOKENIZER -l de -no-escape -threads $N_THREADS > $TGT_TOK
fi
echo "EN monolingual data tokenized in: $SRC_TOK"
echo "DE monolingual data tokenized in: $TGT_TOK"

# learn BPE codes
if [ ! -f "$BPE_CODES" ]; then
  echo "Learning BPE codes..."
  $FASTBPE learnbpe $CODES $SRC_TOK $TGT_TOK > $BPE_CODES
fi
echo "BPE learned in $BPE_CODES"

# apply BPE codes
if ! [[ -f "$SRC_TOK.$CODES" && -f "$TGT_TOK.$CODES" ]]; then
  echo "Applying BPE codes..."
  $FASTBPE applybpe $SRC_TOK.$CODES $SRC_TOK $BPE_CODES
  $FASTBPE applybpe $TGT_TOK.$CODES $TGT_TOK $BPE_CODES
fi
echo "BPE codes applied to EN in: $SRC_TOK.$CODES"
echo "BPE codes applied to DE in: $TGT_TOK.$CODES"

# extract vocabulary
if ! [[ -f "$SRC_VOCAB" && -f "$TGT_VOCAB" && -f "$FULL_VOCAB" ]]; then
  echo "Extracting vocabulary..."
  $FASTBPE getvocab $SRC_TOK.$CODES > $SRC_VOCAB
  $FASTBPE getvocab $TGT_TOK.$CODES > $TGT_VOCAB
  $FASTBPE getvocab $SRC_TOK.$CODES $TGT_TOK.$CODES > $FULL_VOCAB
fi
echo "EN vocab in: $SRC_VOCAB"
echo "DE vocab in: $TGT_VOCAB"
echo "Full vocab in: $FULL_VOCAB"

# binarize data
if ! [[ -f "$SRC_TOK.$CODES.pth" && -f "$TGT_TOK.$CODES.pth" ]]; then
  echo "Binarizing data..."
  $UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TOK.$CODES
  $UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TOK.$CODES
fi
echo "EN binarized data in: $SRC_TOK.$CODES.pth"
echo "DE binarized data in: $TGT_TOK.$CODES.pth"


#
# Download parallel data (for evaluation only)
#

cd $PARA_PATH

echo "Downloading parallel data..."
wget -c http://data.statmt.org/wmt17/translation-task/dev.tgz

echo "Extracting parallel data..."
tar -xzf dev.tgz

# check valid and test files are here
if ! [[ -f "$SRC_VALID.sgm" ]]; then echo "$SRC_VALID.sgm is not found!"; exit; fi
if ! [[ -f "$TGT_VALID.sgm" ]]; then echo "$TGT_VALID.sgm is not found!"; exit; fi
if ! [[ -f "$SRC_TEST.sgm" ]]; then echo "$SRC_TEST.sgm is not found!"; exit; fi
if ! [[ -f "$TGT_TEST.sgm" ]]; then echo "$TGT_TEST.sgm is not found!"; exit; fi

echo "Tokenizing valid and test data..."
$INPUT_FROM_SGM < $SRC_VALID.sgm | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_VALID
$INPUT_FROM_SGM < $TGT_VALID.sgm | $NORM_PUNC -l de | $REM_NON_PRINT_CHAR | $TOKENIZER -l de -no-escape -threads $N_THREADS > $TGT_VALID
$INPUT_FROM_SGM < $SRC_TEST.sgm | $NORM_PUNC -l en | $REM_NON_PRINT_CHAR | $TOKENIZER -l en -no-escape -threads $N_THREADS > $SRC_TEST
$INPUT_FROM_SGM < $TGT_TEST.sgm | $NORM_PUNC -l de | $REM_NON_PRINT_CHAR | $TOKENIZER -l de -no-escape -threads $N_THREADS > $TGT_TEST

echo "Applying BPE to valid and test files..."
$FASTBPE applybpe $SRC_VALID.$CODES $SRC_VALID $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $TGT_VALID.$CODES $TGT_VALID $BPE_CODES $TGT_VOCAB
$FASTBPE applybpe $SRC_TEST.$CODES $SRC_TEST $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $TGT_TEST.$CODES $TGT_TEST $BPE_CODES $TGT_VOCAB

echo "Binarizing data..."
rm -f $SRC_VALID.$CODES.pth $TGT_VALID.$CODES.pth $SRC_TEST.$CODES.pth $TGT_TEST.$CODES.pth
$UMT_PATH/preprocess.py $FULL_VOCAB $SRC_VALID.$CODES
$UMT_PATH/preprocess.py $FULL_VOCAB $TGT_VALID.$CODES
$UMT_PATH/preprocess.py $FULL_VOCAB $SRC_TEST.$CODES
$UMT_PATH/preprocess.py $FULL_VOCAB $TGT_TEST.$CODES


#
# Summary
#
echo ""
echo "===== Data summary"
echo "Monolingual training data:"
echo "    EN: $SRC_TOK.$CODES.pth"
echo "    DE: $TGT_TOK.$CODES.pth"
echo "Parallel validation data:"
echo "    EN: $SRC_VALID.$CODES.pth"
echo "    DE: $TGT_VALID.$CODES.pth"
echo "Parallel test data:"
echo "    EN: $SRC_TEST.$CODES.pth"
echo "    DE: $TGT_TEST.$CODES.pth"
echo ""


#
# Train fastText on concatenated embeddings
#

if ! [[ -f "$CONCAT_BPE" ]]; then
  echo "Concatenating source and target monolingual data..."
  cat $SRC_TOK.$CODES $TGT_TOK.$CODES | shuf > $CONCAT_BPE
fi
echo "Concatenated data in: $CONCAT_BPE"

if ! [[ -f "$CONCAT_BPE.vec" ]]; then
  echo "Training fastText on $CONCAT_BPE..."
  $FASTTEXT skipgram -epoch $N_EPOCHS -minCount 0 -dim 512 -thread $N_THREADS -ws 5 -neg 10 -input $CONCAT_BPE -output $CONCAT_BPE
fi
echo "Cross-lingual embeddings in: $CONCAT_BPE.vec"
