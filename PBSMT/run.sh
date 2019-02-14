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
N_THREADS=48     # number of threads in data preprocessing
SRC=en           # source language
TGT=fr           # target language


#
# Initialize Moses and data paths
#

# main paths
UMT_PATH=$PWD
DATA_PATH=$PWD/data
MONO_PATH=$DATA_PATH/mono
PARA_PATH=$DATA_PATH/para
EMB_PATH=$DATA_PATH/embeddings

# create paths
mkdir -p $DATA_PATH
mkdir -p $MONO_PATH
mkdir -p $PARA_PATH
mkdir -p $EMB_PATH

# moses
MOSES_PATH=/private/home/guismay/tools/mosesdecoder  # PATH_WHERE_YOU_INSTALLED_MOSES
TOKENIZER=$MOSES_PATH/scripts/tokenizer/tokenizer.perl
NORM_PUNC=$MOSES_PATH/scripts/tokenizer/normalize-punctuation.perl
INPUT_FROM_SGM=$MOSES_PATH/scripts/ems/support/input-from-sgm.perl
REM_NON_PRINT_CHAR=$MOSES_PATH/scripts/tokenizer/remove-non-printing-char.perl
TRAIN_TRUECASER=$MOSES_PATH/scripts/recaser/train-truecaser.perl
TRUECASER=$MOSES_PATH/scripts/recaser/truecase.perl
DETRUECASER=$MOSES_PATH/scripts/recaser/detruecase.perl
TRAIN_LM=$MOSES_PATH/bin/lmplz
TRAIN_MODEL=$MOSES_PATH/scripts/training/train-model.perl
MULTIBLEU=$MOSES_PATH/scripts/generic/multi-bleu.perl
MOSES_BIN=$MOSES_PATH/bin/moses

# training directory
TRAIN_DIR=$PWD/moses_train_$SRC-$TGT

# MUSE path
MUSE_PATH=$PWD/MUSE

# files full paths
SRC_RAW=$MONO_PATH/all.$SRC
TGT_RAW=$MONO_PATH/all.$TGT
SRC_TOK=$MONO_PATH/all.$SRC.tok
TGT_TOK=$MONO_PATH/all.$TGT.tok
SRC_TRUE=$MONO_PATH/all.$SRC.true
TGT_TRUE=$MONO_PATH/all.$TGT.true
SRC_VALID=$PARA_PATH/dev/newstest2013-ref.$SRC
TGT_VALID=$PARA_PATH/dev/newstest2013-ref.$TGT
SRC_TEST=$PARA_PATH/dev/newstest2014-fren-src.$SRC
TGT_TEST=$PARA_PATH/dev/newstest2014-fren-src.$TGT
SRC_TRUECASER=$DATA_PATH/$SRC.truecaser
TGT_TRUECASER=$DATA_PATH/$TGT.truecaser
SRC_LM_ARPA=$DATA_PATH/$SRC.lm.arpa
TGT_LM_ARPA=$DATA_PATH/$TGT.lm.arpa
SRC_LM_BLM=$DATA_PATH/$SRC.lm.blm
TGT_LM_BLM=$DATA_PATH/$TGT.lm.blm


#
# Download and install tools
#

# Check Moses files
if ! [[ -f "$TOKENIZER" && -f "$NORM_PUNC" && -f "$INPUT_FROM_SGM" && -f "$REM_NON_PRINT_CHAR" && -f "$TRAIN_TRUECASER" && -f "$TRUECASER" && -f "$DETRUECASER" && -f "$TRAIN_MODEL" ]]; then
  echo "Some Moses files were not found."
  echo "Please update the MOSES variable to the path where you installed Moses."
  exit
fi
if ! [[ -f "$MOSES_BIN" ]]; then
  echo "Couldn't find Moses binary in: $MOSES_BIN"
  echo "Please check your installation."
  exit
fi
if ! [[ -f "$TRAIN_LM" ]]; then
  echo "Couldn't find language model trainer in: $TRAIN_LM"
  echo "Please install KenLM."
  exit
fi


# Download MUSE
if [ ! -d "$MUSE_PATH" ]; then
  echo "Cloning MUSE from GitHub repository..."
  git clone https://github.com/facebookresearch/MUSE.git
  cd $MUSE_PATH/data/
  ./get_evaluation.sh
fi
echo "MUSE found in: $MUSE_PATH"


#
# Download pretrained word embeddings
#

cd $EMB_PATH

if [ ! -f "cc.en.300.vec.gz" ]; then
  echo "Downloading $SRC pretrained embeddings..."
  wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz"
fi
if [ ! -f "cc.fr.300.vec.gz" ]; then
  echo "Downloading $TGT pretrained embeddings..."
  wget -c "https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fr.300.vec.gz"
fi

if [ ! -f "cc.en.300.vec" ]; then
  echo "Decompressing English pretrained embeddings..."
  gunzip -k cc.en.300.vec.gz
fi
if [ ! -f "cc.fr.300.vec" ]; then
  echo "Decompressing French pretrained embeddings..."
  gunzip -k cc.fr.300.vec.gz
fi

if [ "$SRC" == "en" ]; then EMB_SRC=$EMB_PATH/cc.en.300.vec; fi
if [ "$SRC" == "fr" ]; then EMB_SRC=$EMB_PATH/cc.fr.300.vec; fi
if [ "$TGT" == "en" ]; then EMB_TGT=$EMB_PATH/cc.en.300.vec; fi
if [ "$TGT" == "fr" ]; then EMB_TGT=$EMB_PATH/cc.fr.300.vec; fi

echo "Pretrained $SRC embeddings found in: $EMB_SRC"
echo "Pretrained $TGT embeddings found in: $EMB_TGT"


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

echo "Downloading French files..."
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.fr.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.fr.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.fr.shuffled.gz
wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.fr.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.fr.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.fr.shuffled.gz
# wget -c http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.fr.shuffled.gz
# wget -c http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.fr.shuffled.v2.gz
# wget -c http://data.statmt.org/wmt17/translation-task/news.2015.fr.shuffled.gz
# wget -c http://data.statmt.org/wmt17/translation-task/news.2016.fr.shuffled.gz
# wget -c http://data.statmt.org/wmt17/translation-task/news.2017.fr.shuffled.gz

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
  cat $(ls news*fr* | grep -v gz) | head -n $N_MONO > $TGT_RAW
fi
echo "$SRC monolingual data concatenated in: $SRC_RAW"
echo "$TGT monolingual data concatenated in: $TGT_RAW"

# check number of lines
if ! [[ "$(wc -l < $SRC_RAW)" -eq "$N_MONO" ]]; then echo "ERROR: Number of lines doesn't match! Be sure you have $N_MONO sentences in your $SRC monolingual data."; exit; fi
if ! [[ "$(wc -l < $TGT_RAW)" -eq "$N_MONO" ]]; then echo "ERROR: Number of lines doesn't match! Be sure you have $N_MONO sentences in your $TGT monolingual data."; exit; fi

# tokenize data
if ! [[ -f "$SRC_TOK" && -f "$TGT_TOK" ]]; then
  echo "Tokenize monolingual data..."
  cat $SRC_RAW | $NORM_PUNC -l $SRC | $TOKENIZER -l $SRC -no-escape -threads $N_THREADS > $SRC_TOK
  cat $TGT_RAW | $NORM_PUNC -l $TGT | $TOKENIZER -l $TGT -no-escape -threads $N_THREADS > $TGT_TOK
fi
echo "$SRC monolingual data tokenized in: $SRC_TOK"
echo "$TGT monolingual data tokenized in: $TGT_TOK"

# learn truecasers
if ! [[ -f "$SRC_TRUECASER" && -f "$TGT_TRUECASER" ]]; then
  echo "Learning truecasers..."
  $TRAIN_TRUECASER --model $SRC_TRUECASER --corpus $SRC_TOK
  $TRAIN_TRUECASER --model $TGT_TRUECASER --corpus $TGT_TOK
fi
echo "$SRC truecaser in: $SRC_TRUECASER"
echo "$TGT truecaser in: $TGT_TRUECASER"

# truecase data
if ! [[ -f "$SRC_TRUE" && -f "$TGT_TRUE" ]]; then
  echo "Truecsing monolingual data..."
  $TRUECASER --model $SRC_TRUECASER < $SRC_TOK > $SRC_TRUE
  $TRUECASER --model $TGT_TRUECASER < $TGT_TOK > $TGT_TRUE
fi
echo "$SRC monolingual data truecased in: $SRC_TRUE"
echo "$TGT monolingual data truecased in: $TGT_TRUE"

# learn language models
if ! [[ -f "$SRC_LM_ARPA" && -f "$TGT_LM_ARPA" ]]; then
  echo "Learning language models..."
  $TRAIN_LM -o 5 < $SRC_TRUE > $SRC_LM_ARPA
  $TRAIN_LM -o 5 < $TGT_TRUE > $TGT_LM_ARPA
fi
echo "$SRC language model in: $SRC_LM_ARPA"
echo "$TGT language model in: $TGT_LM_ARPA"

# binarize language models
if ! [[ -f "$SRC_LM_BLM" && -f "$TGT_LM_BLM" ]]; then
  echo "Binarizing language models..."
  $MOSES_PATH/bin/build_binary $SRC_LM_ARPA $SRC_LM_BLM
  $MOSES_PATH/bin/build_binary $TGT_LM_ARPA $TGT_LM_BLM
fi
echo "$SRC binarized language model in: $SRC_LM_BLM"
echo "$TGT binarized language model in: $TGT_LM_BLM"


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
$INPUT_FROM_SGM < $SRC_VALID.sgm | $NORM_PUNC -l $SRC | $REM_NON_PRINT_CHAR | $TOKENIZER -l $SRC -no-escape -threads $N_THREADS > $SRC_VALID.tok
$INPUT_FROM_SGM < $TGT_VALID.sgm | $NORM_PUNC -l $TGT | $REM_NON_PRINT_CHAR | $TOKENIZER -l $TGT -no-escape -threads $N_THREADS > $TGT_VALID.tok
$INPUT_FROM_SGM < $SRC_TEST.sgm | $NORM_PUNC -l $SRC | $REM_NON_PRINT_CHAR | $TOKENIZER -l $SRC -no-escape -threads $N_THREADS > $SRC_TEST.tok
$INPUT_FROM_SGM < $TGT_TEST.sgm | $NORM_PUNC -l $TGT | $REM_NON_PRINT_CHAR | $TOKENIZER -l $TGT -no-escape -threads $N_THREADS > $TGT_TEST.tok

echo "Truecasing valid and test data..."
$TRUECASER --model $SRC_TRUECASER < $SRC_VALID.tok > $SRC_VALID.true
$TRUECASER --model $TGT_TRUECASER < $TGT_VALID.tok > $TGT_VALID.true
$TRUECASER --model $SRC_TRUECASER < $SRC_TEST.tok > $SRC_TEST.true
$TRUECASER --model $TGT_TRUECASER < $TGT_TEST.tok > $TGT_TEST.true


#
# Running MUSE to generate cross-lingual embeddings
#

ALIGNED_EMBEDDINGS_SRC=$MUSE_PATH/alignments/wiki-released-$SRC$TGT-identical_char/vectors-$SRC.pth
ALIGNED_EMBEDDINGS_TGT=$MUSE_PATH/alignments/wiki-released-$SRC$TGT-identical_char/vectors-$TGT.pth

if ! [[ -f "$ALIGNED_EMBEDDINGS_SRC" && -f "$ALIGNED_EMBEDDINGS_TGT" ]]; then
  rm -rf $MUSE_PATH/alignments/
  echo "Aligning embeddings with MUSE..."
  python $MUSE_PATH/supervised.py --src_lang $SRC --tgt_lang $TGT \
  --exp_path $MUSE_PATH --exp_name alignments --exp_id wiki-released-$SRC$TGT-identical_char \
  --src_emb $EMB_SRC \
  --tgt_emb $EMB_TGT \
  --n_refinement 5 --dico_train identical_char --export "pth"
fi
echo "$SRC aligned embeddings: $ALIGNED_EMBEDDINGS_SRC"
echo "$TGT aligned embeddings: $ALIGNED_EMBEDDINGS_TGT"


#
# Generating a phrase-table in an unsupervised way
#

PHRASE_TABLE_PATH=$MUSE_PATH/alignments/wiki-released-$SRC$TGT-identical_char/phrase-table.$SRC-$TGT.gz
if ! [[ -f "$PHRASE_TABLE_PATH" ]]; then
  echo "Generating unsupervised phrase-table"
  python $UMT_PATH/create-phrase-table.py \
  --src_lang $SRC \
  --tgt_lang $TGT \
  --src_emb $ALIGNED_EMBEDDINGS_SRC \
  --tgt_emb $ALIGNED_EMBEDDINGS_TGT \
  --csls 1 \
  --max_rank 200 \
  --max_vocab 300000 \
  --inverse_score 1 \
  --temperature 45 \
  --phrase_table_path ${PHRASE_TABLE_PATH::-3}
fi
echo "Phrase-table location: $PHRASE_TABLE_PATH"


#
# Train Moses on the generated phrase-table
#

rm -rf $TRAIN_DIR
echo "Generating Moses configuration in: $TRAIN_DIR"

echo "Creating default configuration file..."
$TRAIN_MODEL -root-dir $TRAIN_DIR \
-f $SRC -e $TGT -alignment grow-diag-final-and -reordering msd-bidirectional-fe \
-lm 0:5:$TGT_LM_BLM:8 -external-bin-dir $MOSES_PATH/tools \
-cores $N_THREADS -max-phrase-length=4 -score-options "--NoLex" -first-step=9 -last-step=9
CONFIG_PATH=$TRAIN_DIR/model/moses.ini

echo "Removing lexical reordering features ..."
mv $TRAIN_DIR/model/moses.ini $TRAIN_DIR/model/moses.ini.bkp
cat $TRAIN_DIR/model/moses.ini.bkp | grep -v LexicalReordering > $TRAIN_DIR/model/moses.ini

echo "Linking phrase-table path..."
ln -sf $PHRASE_TABLE_PATH $TRAIN_DIR/model/phrase-table.gz

echo "Translating test sentences..."
$MOSES_BIN -threads $N_THREADS -f $CONFIG_PATH < $SRC_TEST.true > $TRAIN_DIR/test.$TGT.hyp.true

echo "Detruecasing hypothesis..."
$DETRUECASER < $TRAIN_DIR/test.$TGT.hyp.true > $TRAIN_DIR/test.$TGT.hyp.tok

echo "Evaluating translations..."
$MULTIBLEU $TGT_TEST.true < $TRAIN_DIR/test.$TGT.hyp.true > $TRAIN_DIR/eval.true
$MULTIBLEU $TGT_TEST.tok < $TRAIN_DIR/test.$TGT.hyp.tok > $TRAIN_DIR/eval.tok
cat $TRAIN_DIR/eval.tok

echo "End of training. Experiment is stored in: $TRAIN_DIR"
