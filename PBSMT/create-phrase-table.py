import os
import argparse
import numpy as np

from src.loader import load_embeddings
from src.utils import get_nn_avg_dist, get_translations, get_s2t_scores


N_TRANSLATE = 100
PHRASE_PATTERN = '{src_phrase} ||| {tgt_phrase} ||| {scores} ||| {alignment} ||| {counts} ||| |||'


# read parameters
parser = argparse.ArgumentParser(description='Create an unsupervised phrase')
parser.add_argument("--src_lang", type=str, default="", help="Source language")
parser.add_argument("--tgt_lang", type=str, default="", help="Target language")
parser.add_argument("--src_emb", type=str, default="", help="Source embeddings")
parser.add_argument("--tgt_emb", type=str, default="", help="Target embeddings")
parser.add_argument("--csls", type=int, default=1, help="Use CSLS (0 to disable)")
parser.add_argument("--max_rank", type=int, default=10, help="Max rank of translations to consider")
parser.add_argument("--max_vocab", type=int, default=-1, help="Max vocabulary size (-1 to disable)")
parser.add_argument("--inverse_score", type=int, default=1, help="Include inverse phrase translation probability")
parser.add_argument("--temperature", type=float, default=1, help="Temperature")
parser.add_argument("--phrase_table_path", type=str, default="", help="Phrase table path")
parser.add_argument("--min_proba", type=float, default=1e-12, help="Minimum phrase probability")
parser.add_argument("--delimiter", type=str, default="_", help="Replace phrase word delimiter by empty space (empty string to disable)")
params = parser.parse_args()

# read embeddings
print("Loading embeddings ...")
src_dico, src_emb = load_embeddings(params, source=True)
tgt_dico, tgt_emb = load_embeddings(params, source=False)
n_src = src_emb.size(0)
n_tgt = tgt_emb.size(0)
print("Loaded %i / %i source / target embeddings." % (n_src, n_tgt))

# use CSLS
print("Computing average distance ...")
src_avg_dist = get_nn_avg_dist(emb=tgt_emb, query=src_emb, knn=10) if params.csls else None
tgt_avg_dist = get_nn_avg_dist(emb=src_emb, query=tgt_emb, knn=10) if params.csls else None

# get translations
print("Generating translations ...")
s2t_translations = get_translations(src_emb, tgt_emb, src_avg_dist, tgt_avg_dist, N_TRANSLATE)
if params.inverse_score:
    t2s_translations = get_translations(tgt_emb, src_emb, tgt_avg_dist, src_avg_dist, N_TRANSLATE)

# get scores
print("Generating scores ...")
s2t_scores = get_s2t_scores(src_emb, tgt_emb, s2t_translations, params.temperature)
if params.inverse_score:
    t2s_scores = get_s2t_scores(tgt_emb, src_emb, t2s_translations, params.temperature)

# write the phrase table
print("Writing the phrase table to %s ..." % params.phrase_table_path)
trad_found = 0
no_trad_found = 0
skipped_phrases = 0

with open(params.phrase_table_path, 'w', encoding='utf-8') as f:

    for src_idx in range(len(src_dico)):

        if src_idx % 50000 == 0:
            print(src_idx, end="... ", flush=True)

        # load translations
        pairs = []
        for j in range(s2t_translations.size(1)):
            tgt_idx = s2t_translations[src_idx, j]
            direct_score = s2t_scores[src_idx, j]
            if params.inverse_score:
                back_idx = np.where(t2s_translations[tgt_idx] == src_idx)[0]
                if len(back_idx) == 0:
                    continue
                assert len(back_idx) == 1
                inverse_score = t2s_scores[tgt_idx, back_idx[0]]
            else:
                inverse_score = None
            pairs.append((tgt_dico[tgt_idx.item()], direct_score, inverse_score))
            if len(pairs) == params.max_rank:
                break
        if len(pairs) == 0:
            assert params.inverse_score
            no_trad_found += 1
            continue
        trad_found += len(pairs)

        # renormalize probabilities / sort pairs by probabilities
        sum_prob = sum(d for _, d, _ in pairs)
        pairs = [(w, d / sum_prob, r) for w, d, r in pairs]
        pairs = sorted(pairs, key=lambda x: -x[1])

        # write phrase scores
        for w_tgt, d, r in pairs:
            w_src = src_dico[src_idx]
            if d < params.min_proba or r is not None and r < params.min_proba:
                skipped_phrases += 1
                continue
            if params.inverse_score:
                phrase_scores = '%e %e' % (r, d)
            else:
                phrase_scores = '%e' % d
            if params.delimiter != '':  # split phrase word
                w_src = w_src.replace(params.delimiter, " ") if w_src != params.delimiter else w_src
                w_tgt = w_tgt.replace(params.delimiter, " ") if w_tgt != params.delimiter else w_tgt
            f.write(PHRASE_PATTERN.format(
                src_phrase=w_src,
                tgt_phrase=w_tgt,
                scores=phrase_scores,
                alignment='',
                counts='',
            ))
            f.write('\n')
    print("\nFound %i translations for %i words. %i words did not get assigned "
          "any translation. %i pairs were skipped because of low probability."
          % (trad_found, n_src, no_trad_found, skipped_phrases))

print("Gzip phrase table ...")
os.system("gzip -f %s" % params.phrase_table_path)
