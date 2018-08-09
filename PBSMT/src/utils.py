import torch


def get_nn_avg_dist(emb, query, knn, bs=256):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    """
    all_scores = []
    emb = emb.transpose(0, 1).contiguous()
    for k, i in enumerate(range(0, query.size(0), bs)):
        if k % 50 == 0:
            print(i, end="... ", flush=True)
        scores = query[i:i + bs].mm(emb)
        best_scores, _ = scores.topk(knn, dim=1, largest=True, sorted=True)
        all_scores.append(best_scores.mean(1))
    all_scores = torch.cat(all_scores)
    assert all_scores.size() == (query.size(0),)
    print("")
    return all_scores


def get_translations(src_emb, tgt_emb, src_avg_dist, tgt_avg_dist, n_translate, bs=256):
    """
    Get translations.
    """
    assert not ((src_avg_dist is None) ^ (tgt_avg_dist is None))
    translations = []
    for k, i in enumerate(range(0, src_emb.size(0), bs)):
        if k % 50 == 0:
            print(i, end="... ", flush=True)
        scores = src_emb[i:i + bs].mm(tgt_emb.transpose(0, 1))
        if src_avg_dist is not None:
            scores.mul_(2)
            scores.sub_(src_avg_dist[i:i + bs, None] + tgt_avg_dist[None, :])
        _, idx = scores.topk(n_translate, dim=1, largest=True, sorted=True)
        translations.append(idx.cpu())
    translations = torch.cat(translations, 0)
    assert translations.size() == (len(src_emb), n_translate)
    print("")
    return translations


def get_s2t_scores(src_emb, tgt_emb, s2t_translations, temperature, bs=256):
    """
    Get source-to-target scores.
    """
    assert s2t_translations.size(0) == src_emb.size(0)
    all_scores = []
    _s2t_translations = s2t_translations.transpose(0, 1).cuda()
    for k, i in enumerate(range(0, src_emb.size(0), bs)):
        if k % 50 == 0:
            print(i, end="... ", flush=True)
        scores = tgt_emb.mm(src_emb[i:i + bs].transpose(0, 1))
        scores.mul_(temperature).exp_()
        scores.div_(scores.sum(0, keepdim=True).expand_as(scores))
        scores = scores.gather(0, _s2t_translations[:, i:i + bs]).transpose(0, 1)
        all_scores.append(scores.cpu())
    all_scores = torch.cat(all_scores, 0)
    assert all_scores.size() == s2t_translations.size()
    print("")
    return all_scores
