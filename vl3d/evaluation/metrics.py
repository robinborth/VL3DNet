import copy
import math
from collections import defaultdict

import evaluate
import numpy as np
import torch
from absl import logging

from vl3d.evaluation.utils import bbox_iou

logging.set_verbosity(logging.ERROR)
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")


def acc_at_k_iou(gt_bboxes, pred_bboxes, unique_mask, threshold):
    """Counts numbers of positive/negative and unique/multiple predicitons based on a threshold.

    Args:
        gt_bboxes: The ground truth bounding boxes as list.
        pred_bboxes: The prediction bounding boxes as list.
        unique_mask: A list of [0, 1] where unique=0 and multiple=1.
        threshold: A threshold in (0, 1) which decided if a prediciton bbox is a positive one.
    """
    ious = bbox_iou(gt_bboxes, pred_bboxes)
    data = {
        "unique_positive": ((unique_mask == 0) & (ious > threshold)).sum(),
        "unique_negative": ((unique_mask == 0) & (ious <= threshold)).sum(),
        "multiple_positive": ((unique_mask == 1) & (ious > threshold)).sum(),
        "multiple_negative": ((unique_mask == 1) & (ious <= threshold)).sum(),
    }
    data["unique_acc"] = data["unique_positive"] / max((data["unique_positive"] + data["unique_negative"]), 1)
    data["multiple_acc"] = data["multiple_positive"] / max((data["multiple_positive"] + data["multiple_negative"]), 1)
    data["overall_acc"] = (data["unique_positive"] + data["multiple_positive"]) / max(
        (data["unique_positive"] + data["unique_negative"] + data["multiple_positive"] + data["multiple_negative"]), 1
    )
    return data


def grounding_metrics(output_dict) -> dict:
    metrics = {}
    for threshold in [0.25, 0.5]:
        m = acc_at_k_iou(
            gt_bboxes=output_dict["gt_bbox"],
            pred_bboxes=output_dict["pred_bbox"],
            unique_mask=output_dict["unique_mask"],
            threshold=threshold,
        )
        for acc_type in ["unique", "multiple", "overall"]:
            metrics[f"acc@{threshold}IoU_{acc_type}"] = m[f"{acc_type}_acc"]
    return metrics.items()


def captioning_metrics(output_dict, iou_threshold: float = 0.5) -> dict:
    # get the predictions and references
    predictions, references = output_dict["pred_caption"], output_dict["gt_caption"]
    references = [[ref] for ref in references]
    assert len(predictions) == len(references)
    batch_size = len(predictions)

    # calculate the ious to filter
    idx = output_dict["pred_bboxes_labels"].argmax(dim=1)
    running_idx = torch.arange(output_dict["pred_bboxes"].shape[0])
    pred_bbox = output_dict["pred_bboxes"][running_idx, idx]
    ious = bbox_iou(pred_bbox, output_dict["gt_bbox"])
    assert len(ious) == len(predictions)

    # filter predicitons and references based on iou threshold 0.5
    predictions = [p for (p, iou) in zip(predictions, ious) if iou >= iou_threshold]
    references = [p for (p, iou) in zip(references, ious) if iou >= iou_threshold]

    # prepare cider score
    cider_metric = CiderScorer()
    for test, refs in zip(predictions, references):
        cider_metric += (test, refs)

    # calculate the metrics
    metrics = {}
    metrics["B@0.5IoU"] = bleu_metric.compute(predictions=predictions, references=references)["bleu"]
    metrics["R@0.5IoU"] = rouge_metric.compute(predictions=predictions, references=references)["rougeL"]
    metrics["M@0.5IoU"] = meteor_metric.compute(predictions=predictions, references=references)["meteor"]
    metrics["C@0.5IoU"] = cider_metric.compute_score()[0]

    # normalize the metrics based on the batch size
    for name, value in metrics.items():
        if predictions:
            metrics[name] = value * (len(predictions) / batch_size)

    return metrics.items()


def f1_captioning_metrics(score, gt_count, pred_count):
    """Normalizes the metrics similar to the papers."""
    if score == 0:
        return 0
    ratio = gt_count / pred_count
    return (2 * score * score * ratio) / (score * ratio + score)


class CiderScorer(object):
    # implementation of cider metric
    # reference: https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/cider/cider_scorer.py

    def copy(self):
        """copy the refs."""
        new = CiderScorer(n=self.n)
        new.ctest = copy.copy(self.ctest)
        new.crefs = copy.copy(self.crefs)
        return new

    def __init__(self, test=None, refs=None, n=4, sigma=6.0):
        """singular instance"""
        self.n = n
        self.sigma = sigma
        self.crefs = []
        self.ctest = []
        self.document_frequency = defaultdict(float)
        self.cook_append(test, refs)
        self.ref_len = None

    def precook(self, s):
        """
        Takes a string as input and returns an object that can be given to
        either cook_refs or cook_test. This is optional: cook_refs and cook_test
        can take string arguments as well.
        :param s: string : sentence to be converted into ngrams
        :param n: int    : number of ngrams for which representation is calculated
        :return: term frequency vector for occuring ngrams
        """
        words = s.split()
        counts = defaultdict(int)
        for k in range(1, self.n + 1):
            for i in range(len(words) - k + 1):
                ngram = tuple(words[i : i + k])
                counts[ngram] += 1
        return counts

    def cook_append(self, test, refs):
        """called by constructor and __iadd__ to avoid creating new instances."""

        if refs is not None:
            self.crefs.append([self.precook(ref) for ref in refs])
            if test is not None:
                self.ctest.append(self.precook(test))
            else:
                self.ctest.append(None)

    def size(self):
        assert len(self.crefs) == len(self.ctest), "refs/test mismatch! %d<>%d" % (len(self.crefs), len(self.ctest))
        return len(self.crefs)

    def __iadd__(self, other):
        """add an instance (e.g., from another sentence)."""

        if type(other) is tuple:
            ## avoid creating new CiderScorer instances
            self.cook_append(other[0], other[1])
        else:
            self.ctest.extend(other.ctest)
            self.crefs.extend(other.crefs)

        return self

    def compute_doc_freq(self):
        """
        Compute term frequency for reference data.
        This will be used to compute idf (inverse document frequency later)
        The term frequency is stored in the object
        :return: None
        """
        for refs in self.crefs:
            # refs, k ref captions of one image
            for ngram in set([ngram for ref in refs for (ngram, count) in ref.items()]):
                self.document_frequency[ngram] += 1
            # maxcounts[ngram] = max(maxcounts.get(ngram,0), count)

    def compute_cider(self):
        def counts2vec(cnts):
            """
            Function maps counts of ngram to vector of tfidf weights.
            The function returns vec, an array of dictionary that store mapping of n-gram and tf-idf weights.
            The n-th entry of array denotes length of n-grams.
            :param cnts:
            :return: vec (array of dict), norm (array of float), length (int)
            """
            vec = [defaultdict(float) for _ in range(self.n)]
            length = 0
            norm = [0.0 for _ in range(self.n)]
            for (ngram, term_freq) in cnts.items():
                # give word count 1 if it doesn't appear in reference corpus
                df = np.log(max(1.0, self.document_frequency[ngram]))
                # ngram index
                n = len(ngram) - 1
                # tf (term_freq) * idf (precomputed idf) for n-grams
                vec[n][ngram] = float(term_freq) * (self.ref_len - df)
                # compute norm for the vector.  the norm will be used for computing similarity
                norm[n] += pow(vec[n][ngram], 2)

                if n == 1:
                    length += term_freq
            norm = [np.sqrt(n) for n in norm]
            return vec, norm, length

        def sim(vec_hyp, vec_ref, norm_hyp, norm_ref, length_hyp, length_ref):
            """
            Compute the cosine similarity of two vectors.
            :param vec_hyp: array of dictionary for vector corresponding to hypothesis
            :param vec_ref: array of dictionary for vector corresponding to reference
            :param norm_hyp: array of float for vector corresponding to hypothesis
            :param norm_ref: array of float for vector corresponding to reference
            :param length_hyp: int containing length of hypothesis
            :param length_ref: int containing length of reference
            :return: array of score for each n-grams cosine similarity
            """
            delta = float(length_hyp - length_ref)
            # measure consine similarity
            val = np.array([0.0 for _ in range(self.n)])
            for n in range(self.n):
                # ngram
                for (ngram, count) in vec_hyp[n].items():
                    # vrama91 : added clipping
                    val[n] += min(vec_hyp[n][ngram], vec_ref[n][ngram]) * vec_ref[n][ngram]

                if (norm_hyp[n] != 0) and (norm_ref[n] != 0):
                    val[n] /= norm_hyp[n] * norm_ref[n]

                assert not math.isnan(val[n])
                # vrama91: added a length based gaussian penalty
                val[n] *= np.e ** (-(delta**2) / (2 * self.sigma**2))
            return val

        # compute log reference length
        self.ref_len = np.log(float(len(self.crefs)))

        scores = []
        for test, refs in zip(self.ctest, self.crefs):
            # compute vector for test captions
            vec, norm, length = counts2vec(test)
            # compute vector for ref captions
            score = np.array([0.0 for _ in range(self.n)])
            for ref in refs:
                vec_ref, norm_ref, length_ref = counts2vec(ref)
                score += sim(vec, vec_ref, norm, norm_ref, length, length_ref)
            # change by vrama91 - mean of ngram scores, instead of sum
            score_avg = np.mean(score)
            # divide by number of references
            score_avg /= len(refs)
            # multiply score by 10
            score_avg *= 10.0
            # append score of an image to the score list
            scores.append(score_avg)
        return scores

    def compute_score(self, option=None, verbose=0):
        # compute idf
        self.compute_doc_freq()
        # assert to check document frequency
        assert len(self.ctest) >= max(self.document_frequency.values())
        # compute cider score
        score = self.compute_cider()
        return np.mean(np.array(score)), np.array(score)
