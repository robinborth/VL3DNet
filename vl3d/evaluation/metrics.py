import evaluate
import torch
from absl import logging

from vl3d.evaluation.utils import bbox_iou
from vl3d.evaluation.cider import CiderScorer

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
