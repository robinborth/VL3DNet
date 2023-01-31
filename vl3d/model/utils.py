from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


def encode_teacher_sequences(attention_masks):
    """Prepares input_ids for teacher forcing.

    Args:
        input_ids: The raw input ids from the description
        attention_mask: The raw attention mask that masked the raw input from padding

    Returns:
       A tensor/list of all the sequences that are needed for teacher forcing.
    """
    batch_size, max_length = attention_masks.shape[:2]
    actual_token_numbers = attention_masks.sum(dim=1)
    attention_masks_2d = []
    for idx in range(batch_size):
        actual_token_number = int(actual_token_numbers[idx])
        attention_mask_2d = attention_masks.new_full((actual_token_number, max_length), fill_value=1)
        attention_mask_2d = torch.tril(attention_mask_2d, diagonal=0)
        attention_mask_2d = F.pad(attention_mask_2d, (0, 0, 0, max_length - actual_token_number), value=0)
        diagonal_eye = torch.eye(max_length, max_length)
        attention_mask_2d[diagonal_eye == 1] = 1
        attention_masks_2d.append(attention_mask_2d)
    attention_masks_2d = torch.stack(attention_masks_2d, dim=0)
    return attention_masks_2d.bool()


def vision_language_attention_mask(vision_masks, language_masks):
    """Creates a padded attention mask for vision language fusion model.

    Args:
        vision_mask: data_dict["pred_bboxes_attention_mask"] of shape (batch_size, max_vision_length, max_vision_length)
        language_attention_mask: A attention mask received from the encode_teacher_sequences of shape
        (batch_size, language_max_length, language_max_length)
    """
    language_masks = language_masks.int()
    batch_size, vision_max_length = vision_masks.shape
    language_max_length = language_masks.shape[1]
    actual_token_lengths = torch.max(language_masks.sum(dim=-1), dim=-1).values

    fusion_masks = []
    for idx in range(batch_size):
        vision_1d_mask = vision_masks[idx]
        vision_2d_mask = vision_1d_mask.new_full((vision_max_length, vision_max_length), fill_value=1)
        vision_2d_mask[torch.nonzero(~vision_1d_mask.bool())] = vision_1d_mask
        diagonal_eye = torch.eye(vision_max_length, vision_max_length)
        vision_2d_mask[diagonal_eye == 1] = 0
        language_mask = language_masks[idx]
        actual_token_length = actual_token_lengths[idx]

        left_mask = torch.concat(
            (
                vision_2d_mask,
                (x := torch.stack([vision_1d_mask] * actual_token_length, dim=0)),
                (
                    y := vision_1d_mask.new_full(
                        (language_max_length - actual_token_length, vision_max_length), fill_value=1
                    )
                ),
            ),
            dim=0,
        )
        right_mask = torch.concat(
            (language_mask.new_full((vision_max_length, language_max_length), fill_value=0), language_mask)
        ).bool()
        right_mask = ~right_mask
        fusion_mask = torch.concat((left_mask, right_mask), dim=1)
        fusion_masks.append(fusion_mask)
    fusion_masks = torch.stack(fusion_masks, dim=0)
    return fusion_masks.bool()


def vision_language_key_padding_mask(vision_mask, language_mask):
    """Creates the key padded mask for the fusion module for grounding."""
    return torch.cat((vision_mask.bool(), ~language_mask.bool()), dim=1)


def greedy_search(forward, data_dict, max_length: int = 60):
    tmp_data_dict = deepcopy(data_dict)

    # clear the language_attention_mask
    tmp_data_dict["language_attention_mask"][:, :] = 0
    tmp_data_dict["language_attention_mask"][:, 0] = 1

    # clear the input_ids
    tmp_data_dict["input_ids"][:, :] = 0
    tmp_data_dict["input_ids"][:, 0] = 101

    for step in range(max_length - 1):
        output_token = forward(tmp_data_dict)["captioning_logits"][:, step, :].argmax(dim=1)
        tmp_data_dict["language_attention_mask"][:, step + 1] = 1
        tmp_data_dict["input_ids"][:, step + 1] = output_token

    return tmp_data_dict["input_ids"]


def beam_search(forward, data_dict, beam_size: int = 4, max_length: int = 60, alpha: float = 0.7):
    copy_data_dict = deepcopy(data_dict)
    # clear the language_attention_mask
    copy_data_dict["language_attention_mask"][:, :] = 0
    copy_data_dict["language_attention_mask"][:, 0] = 1

    # clear the input_ids
    copy_data_dict["input_ids"][:, :] = 0
    copy_data_dict["input_ids"][:, 0] = 101

    output_logits = forward(copy_data_dict)["captioning_logits"][:, 0]
    log_probs = torch.log(F.softmax(output_logits, dim=0))
    top_log_probs, top_output_tokens = torch.topk(log_probs, beam_size, dim=1, largest=True, sorted=True)

    beam_input_ids = []

    # calculate the description for each batch item seperate
    for batch_idx in range(copy_data_dict["input_ids"].size(0)):
        # set the first predicted tokens for the batch
        tmp_data_dict = {
            name: copy_data_dict[name][batch_idx].repeat(beam_size, 1)
            for name in [
                "gt_bbox",
                "pred_bboxes",
                "pred_bboxes_features",
                "pred_bboxes_attention_mask",
                "pred_bboxes_labels",
                "input_ids",
                "language_attention_mask",
            ]
        }
        tmp_data_dict["pred_bboxes"] = tmp_data_dict["pred_bboxes"].view(beam_size, 60, -1)
        tmp_data_dict["pred_bboxes_features"] = tmp_data_dict["pred_bboxes_features"].view(beam_size, 60, -1)
        tmp_data_dict["input_ids"] = copy_data_dict["input_ids"][batch_idx].repeat(beam_size, 1)
        tmp_data_dict["input_ids"][:, 1] = top_output_tokens[batch_idx]
        tmp_data_dict["language_attention_mask"] = copy_data_dict["language_attention_mask"][batch_idx].repeat(
            beam_size, 1
        )
        tmp_data_dict["language_attention_mask"][:, 1] = 1
        top_log_probs_store = top_log_probs[batch_idx]  # (beam_size)

        sequence_store = []

        # calculate autoregressivly the tokens
        for step in range(1, max_length - 1):
            # get the best beam_size tokens
            with torch.no_grad():
                output_logits = forward(tmp_data_dict)["captioning_logits"][:, step]

            log_probs = torch.log(F.softmax(output_logits, dim=0))  # (beam_size, 60, V)
            log_probs = log_probs + top_log_probs_store[:, None]
            tmp_top_log_probs, tmp_top_output_tokens = torch.topk(
                log_probs, beam_size, dim=1, largest=True, sorted=True
            )
            tmp_top_log_probs = tmp_top_log_probs.view(-1)
            tmp_top_log_probs, top_idx = torch.topk(tmp_top_log_probs, beam_size, dim=0, largest=True, sorted=True)
            tmp_top_output_tokens = tmp_top_output_tokens.view(-1)[top_idx]

            tmp_data_dict["language_attention_mask"][:, step + 1] = 1

            # get the best paths
            ids = []
            for input_ids in tmp_data_dict["input_ids"]:
                ids.extend(input_ids.repeat(beam_size, 1))
            stacked_input_ids = torch.stack(ids)
            stacked_input_ids[top_idx, step + 1] = tmp_top_output_tokens
            tmp_data_dict["input_ids"] = stacked_input_ids[top_idx]

            # when sequence reaches end of seaquence stop and store the sequence
            eos_idx = tmp_top_output_tokens == 102
            if eos_idx.sum() != 0:
                for log_probs, input_ids in zip(tmp_top_log_probs[eos_idx], tmp_data_dict["input_ids"][eos_idx]):
                    sequence_store.append({"input_ids": input_ids, "step": step, "log_probs": log_probs})
                tmp_top_log_probs[tmp_top_output_tokens == 102] = -1e5

            top_log_probs_store = tmp_top_log_probs

        # add the input ids that didn't reach the eos token
        for log_probs, input_ids in zip(tmp_top_log_probs, tmp_data_dict["input_ids"]):
            sequence_store.append({"input_ids": input_ids, "step": step, "log_probs": log_probs})

        best_score = -1e20
        best_input_ids = None
        for sequence in sequence_store:
            modifier = ((5 + sequence["step"]) ** alpha) / ((5 + 1) ** alpha)
            score = sequence["log_probs"] / modifier
            if score > best_score:
                best_score = score
                best_input_ids = sequence["input_ids"]

        beam_input_ids.append(best_input_ids)

    batch_input_ids = torch.stack(beam_input_ids, dim=0)
    return batch_input_ids
