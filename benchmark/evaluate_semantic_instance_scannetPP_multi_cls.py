
import os
import os.path as osp
import sys
from copy import deepcopy
from uuid import uuid4

from matplotlib import pyplot as plt
import numpy as np

import benchmark.util_3d as util_3d

# ---------- Evaluation params ---------- #
opt = {}
opt["overlaps"] = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
opt["min_region_sizes"] = np.array([100])
opt["distance_threshes"] = np.array([float("inf")])
opt["distance_confs"] = np.array([-float("inf")])


def _build_label_maps(class_names, valid_class_ids):
    """
    Build CLASS_LABELS / ID_TO_LABEL / LABEL_TO_ID.

    - valid_class_ids are the ids used in gt_ids//1000 and pred_classes.
    - If class_names is provided, id `k` maps to class_names[k-1] (so ids are 1-based).
    """
    class_labels = []
    id_to_label = {}
    label_to_id = {}

    valid_class_ids = np.asarray(valid_class_ids, dtype=np.int64)
    for cid in valid_class_ids.tolist():
        if class_names is not None and 1 <= cid <= len(class_names):
            name = str(class_names[cid - 1])
        else:
            name = f"class_{cid}"
        class_labels.append(name)
        id_to_label[int(cid)] = name
        label_to_id[name] = int(cid)
    return class_labels, valid_class_ids, id_to_label, label_to_id


def evaluate_matches(matches, prcurv_save_dir):
    overlaps = opt["overlaps"]
    min_region_sizes = [opt["min_region_sizes"][0]]
    dist_threshes = [opt["distance_threshes"][0]]
    dist_confs = [opt["distance_confs"][0]]

    ap = np.zeros((len(dist_threshes), len(eval_class_labels), len(overlaps)), float)
    rc = np.zeros((len(dist_threshes), len(eval_class_labels), len(overlaps)), float)
    pr = np.zeros((len(dist_threshes), len(eval_class_labels), len(overlaps)), float)
    for di, (min_region_size, distance_thresh, distance_conf) in enumerate(
        zip(min_region_sizes, dist_threshes, dist_confs)
    ):
        for oi, overlap_th in enumerate(overlaps):
            pred_visited = {}
            for m in matches:
                for label_name in eval_class_labels:
                    for p in matches[m]["pred"][label_name]:
                        if "uuid" in p:
                            pred_visited[p["uuid"]] = False

            for li, label_name in enumerate(eval_class_labels):
                y_true = np.empty(0)
                y_score = np.empty(0)
                hard_false_negatives = 0
                has_gt = False
                has_pred = False

                for m in matches:
                    pred_instances = matches[m]["pred"][label_name]
                    gt_instances = matches[m]["gt"][label_name]
                    gt_instances = [
                        gt
                        for gt in gt_instances
                        if gt["instance_id"] >= 0
                        and gt["vert_count"] >= min_region_size
                        and gt["med_dist"] <= distance_thresh
                        and gt["dist_conf"] >= distance_conf
                    ]
                    if gt_instances:
                        has_gt = True
                    if pred_instances:
                        has_pred = True

                    cur_true = np.ones(len(gt_instances))
                    cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                    cur_match = np.zeros(len(gt_instances), dtype=bool)

                    for gti, gt in enumerate(gt_instances):
                        found_match = False
                        for pred in gt["matched_pred"]:
                            if pred_visited[pred["uuid"]]:
                                continue
                            overlap = float(pred["intersection"]) / (
                                gt["vert_count"] + pred["vert_count"] - pred["intersection"]
                            )
                            if overlap > overlap_th:
                                confidence = pred["confidence"]
                                if cur_match[gti]:
                                    max_score = max(cur_score[gti], confidence)
                                    min_score = min(cur_score[gti], confidence)
                                    cur_score[gti] = max_score
                                    cur_true = np.append(cur_true, 0)
                                    cur_score = np.append(cur_score, min_score)
                                    cur_match = np.append(cur_match, True)
                                else:
                                    found_match = True
                                    cur_match[gti] = True
                                    cur_score[gti] = confidence
                                    pred_visited[pred["uuid"]] = True
                        if not found_match:
                            hard_false_negatives += 1

                    cur_true = cur_true[cur_match == True]
                    cur_score = cur_score[cur_match == True]

                    for pred in pred_instances:
                        found_gt = False
                        for gt in pred["matched_gt"]:
                            overlap = float(gt["intersection"]) / (
                                gt["vert_count"] + pred["vert_count"] - gt["intersection"]
                            )
                            if overlap > overlap_th:
                                found_gt = True
                                break
                        if not found_gt:
                            num_ignore = pred["void_intersection"]
                            for gt in pred["matched_gt"]:
                                # group? (kept from ScanNet scripts; safe if class ids are 1-based)
                                if gt["instance_id"] < 1000:
                                    num_ignore += gt["intersection"]
                                if (
                                    gt["vert_count"] < min_region_size
                                    or gt["med_dist"] > distance_thresh
                                    or gt["dist_conf"] < distance_conf
                                ):
                                    num_ignore += gt["intersection"]
                            proportion_ignore = float(num_ignore) / pred["vert_count"]
                            if proportion_ignore <= overlap_th:
                                cur_true = np.append(cur_true, 0)
                                cur_score = np.append(cur_score, pred["confidence"])

                    y_true = np.append(y_true, cur_true)
                    y_score = np.append(y_score, cur_score)

                if has_gt and has_pred:
                    score_arg_sort = np.argsort(y_score)
                    y_score_sorted = y_score[score_arg_sort]
                    y_true_sorted = y_true[score_arg_sort]
                    y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                    thresholds, unique_indices = np.unique(y_score_sorted, return_index=True)
                    num_prec_recall = len(unique_indices) + 1

                    num_examples = len(y_score_sorted)
                    num_true_examples = y_true_sorted_cumsum[-1] if len(y_true_sorted_cumsum) > 0 else 0
                    precision = np.zeros(num_prec_recall)
                    recall = np.zeros(num_prec_recall)

                    y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                    for idx_res, idx_scores in enumerate(unique_indices):
                        cumsum = y_true_sorted_cumsum[idx_scores - 1]
                        tp = num_true_examples - cumsum
                        fp = num_examples - idx_scores - tp
                        fn = cumsum + hard_false_negatives
                        precision[idx_res] = float(tp) / (tp + fp)
                        recall[idx_res] = float(tp) / (tp + fn)

                    rc_current = recall[0]
                    pr_current = precision[0]

                    precision[-1] = 1.0
                    recall[-1] = 0.0

                    fig = plt.figure(figsize=(15, 5))
                    plt.subplot(1, 3, 1)
                    plt.plot(recall, precision)
                    plt.plot(recall, precision, "r*")
                    plt.grid()
                    plt.xlabel("Recall")
                    plt.xlim((0.0, 1.0))
                    plt.ylabel("Precision")
                    plt.ylim((0.0, 1.0))
                    plt.title(f"PR di={di} iou={overlap_th:.3f} {label_name}")

                    plt.subplot(1, 3, 2)
                    plt.plot(thresholds, precision[:-1])
                    plt.plot(thresholds, precision[:-1], "r*")
                    plt.grid()
                    plt.xlabel("conf TH")
                    plt.xlim((0.0, 1.0))
                    plt.ylabel("Precision")
                    plt.ylim((0.0, 1.0))
                    plt.title(f"P-TH di={di} iou={overlap_th:.3f} {label_name}")

                    plt.subplot(1, 3, 3)
                    plt.plot(thresholds, recall[:-1])
                    plt.plot(thresholds, recall[:-1], "r*")
                    plt.grid()
                    plt.xlabel("conf TH")
                    plt.xlim((0.0, 1.0))
                    plt.ylabel("Recall")
                    plt.ylim((0.0, 1.0))
                    plt.title(f"R-TH di={di} iou={overlap_th:.3f} {label_name}")

                    if prcurv_save_dir is not None:
                        os.makedirs(prcurv_save_dir, exist_ok=True)
                        plt.savefig(osp.join(prcurv_save_dir, f"{di}_iou={overlap_th:.3f}_{label_name}.png"))
                    plt.close()

                    recall_for_conv = np.copy(recall)
                    recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                    recall_for_conv = np.append(recall_for_conv, 0.0)
                    stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], "valid")
                    ap_current = np.dot(precision, stepWidths)
                elif has_gt:
                    ap_current = 0.0
                    rc_current = 0.0
                    pr_current = 0.0
                else:
                    ap_current = float("nan")
                    rc_current = float("nan")
                    pr_current = float("nan")

                ap[di, li, oi] = ap_current
                rc[di, li, oi] = rc_current
                pr[di, li, oi] = pr_current

    return ap, rc, pr


def compute_averages(aps, rcs, prs):
    d_inf = 0
    o50 = np.where(np.isclose(opt["overlaps"], 0.5))
    o25 = np.where(np.isclose(opt["overlaps"], 0.25))
    oAllBut25 = np.where(np.logical_not(np.isclose(opt["overlaps"], 0.25)))
    avg_dict = {}
    avg_dict["all_ap"] = np.nanmean(aps[d_inf, :, oAllBut25])
    avg_dict["all_ap_50%"] = np.nanmean(aps[d_inf, :, o50])
    avg_dict["all_ap_25%"] = np.nanmean(aps[d_inf, :, o25])
    avg_dict["all_rc"] = np.nanmean(rcs[d_inf, :, oAllBut25])
    avg_dict["all_rc_50%"] = np.nanmean(rcs[d_inf, :, o50])
    avg_dict["all_rc_25%"] = np.nanmean(rcs[d_inf, :, o25])
    avg_dict["all_pr"] = np.nanmean(prs[d_inf, :, oAllBut25])
    avg_dict["all_pr_50%"] = np.nanmean(prs[d_inf, :, o50])
    avg_dict["all_pr_25%"] = np.nanmean(prs[d_inf, :, o25])
    avg_dict["classes"] = {}
    for li, label_name in enumerate(eval_class_labels):
        avg_dict["classes"][label_name] = {}
        avg_dict["classes"][label_name]["ap"] = np.average(aps[d_inf, li, oAllBut25])
        avg_dict["classes"][label_name]["ap50%"] = np.average(aps[d_inf, li, o50])
        avg_dict["classes"][label_name]["ap25%"] = np.average(aps[d_inf, li, o25])
        avg_dict["classes"][label_name]["rc"] = np.average(rcs[d_inf, li, oAllBut25])
        avg_dict["classes"][label_name]["rc50%"] = np.average(rcs[d_inf, li, o50])
        avg_dict["classes"][label_name]["rc25%"] = np.average(rcs[d_inf, li, o25])
        avg_dict["classes"][label_name]["pr"] = np.average(prs[d_inf, li, oAllBut25])
        avg_dict["classes"][label_name]["pr50%"] = np.average(prs[d_inf, li, o50])
        avg_dict["classes"][label_name]["pr25%"] = np.average(prs[d_inf, li, o25])
    return avg_dict


def print_results(avgs):
    sep = ""
    col1 = ":"
    lineLen = 100

    print("")
    print("#" * lineLen)
    line = ""
    line += "{:<15}".format("what") + sep + col1
    line += "{:>8}".format("AP") + sep
    line += "{:>8}".format("AP_50%") + sep
    line += "{:>8}".format("AP_25%") + sep
    line += "{:>2}".format("|") + sep
    line += "{:>8}".format("RC") + sep
    line += "{:>8}".format("RC_50%") + sep
    line += "{:>8}".format("RC_25%") + sep
    line += "{:>2}".format("|") + sep
    line += "{:>8}".format("PR") + sep
    line += "{:>8}".format("PR_50%") + sep
    line += "{:>8}".format("PR_25%") + sep
    print(line)
    print("#" * lineLen)

    for label_name in eval_class_labels:
        ap_avg = avgs["classes"][label_name]["ap"]
        ap_50o = avgs["classes"][label_name]["ap50%"]
        ap_25o = avgs["classes"][label_name]["ap25%"]
        rc_avg = avgs["classes"][label_name]["rc"]
        rc_50o = avgs["classes"][label_name]["rc50%"]
        rc_25o = avgs["classes"][label_name]["rc25%"]
        pr_avg = avgs["classes"][label_name]["pr"]
        pr_50o = avgs["classes"][label_name]["pr50%"]
        pr_25o = avgs["classes"][label_name]["pr25%"]
        line = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>8.3f}".format(ap_avg) + sep
        line += sep + "{:>8.3f}".format(ap_50o) + sep
        line += sep + "{:>8.3f}".format(ap_25o) + sep
        line += "{:>2}".format("|") + sep
        line += sep + "{:>8.3f}".format(rc_avg) + sep
        line += sep + "{:>8.3f}".format(rc_50o) + sep
        line += sep + "{:>8.3f}".format(rc_25o) + sep
        line += "{:>2}".format("|") + sep
        line += sep + "{:>8.3f}".format(pr_avg) + sep
        line += sep + "{:>8.3f}".format(pr_50o) + sep
        line += sep + "{:>8.3f}".format(pr_25o) + sep
        print(line)

    all_ap_avg = avgs["all_ap"]
    all_ap_50o = avgs["all_ap_50%"]
    all_ap_25o = avgs["all_ap_25%"]
    all_rc_avg = avgs["all_rc"]
    all_rc_50o = avgs["all_rc_50%"]
    all_rc_25o = avgs["all_rc_25%"]
    all_pr_avg = avgs["all_pr"]
    all_pr_50o = avgs["all_pr_50%"]
    all_pr_25o = avgs["all_pr_25%"]

    print("-" * lineLen)
    line = "{:<15}".format("average") + sep + col1
    line += "{:>8.3f}".format(all_ap_avg) + sep
    line += "{:>8.3f}".format(all_ap_50o) + sep
    line += "{:>8.3f}".format(all_ap_25o) + sep
    line += "{:>2}".format("|") + sep
    line += "{:>8.3f}".format(all_rc_avg) + sep
    line += "{:>8.3f}".format(all_rc_50o) + sep
    line += "{:>8.3f}".format(all_rc_25o) + sep
    line += "{:>2}".format("|") + sep
    line += "{:>8.3f}".format(all_pr_avg) + sep
    line += "{:>8.3f}".format(all_pr_50o) + sep
    line += "{:>8.3f}".format(all_pr_25o) + sep
    print(line)
    print("")


def make_pred_info(pred: dict):
    pred_info = {}
    assert pred["pred_classes"].shape[0] == pred["pred_scores"].shape[0] == pred["pred_masks"].shape[1]
    for i in range(len(pred["pred_classes"])):
        info = {}
        info["label_id"] = int(pred["pred_classes"][i])
        info["conf"] = float(pred["pred_scores"][i])
        info["mask"] = pred["pred_masks"][:, i]
        pred_info[uuid4()] = info
    return pred_info


def assign_instances_for_scan(use_label: bool, pred: dict, gt_ids: np.ndarray):
    pred_info = make_pred_info(pred)
    gt_ids = np.asarray(gt_ids, dtype=np.int64)

    gt_instances = util_3d.get_instances(gt_ids, VALID_CLASS_IDS, CLASS_LABELS, ID_TO_LABEL)

    if use_label:
        gt2pred = deepcopy(gt_instances)
        for label in gt2pred:
            for gt in gt2pred[label]:
                gt["matched_pred"] = []
    else:
        gt2pred = {}
        agnostic_instances = []
        for _, instances in gt_instances.items():
            agnostic_instances += deepcopy(instances)
        for gt in agnostic_instances:
            gt["matched_pred"] = []
        gt2pred[eval_class_labels[0]] = agnostic_instances

    pred2gt = {label: [] for label in eval_class_labels}
    num_pred_instances = 0

    bool_void = np.logical_not(np.in1d(gt_ids // 1000, VALID_CLASS_IDS))

    for uuid in pred_info:
        if use_label:
            label_id = int(pred_info[uuid]["label_id"])
            if label_id not in ID_TO_LABEL:
                continue
            label_name = ID_TO_LABEL[label_id]
        else:
            label_id = None
            label_name = eval_class_labels[0]

        conf = float(pred_info[uuid]["conf"])
        pred_mask = pred_info[uuid]["mask"]
        assert len(pred_mask) == len(gt_ids)
        pred_mask = np.not_equal(pred_mask, 0)
        num = np.count_nonzero(pred_mask)
        if num < opt["min_region_sizes"][0]:
            continue

        pred_instance = {
            "uuid": uuid,
            "pred_id": num_pred_instances,
            "label_id": label_id if use_label else None,
            "vert_count": num,
            "confidence": conf,
            "void_intersection": np.count_nonzero(np.logical_and(bool_void, pred_mask)),
        }

        matched_gt = []
        for gt_num, gt_inst in enumerate(gt2pred[label_name]):
            intersection = np.count_nonzero(np.logical_and(gt_ids == gt_inst["instance_id"], pred_mask))
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy["intersection"] = intersection
                pred_copy["intersection"] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_num]["matched_pred"].append(pred_copy)

        pred_instance["matched_gt"] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)

    return gt2pred, pred2gt


def evaluate(
    use_label: bool,
    preds: dict,
    gt: dict,
    class_names=None,
    valid_class_ids=None,
    logger=None,
    log: bool = False,
    prcurv_save_dir: str = None,
):
    """
    If valid_class_ids is None, infer from union of gt_ids//1000 and pred_classes (when use_label=True).
    """
    global CLASS_LABELS, VALID_CLASS_IDS, ID_TO_LABEL, LABEL_TO_ID, eval_class_labels

    if valid_class_ids is None:
        class_ids = set()
        for scene, gt_ids in gt.items():
            gt_ids = np.asarray(gt_ids, dtype=np.int64)
            sem = gt_ids // 1000
            for cid in np.unique(sem):
                if cid <= 0:
                    continue
                class_ids.add(int(cid))
        if use_label:
            for scene, pred in preds.items():
                for cid in np.unique(np.asarray(pred["pred_classes"], dtype=np.int64)):
                    if cid > 0:
                        class_ids.add(int(cid))
        valid_class_ids = np.array(sorted(class_ids), dtype=np.int64)

    CLASS_LABELS, VALID_CLASS_IDS, ID_TO_LABEL, LABEL_TO_ID = _build_label_maps(class_names, valid_class_ids)

    if not use_label:
        eval_class_labels = ["class_agnostic"]
    else:
        eval_class_labels = CLASS_LABELS

    print("evaluating", len(preds), "scans...")
    matches = {}
    for i, (k, v) in enumerate(preds.items()):
        matches_key = k
        gt2pred, pred2gt = assign_instances_for_scan(use_label, v, gt[k])
        matches[matches_key] = {"gt": gt2pred, "pred": pred2gt}
        sys.stdout.write("\rscans processed: {}".format(i + 1))
        sys.stdout.flush()
    print("")
    ap_scores, recall_scores, precision_scores = evaluate_matches(matches, prcurv_save_dir)
    avgs = compute_averages(ap_scores, recall_scores, precision_scores)
    print_results(avgs)
    if (logger is not None) and log:
        # keep same logging contract as other scripts: just print to logger if provided
        logger.info(str(avgs))

