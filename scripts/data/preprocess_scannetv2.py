"""
This module preprocesses the raw scannet dataset into ply files, that can be used
for further modules.
Reference: https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py
"""
#%%
import json
import os
from functools import partial

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData
from tqdm.contrib.concurrent import process_map

from vl3d.data.utils import extract_scene_ids
from vl3d.utils import cfg

#%%


def get_raw2scannetv2_label_map():
    lines = [line.rstrip() for line in open(cfg.scannetv2.labels_combined_path)]
    lines = lines[1:]
    raw2scannet = {}
    for i in range(len(lines)):
        label_classes_set = set(cfg.scannetv2.softgroup_names)
        elements = lines[i].split("\t")
        raw_name = elements[1]
        nyu40_name = elements[7]
        if nyu40_name not in label_classes_set:
            raw2scannet[raw_name] = "unannotated"
        else:
            raw2scannet[raw_name] = nyu40_name
    return raw2scannet


def read_mesh_file(mesh_file):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.compute_vertex_normals()
    return (
        np.asarray(mesh.vertices, dtype=np.float32),
        np.rint(np.asarray(mesh.vertex_colors) * 255).astype(np.uint8),
        np.asarray(mesh.vertex_normals, dtype=np.float32),
    )


#%%
def read_label_file(label_file, mapping_classes_ids, ignore_label):
    plydata = PlyData.read(label_file)
    nyu_40_sem_labels = np.array(plydata["vertex"]["label"], dtype=np.int32)  # nyu40
    sem_labels = np.full(shape=nyu_40_sem_labels.shape, fill_value=ignore_label, dtype=np.int32)
    for index, id in enumerate(mapping_classes_ids):
        sem_labels[nyu_40_sem_labels == id] = index
    return sem_labels


#%%


def read_agg_file(agg_file, label_map):
    object_id2segs = {}
    label2segs = {}
    # object_id = 0
    with open(agg_file, "r") as json_data:
        data = json.load(json_data)
        for group in data["segGroups"]:
            object_id = group["objectId"]
            label = group["label"]
            segs = group["segments"]
            # if label_map[label] not in ["wall", "floor", "ceiling"]:
            if label not in ["wall", "floor", "ceiling"]:
                object_id2segs[object_id] = segs
                # object_id += 1
                if label in label2segs:
                    label2segs[label].extend(segs)
                else:
                    label2segs[label] = segs.copy()
    return object_id2segs, label2segs


#%%
def read_seg_file(seg_file):
    seg2verts = {}
    with open(seg_file, "r") as json_data:
        data = json.load(json_data)
        num_verts = len(data["segIndices"])
        for vert, seg in enumerate(data["segIndices"]):
            if seg in seg2verts:
                seg2verts[seg].append(vert)
            else:
                seg2verts[seg] = [vert]
    return seg2verts, num_verts


#%%


def get_instance_ids(objectId2segs, seg2verts, sem_labels, ignore_label):
    object_id2label_id = {}
    instance_ids = np.full(shape=len(sem_labels), fill_value=ignore_label, dtype=np.int32)
    for objectId, segs in objectId2segs.items():
        for seg in segs:
            verts = seg2verts[seg]
            instance_ids[verts] = objectId
        if objectId not in object_id2label_id:
            object_id2label_id[objectId] = sem_labels[verts][0]
    return instance_ids, object_id2label_id


def export(scene, label_map):
    mesh_file_path = os.path.join(cfg.scannetv2.raw_scans_path, scene, scene + "_vh_clean_2.ply")
    label_file_path = os.path.join(cfg.scannetv2.raw_scans_path, scene, scene + "_vh_clean_2.labels.ply")
    agg_file_path = os.path.join(cfg.scannetv2.raw_scans_path, scene, scene + ".aggregation.json")
    seg_file_path = os.path.join(cfg.scannetv2.raw_scans_path, scene, scene + "_vh_clean_2.0.010000.segs.json")

    # read mesh_file
    xyz, rgb, normal = read_mesh_file(mesh_file_path)
    num_verts = len(xyz)

    if os.path.exists(agg_file_path):
        # read label_file
        sem_labels = read_label_file(
            label_file_path,
            cfg.scannetv2.mapping_classes_ids,
            cfg.scannetv2.ignore_label,
        )
        # read seg_file
        seg2verts, num = read_seg_file(seg_file_path)
        assert num_verts == num
        # read agg_file
        object_id2segs, label2segs = read_agg_file(agg_file_path, label_map)
        # get instance labels
        instance_ids, object_id2label_id = get_instance_ids(
            object_id2segs, seg2verts, sem_labels, cfg.scannetv2.ignore_label
        )
    else:
        # use zero as placeholders for the test scene
        sem_labels = np.full(
            shape=num_verts,
            fill_value=cfg.scannetv2.ignore_label,
            dtype=np.int32,
        )  # 0: unannotated
        instance_ids = np.full(
            shape=num_verts,
            fill_value=cfg.scannetv2.ignore_label,
            dtype=np.int32,
        )
    return xyz, rgb, normal, sem_labels, instance_ids


def process_one_scan(scan, split, label_map):
    xyz, rgb, normal, sem_labels, instance_ids = export(scan, label_map)
    torch.save(
        {
            "xyz": xyz,
            "rgb": rgb,
            "normal": normal,
            "sem_labels": sem_labels,
            "instance_ids": instance_ids,
        },
        os.path.join(cfg.scannetv2.path, split, f"{scan}.pth"),
    )


def main():
    os.makedirs(os.path.join(cfg.scannetv2.path, "train"), exist_ok=True)
    os.makedirs(os.path.join(cfg.scannetv2.path, "val"), exist_ok=True)

    with open(cfg.scanrefer.train_split, "r") as f:
        train_list = extract_scene_ids(json.load(f))

    with open(cfg.scanrefer.val_split) as f:
        val_list = extract_scene_ids(json.load(f))

    label_map = get_raw2scannetv2_label_map()

    print("==> Processing train split ...")
    process_map(
        partial(process_one_scan, split="train", label_map=label_map),
        train_list,
        chunksize=1,
    )
    print("==> Processing val split ...")
    process_map(
        partial(process_one_scan, split="val", label_map=label_map),
        val_list,
        chunksize=1,
    )


if __name__ == "__main__":
    main()
