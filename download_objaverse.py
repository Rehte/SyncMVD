import os
import glob
import gzip
import json
import multiprocessing
import os
import pandas as pd
import yaml

import urllib.request
import warnings
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import trimesh


BASE_PATH = os.path.join("./assets/objaverse")
BASE_PATH = os.path.join("./objaverse")
os.makedirs(BASE_PATH, exist_ok=True)

__version__ = "<REPLACE_WITH_VERSION>"
_VERSIONED_PATH = os.path.join(BASE_PATH, "hf-objaverse-v1")
os.makedirs(_VERSIONED_PATH, exist_ok=True)



def glb2obj(glb_path, obj_path):
    mesh = trimesh.load(glb_path)
    
    if isinstance(mesh, trimesh.Scene):
        vertices = 0
        for g in mesh.geometry.values():
            vertices += g.vertices.shape[0]
    elif isinstance(mesh, trimesh.Trimesh):
        vertices = mesh.vertices.shape[0]
    else:
        raise ValueError(f'{glb_path} is not mesh or scene')
    
    # if vertices > 100000:
    #     print(f'Too many vertices in {glb_path}. Skip this mesh')
    #     del mesh, vertices
    #     return 0
    if not os.path.exists(os.path.dirname(obj_path)):
        os.makedirs(os.path.dirname(obj_path))
    mesh.export(obj_path)
    
    del mesh, vertices
    return 1



def load_annotations(uids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Load the full metadata of all objects in the dataset.

    Args:
        uids: A list of uids with which to load metadata. If None, it loads
        the metadata for all uids.
    """
    metadata_path = os.path.join(_VERSIONED_PATH, "metadata")
    object_paths = _load_object_paths()
    dir_ids = (
        set([object_paths[uid].split("/")[1] for uid in uids])
        if uids is not None
        else [f"{i // 1000:03d}-{i % 1000:03d}" for i in range(160)]
    )
    if len(dir_ids) > 10:
        dir_ids = tqdm(dir_ids)
    out = {}
    for i_id in dir_ids:
        json_file = f"{i_id}.json.gz"
        local_path = os.path.join(metadata_path, json_file)
        if not os.path.exists(local_path):
            hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/metadata/{i_id}.json.gz"
            # wget the file and put it in local_path
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            urllib.request.urlretrieve(hf_url, local_path)
        with gzip.open(local_path, "rb") as f:
            data = json.load(f)
        if uids is not None:
            data = {uid: data[uid] for uid in uids if uid in data}
        out.update(data)
        if uids is not None and len(out) == len(uids):
            break
    return out


def _load_object_paths() -> Dict[str, str]:
    """Load the object paths from the dataset.

    The object paths specify the location of where the object is located
    in the Hugging Face repo.

    Returns:
        A dictionary mapping the uid to the object path.
    """
    object_paths_file = "object-paths.json.gz"
    local_path = os.path.join(_VERSIONED_PATH, object_paths_file)
    if not os.path.exists(local_path):
        hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_paths_file}"
        # wget the file and put it in local_path
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        urllib.request.urlretrieve(hf_url, local_path)
    with gzip.open(local_path, "rb") as f:
        object_paths = json.load(f)
    return object_paths


def load_uids() -> List[str]:
    """Load the uids from the dataset.

    Returns:
        A list of uids.
    """
    return list(_load_object_paths().keys())


def _download_object(
    uid: str,
    object_path: str,
    total_downloads: float,
    start_file_count: int,
) -> Tuple[str, str]:
    """Download the object for the given uid.

    Args:
        uid: The uid of the object to load.
        object_path: The path to the object in the Hugging Face repo.

    Returns:
        The local path of where the object was downloaded.
    """
    # print(f"downloading {uid}")
    local_path = os.path.join(_VERSIONED_PATH, object_path)
    tmp_local_path = os.path.join(_VERSIONED_PATH, object_path + ".tmp")
    hf_url = (
        f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_path}"
    )
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(hf_url, tmp_local_path)
    
    if not os.path.exists(tmp_local_path):
        print(f"Mesh Download Failed: {uid}")
        return uid, None

    os.rename(tmp_local_path, local_path)

    files = glob.glob(os.path.join(_VERSIONED_PATH, "glbs", "*", "*.glb"))
    print(
        "Downloaded",
        len(files) - start_file_count,
        "/",
        total_downloads,
        "objects",
    )

    return uid, local_path


def load_objects(uids: List[str], download_processes: int = 1) -> Dict[str, str]:
    """Return the path to the object files for the given uids.

    If the object is not already downloaded, it will be downloaded.

    Args:
        uids: A list of uids.
        download_processes: The number of processes to use to download the objects.

    Returns:
        A dictionary mapping the object uid to the local path of where the object
        downloaded.
    """
    object_paths = _load_object_paths()
    out = {}
    if download_processes == 1:
        uids_to_download = []
        for uid in uids:
            if uid.endswith(".glb"):
                uid = uid[:-4]
            if uid not in object_paths:
                warnings.warn("Could not find object with uid. Skipping it.")
                continue
            object_path = object_paths[uid]
            local_path = os.path.join(_VERSIONED_PATH, object_path)
            if os.path.exists(local_path):
                out[uid] = local_path
                continue
            uids_to_download.append((uid, object_path))
        if len(uids_to_download) == 0:
            return out
        start_file_count = len(
            glob.glob(os.path.join(_VERSIONED_PATH, "glbs", "*", "*.glb"))
        )
        for uid, object_path in uids_to_download:
            uid, local_path = _download_object(
                uid, object_path, len(uids_to_download), start_file_count
            )
            if local_path is not None:
                out[uid] = local_path
    else:
        args = []
        for uid in uids:
            if uid.endswith(".glb"):
                uid = uid[:-4]
            if uid not in object_paths:
                warnings.warn(f"Could not find object with uid. Skipping it.: {uid}")
                continue
            object_path = object_paths[uid]
            local_path = os.path.join(_VERSIONED_PATH, object_path)
            if not os.path.exists(local_path):
                args.append((uid, object_paths[uid]))
            else:
                out[uid] = local_path
        if len(args) == 0:
            return out
        print(
            f"starting download of {len(args)} objects with {download_processes} processes"
        )
        start_file_count = len(
            glob.glob(os.path.join(_VERSIONED_PATH, "glbs", "*", "*.glb"))
        )
        args = [(*arg, len(args), start_file_count) for arg in args]
        with multiprocessing.Pool(download_processes) as pool:
            r = pool.starmap(_download_object, args)
            for uid, local_path in r:
                if local_path is not None:
                    out[uid] = local_path
    return out


def load_lvis_annotations() -> Dict[str, List[str]]:
    """Load the LVIS annotations.

    If the annotations are not already downloaded, they will be downloaded.

    Returns:
        A dictionary mapping the LVIS category to the list of uids in that category.
    """
    hf_url = f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/lvis-annotations.json.gz"
    local_path = os.path.join(_VERSIONED_PATH, "lvis-annotations.json.gz")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        urllib.request.urlretrieve(hf_url, local_path)
    with gzip.open(local_path, "rb") as f:
        lvis_annotations = json.load(f)
    return lvis_annotations



if __name__ == '__main__':    
    # key: uid, value: mesh_name
    df = pd.read_csv('Objects.csv')
    uid_to_name = dict(zip(df['uid'], df['name']))
    uid_list = df['uid'].tolist()
    uid_to_description = dict(zip(df['uid'], df['description']))

    obj_dict = load_objects(uid_list, download_processes=10)
    
    config_data = {
        'mesh': "./model.obj",
        'mesh_config_relative': True,
        'use_mesh_name': False,
        'prompt': f"Photo of ",  
        'steps': 30,
        'cond_type': "depth",
        'seed': 2,
        'log_interval': 10,
        'mesh_scale': 1,
        'tex_fast_preview': True,
        'view_fast_preview': True
    }
    
    mesh_path_list = glob.glob(os.path.join(_VERSIONED_PATH, "glbs", "*", "*.glb"))
    convert_success = 0
    for mesh_path in tqdm(mesh_path_list, desc="Converting glb to obj"):
        uid = mesh_path.split("/")[-1].split(".")[0]
        mesh_name = uid_to_name.get(uid, 'UID not found').replace(' ', '_')
        
        description = uid_to_description.get(uid, 'Description not found')
        config_data['prompt'] = f"Photo of a {description}"
        config_data['mesh_name'] = mesh_name
        
        mesh_name = uid # Set the directory name as uid
        os.makedirs(f"{BASE_PATH}/{mesh_name}", exist_ok=True)
        config_path = f"{BASE_PATH}/{mesh_name}/config.yaml"
        with open(config_path, 'w') as yaml_file:
            yaml.dump(config_data, yaml_file, default_flow_style=False)
        os.system(f"mv {mesh_path} {BASE_PATH}/{mesh_name}/model.glb")
        print(mesh_name)
        convert_success += glb2obj(f"{BASE_PATH}/{mesh_name}/model.glb", f"{BASE_PATH}/{mesh_name}/model.obj")
    
    os.system(f"rm -r {_VERSIONED_PATH}")
    print(f"{len(obj_dict)} meshes downloaded at {BASE_PATH}")
    print(f"{convert_success}/{len(obj_dict)} meshes converted to obj")