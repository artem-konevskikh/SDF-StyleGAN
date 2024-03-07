import os
import glob
import logging
from typing import List

from tqdm import tqdm
import numpy as np
import trimesh
import mesh2sdf


logger = logging.getLogger("trimesh")
logger.setLevel(logging.ERROR)

size = 128  # resolution of SDF
level = 2 / size  # 2/128 = 0.015625
shape_scale = 0.5  # rescale the shape into [-0.5, 0.5]


def get_filenames(dir: str, ext: str = "obj") -> List[str]:
    return sorted(glob.glob(f"{dir}/*.{ext}"))


def convert_mesh2sdf(filenames: List[str], dataset_dir: str) -> None:
    mesh_scale = 0.8
    for filename in tqdm(filenames, ncols=80):
        model_name = filename.split("/")[-1].lower()
        model_name = model_name.replace(" ", "_")
        model_name = model_name.replace(".", "_")
        model_name = model_name.replace("(", "").replace(")", "")
        filename_box = f"{dataset_dir}/box_{model_name}.npz"
        filename_npy = f"{dataset_dir}/sdf_{model_name}.npy"
        filename_obj = f"{dataset_dir}/mesh_{model_name}.obj"
        if os.path.exists(filename_box):
            continue
        # load the raw mesh
        try:
            mesh = trimesh.load(filename, force="mesh")
        except Exception as e:
            print(e, filename)
            continue
        # rescale mesh to [-1, 1] for mesh2sdf, note the factor **mesh_scale**
        vertices = mesh.vertices
        bbmin, bbmax = vertices.min(0), vertices.max(0)
        center = (bbmin + bbmax) * 0.5
        scale = 2.0 * mesh_scale / (bbmax - bbmin).max()
        vertices = (vertices - center) * scale

        # run mesh2sdf
        sdf, mesh_new = mesh2sdf.compute(vertices, mesh.faces, size, fix=True,
                                         level=level, return_mesh=True)
        # mesh_new.vertices = mesh_new.vertices * shape_scale

        # save
        np.savez(filename_box, bbmax=bbmax, bbmin=bbmin, mul=mesh_scale)
        np.save(filename_npy, sdf)
        mesh_new.export(filename_obj)


def main() -> None:
    mesh_dir = "skulls-raw"
    dataset_dir = "skulls-sdf"
    filenames = []
    filenames += get_filenames(mesh_dir, "obj")
    filenames += get_filenames(mesh_dir, "stl")
    convert_mesh2sdf(filenames[:3], dataset_dir)


if __name__ == "__main__":
    main()
