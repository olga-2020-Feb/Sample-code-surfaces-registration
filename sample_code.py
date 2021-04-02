import os
import re

import numpy as np
import torch

from models.smpl4garment import SMPL4Garment
from psbody.mesh import Mesh


input_obj_file_path = os.path.join('input', 'Top.obj')
input_info_file_path = os.path.join('input', 'info.npz')

output_old_file_path = os.path.join('output', 'old_top.obj')
output_new_file_path = os.path.join('output', 'new_top.obj')
output_smpl_cut_file_path = os.path.join('output', 'smpl_cut.obj')


def get_vert_indices_from_joints_smpl4(garment_joints, smpl4, threshold = 0.5):
    joints_mask = np.zeros(24)
    joints_mask[garment_joints] = 1
    related_weights4 = np.dot(smpl4.smpl_base.weights, joints_mask);
    indices4 = np.where(related_weights4 > threshold)
    return indices4[0]

def get_faces_from_indices(indices, smpl4):

    strange_mesh = Mesh(v=smpl4.smpl_base.r, f=smpl4.smpl_base.f)
    strange_mesh_cut = strange_mesh.keep_vertices(indices)

    return strange_mesh_cut.f

def write_obj(file_path, vertices, faces):
    with open(file_path, 'w') as fp:
        vertices_string_list = [str('v ' + (np.array2string(np.array(vertex)).strip('[]'))) for vertex in vertices]
        vertices_string = np.array2string(np.array(vertices_string_list),
                                           threshold=(len(vertices) + 1),
                                           suppress_small=True,
                                           separator='\n')
        vertices_string = re.sub(' +', ' ', vertices_string).replace('\'', '')
        vertices_string = vertices_string.replace('[', '').replace(']', '').replace('\n ', '\n').replace('\n ', '\n')
        fp.write(vertices_string)
        fp.write('\n')
        faces_string_list = [str('f ' + (np.array2string(np.array(face) + 1).strip('[]'))) for face in faces]
        faces_string = np.array2string(np.array(faces_string_list),
                                       threshold=(len(faces) + 1),
                                       separator='\n')
        faces_string = re.sub(' +', ' ', faces_string).replace('\'', '')
        faces_string = faces_string.replace('[', '').replace(']', '').replace('\n ', '\n').replace('\n ', '\n')
        fp.write(faces_string)
        fp.write('\n')

def read_obj(file):
	vertex_list, faces = [], []

	with open(file, 'r') as f:
		lines = f.readlines()
	for line in lines:
		if line.startswith('v '):
			vertex = [float(n) for n in line.replace('v ','').split(' ')]
			vertex_list += [vertex]
		elif line.startswith('f '):
			idx = [n.split('/') for n in line.replace('f ','').split(' ')]
			face = [int(n[0]) - 1 for n in idx]
			faces += [face]

	vertices = np.array(vertex_list)

	return vertices, faces


def pair_wise_squared_dist_matrix(points1_tensor, points2_tensor):
    N1 = points1_tensor.shape[0];
    N2 = points2_tensor.shape[0];
    D = 3;

    dot_prod = np.matmul(points1_tensor, points2_tensor.T);
    x1_tensor_square = points1_tensor * points1_tensor;
    x2_tensor_square = points2_tensor * points2_tensor;
    x1_tensor_square_sum = np.sum(x1_tensor_square, 1);
    x2_tensor_square_sum = np.sum(x2_tensor_square, 1);

    x1_tensor_square_sum_matrix = np.tile(np.array([x1_tensor_square_sum]), (N2, 1))
    x2_tensor_square_sum_matrix = np.tile(np.array([x2_tensor_square_sum]).T, (1, N1))

    square_dists = (x1_tensor_square_sum_matrix + x2_tensor_square_sum_matrix - 2 * dot_prod.T).T;

    return square_dists;

def bad_surface_registration(smpl_vertices, smpl_faces, garment_vertices, garment_faces):
    dist_matrix = pair_wise_squared_dist_matrix(smpl_vertices, garment_vertices)
    min_indices = np.argmin(dist_matrix, axis=1)
    new_smpl_vertices = garment_vertices[min_indices]
    return new_smpl_vertices

smpl_joints = np.array([0, 3, 6, 9, 13, 14])

smpl4 = SMPL4Garment(gender='female')

indices4 = get_vert_indices_from_joints_smpl4(smpl_joints, smpl4)

smpl_faces = get_faces_from_indices(indices4, smpl4)
smpl_vertices = np.array(smpl4.smpl_base.v[indices4])

info = np.load(input_info_file_path)
garment_vertices, garment_faces = read_obj(input_obj_file_path)

#TODO: replace with good surface registration
new_smpl_vertices = bad_surface_registration(smpl_vertices, smpl_faces, garment_vertices, garment_faces)

write_obj(output_old_file_path, garment_vertices, garment_faces)
write_obj(output_smpl_cut_file_path, smpl_vertices, smpl_faces)
write_obj(output_new_file_path, new_smpl_vertices, smpl_faces)



