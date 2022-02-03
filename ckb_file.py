import math
from typing import List

import numpy as np
from pathlib import Path

from .byte_io_ckb import ByteIO, split
from .ckb_reader import CKBReader


class HeroGeomerty:
    def __init__(self):
        self.index = []
        self.positions = []
        self.normals = []
        self.uv = []
        self.uv2 = []
        self.vertex_colors = {}
        self.shape_key_data = {}
        self.skin_indices = np.array([])  # type:np.ndarray
        self.skin_weights = np.array([])  # type:np.ndarray
        self.original_indices = []
        self.main_skeleton = False
        self.has_geometry = False
        self.skinned = False
        self.bounds = []
        self.scale = []
        self.offset = []
        self.bones = []  # type: List[HeroBone]
        self.poses = {}
        self.locations = {}


class HeroBone:

    def __init__(self):
        self.name = ''
        self.parent_id = -1
        self.bone_id = -1
        self.real_bone_id = -1
        self.pos = []
        self.scale = []
        self.quat = []

    def convert_name(self):
        bone_name = self.name
        if not bone_name.endswith('_bind_jnt'):
            return

        tmp = bone_name.split('_')
        if not tmp[-3].isnumeric():
            return
        else:
            bone_id = int(tmp[-3])
            if bone_id >= 500:
                bone_id += 4500
            tmp[-3] = f"{bone_id:04}"
            self.name = '_'.join(tmp)
            self.real_bone_id = bone_id
            # assert self.real_bone_id == self.bone_id

    def __repr__(self) -> str:
        return f'<HeroBone {self.name} parent_id={self.parent_id}>'


class CKBFile:
    me = (2 ** 8) - 1
    ge = (2 ** 16) - 1
    H = math.pow(2, 16) - 1
    X = (math.pow(2, 16) - 2) / 2

    def __init__(self, data, name):
        self.reader = CKBReader(ByteIO(data))
        self.version = self.reader.version
        self.name = name
        self.options = {}
        self.geometry = HeroGeomerty()
        self.vertex_count = 0

    @property
    def i32_indices(self):
        return self.options.get('indices32bit', False)

    @property
    def after_1_7(self):
        return self.version > 1.75

    @property
    def before_1_7(self):
        return self.version < 1.7

    def read(self):
        self._init_settings()
        self._init_indices()
        self._init_points()
        self._init_normals()
        self._init_uvs()
        self._init_vertex_colors()
        self._init_blends()
        self._init_weights()
        self._init_parent()
        self._init_poses()

    def _init_settings(self):
        default_attributes = ["mesh", "normals", "uv1", "uv2", "blendTargets", "blendNormals", "weights", "animations",
                              "jointScales", "addon", "paintMapping", "singleParent", "frameMappings", "indices32bit",
                              "originalIndices", "vertexColors"]

        if self.version >= 1.2:
            default_attributes.append('posGroups')
        if self.version >= 1.25:
            default_attributes.append('uvSeams')
            default_attributes.append('rivets')
            default_attributes.append('externalSkel')
            default_attributes.append('faceSizes')
            default_attributes.append('creases')
            default_attributes.append('extremeIndices')
            default_attributes.append('vertToPointIndices')
            default_attributes.append('jointMasks')
        for attr in default_attributes:
            self.options[attr] = self.reader.get_bit()
        if self.version >= 1.2:
            self.reader.i1_pointer += 49 - len(default_attributes)
        self.geometry.main_skeleton = (not self.options['externalSkel']) and \
                                      (not self.options['addon']) and \
                                      self.options['weights']

    def _init_indices(self):
        if self.options['mesh']:
            indices_count = self.reader.get_int32()
            if self.i32_indices:
                self.geometry.index = self.reader.get_uint32_array(indices_count)
            else:
                self.geometry.index = self.reader.get_uint16_array(indices_count)
            if self.options['originalIndices']:
                if self.i32_indices:
                    self.geometry.original_indices = self.reader.get_uint32_array(indices_count)
                else:
                    self.geometry.original_indices = self.reader.get_uint16_array(indices_count)

    def _init_points(self):
        if self.options['mesh']:
            vertex_count = self.reader.get_int32() if self.i32_indices else self.reader.get_int16()
            self.vertex_count = vertex_count
            self.geometry.has_geometry = True
            # Z Y X
            bbox = [self.reader.get_float() for _ in range(6)]
            scale = [bbox[3] - bbox[0], bbox[4] - bbox[1], (bbox[5] - bbox[2])]
            self.geometry.offset = [bbox[0] * scale[0], bbox[1] * scale[1], bbox[2] * scale[2]]
            self.geometry.bounds = [bbox[0:3], bbox[3:6]]
            verts = np.zeros((vertex_count, 3), np.float32)
            for i in range(vertex_count):
                verts[i, 0] = self.reader.get_uint16() / self.ge * scale[0] + bbox[0]
                verts[i, 1] = self.reader.get_uint16() / self.ge * scale[1] + bbox[1]
                verts[i, 2] = self.reader.get_uint16() / self.ge * scale[2] + bbox[2]

            self.geometry.positions = verts

    def _init_normals(self):
        if self.options['normals']:
            if self.vertex_count > 0:
                normals = np.zeros((self.vertex_count, 3), np.float32)
                for i in range(self.vertex_count):
                    x = normals[i, 0] = (((self.reader.get_int8() / self.me) % 1) * 2 - 1)
                    y = normals[i, 1] = (((self.reader.get_int8() / self.me) % 1) * 2 - 1)
                    normals[i, 2] = (2 * self.reader.get_bit() - 1) * math.sqrt(max(1 - x * x - y * y, 0))
                self.geometry.normals = normals

    def _init_uvs(self):
        if self.options['uv1']:
            uvs = ['uv']
            if self.options['uv2']:
                uvs.append('uv2')
            for uv_layer_name in uvs:
                uv_bbox = [self.reader.get_float() for _ in range(4)]
                uv_scale = [uv_bbox[2] - uv_bbox[0], uv_bbox[3] - uv_bbox[1]]
                uvs = []
                for i in range(self.vertex_count):
                    uvs.append((
                        ((self.reader.get_int16() / self.ge) % 1) * uv_scale[0] + uv_bbox[0],
                        ((self.reader.get_int16() / self.ge) % 1) * uv_scale[1] + uv_bbox[1],
                    ))
                uvs = np.asarray(uvs)
                setattr(self.geometry, uv_layer_name, uvs)

    def _init_vertex_colors(self):
        if self.options['vertexColors']:
            layer_count = self.reader.get_int8()
            for t in range(layer_count):
                layer_name = self.reader.get_string()
                v_colors = np.ones(self.vertex_count, np.float32)
                v_colors[:] = self.reader.get_int8_array(self.vertex_count)
                v_colors[:] /= 255
                self.geometry.vertex_colors[layer_name] = v_colors

    def _init_blends(self):
        if self.options['blendTargets']:
            shape_key_count = self.reader.get_int16() if self.after_1_7 else self.reader.get_int8()
            if shape_key_count:
                shape_key_data = {}
                for shape_key_id in range(shape_key_count):
                    shape_key_name = self.reader.get_string()
                    bbox = [self.reader.get_float() for _ in range(6)]
                    scale = [bbox[3] - bbox[0], bbox[4] - bbox[1], bbox[5] - bbox[2]]
                    c = []
                    for d in range(self.vertex_count):
                        c.append(((self.reader.get_int8() / self.me) % 1) * scale[0] + bbox[0])
                        c.append(((self.reader.get_int8() / self.me) % 1) * scale[1] + bbox[1])
                        c.append(((self.reader.get_int8() / self.me) % 1) * scale[2] + bbox[2])
                    shape_key_data[shape_key_name] = np.asarray(c).reshape((-1, 3))
                    if self.options['blendNormals']:
                        for _ in range(self.vertex_count):
                            self.reader.get_int8()
                            self.reader.get_int8()
                            self.reader.get_bit()
                self.geometry.shape_key_data = shape_key_data

    def _init_weights(self):
        vertex_count = self.vertex_count
        if self.options['weights']:
            self.geometry.skinned = True
            total_influences = self.reader.get_int8()
            skin_weights = np.zeros((vertex_count, total_influences), np.float32)

            skin_indices = self.reader.get_uint16_array(vertex_count * total_influences).reshape(
                (-1, total_influences)).copy()
            if self.before_1_7:
                skin_indices[skin_indices >= 500] += 4500
            m = self.options['originalIndices']

            for c in range(vertex_count):
                for w in range(total_influences):
                    r = self.reader.get_uint16()
                    val = r / self.ge
                    # M = 4 * (r >> 0 & 1) + 2 * (r >> 2 & 1) + (r >> 3 & 1) - 3
                    # x = (M + 4) % 2 == 0
                    # K = (val * (1 if x else -1) + (0 if x else 1) + M) * (2 * (r >> 1 & 1) - 1)
                    # if val <= 0 or val >= 1 or m:
                    #     K = val

                    # skin_weights[c, w] = K
                    skin_weights[c, w] = val

            self.geometry.skin_indices = skin_indices
            self.geometry.skin_weights = skin_weights

    def _init_parent(self):
        if self.options['singleParent']:
            bone = HeroBone()
            bone.name = self.reader.get_string()
            bone_id = self.reader.get_uint16()
            if self.before_1_7 and bone_id >= 500:
                bone_id += 4500
            bone.bone_id = bone_id
            self.geometry.bones.append(bone)

            r = np.zeros((self.vertex_count, 4), dtype=np.uint16)
            i = np.zeros((self.vertex_count, 4), dtype=np.float32)
            r[:, 0] = bone_id
            i[:, 0] = 1
            self.geometry.skin_indices = r
            self.geometry.skin_weights = i

    def _init_poses(self):
        if self.options['animations']:
            bone_group_count = self.reader.get_int16() if self.after_1_7 else self.reader.get_int8()
            if self.options['frameMappings']:
                n = self.reader.get_uint16()
                a = self.reader.get_int16_array(n)
                if n:
                    frame_mappings = {}
                    for s in range(n):
                        frame_mappings[a[s]] = s
            pos_scale = self.reader.get_float()
            joint_scales = self.options['jointScales']
            joint_pre_scale = self.reader.get_float() if joint_scales else 1
            locators = {}
            bones = []

            def get_bone_matrix(frame_count_):
                res = {
                    "frameMapping": frame_mappings if self.options["frameMappings"] else None
                }
                if self.reader.get_bit():
                    res['pos'] = self.reader.get_position_array(1, pos_scale)
                else:
                    res['pos'] = self.reader.get_position_array(frame_count_, pos_scale)

                if self.reader.get_bit():
                    res['rot'] = self.reader.get_quaternion_array(1)
                else:
                    res['rot'] = self.reader.get_quaternion_array(frame_count_)

                if joint_scales:
                    if self.reader.get_bit():
                        res['scl'] = self.reader.get_scale_array(1, joint_pre_scale)
                    else:
                        res['scl'] = self.reader.get_scale_array(frame_count_, joint_pre_scale)
                else:
                    res['scl'] = [1, 1, 1]
                return res

            for y in range(bone_group_count):
                bone_type = self.reader.get_string()
                bone_count = self.reader.get_int16()
                frame_count = self.reader.get_int16()
                if bone_type == 'main':
                    for bone_id in range(bone_count):
                        bone_parent = self.reader.get_uint16()
                        bone = HeroBone()
                        bone.name = self.reader.get_string()
                        if self.before_1_7:
                            if bone_parent == 5e3:
                                bone_parent = 9999
                            else:
                                if bone_parent >= 500:
                                    bone_parent += 4500
                        if bone_parent == 9999:
                            bone_parent = -1
                            self.geometry.main_skeleton = True

                        bone.parent_id = bone_parent
                        bone.bone_id = bone_id

                        matrix = get_bone_matrix(frame_count)
                        bone.pos = matrix['pos']
                        bone.quat = matrix['rot']
                        bone.scale = matrix['scl']
                        bones.append(bone)
                elif bone_type == 'locators':
                    for R in range(bone_count):
                        bone = HeroBone()
                        bone.name = self.reader.get_string()
                        matrix = get_bone_matrix(frame_count)
                        bone.pos = matrix['pos']
                        bone.scale = matrix['scl']
                        bone.quat = matrix['rot']
                        locators[bone.name] = bone
                else:
                    c = {}
                    for x in range(bone_count):
                        name = self.reader.get_string()
                        c[name] = get_bone_matrix(frame_count)
                    self.geometry.poses[bone_type] = c
            self.geometry.bones = bones
            self.geometry.locations = locators
