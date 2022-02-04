import json
from typing import Dict

import numpy as np
from requests import get

from .byte_io_ckb import split
from .ckb_file import CKBFile
from multiprocessing import Pool


def download(url, part):
    return get(url), part


class HeroBuilder:
    host_url = 'https://www.heroforge.com'
    mesh_bundle_url = "/static/herobundles/"
    mesh_bundle_path_high_url = "/forge_static/herobundles/"
    parts_info_url = f'{host_url}/static/js/options.min.json?version=heroforge08.1.7.95'
    prefix = 'hf'
    forge = True
    vintage = 9999.0

    def __init__(self, config_url: str):
        self.parts_info = get(self.parts_info_url).json()
        self.model_config = {}

        config_page = get(config_url)
        config_text = config_page.text
        for line in config_text.split('\n'):
            line = line.strip('\t\n ').rstrip('\t\n ')
            if line.startswith('var LOAD_CONFIG'):
                line = line.replace('var LOAD_CONFIG = \'', '')[:-2]
                self.model_config = json.loads(line)

        self.user_parts = {}
        self.parts: Dict[int, CKBFile] = {}
        self.skeleton = self.get_skeleton()

    @property
    def name(self):
        return self.model_config['meta']['character_name']

    @staticmethod
    def _find_part_by_id(parts, part_id):
        for part in parts.values():
            if part['id'] == part_id:
                return part
        return None

    @staticmethod
    def _find_part_by_link(parts, link_id):
        for part in parts.values():
            if part['link'] == link_id:
                return part
        return None

    def fetch_parts(self):
        for part_type, part_id in self.model_config['parts'].items():
            if part_type.endswith('L'):
                part_lib_name = part_type[:-1] + 'R'
            else:
                part_lib_name = part_type
            part_lib = self.parts_info['parts'][part_lib_name]
            if part_type.endswith('L'):
                part = self._find_part_by_link(part_lib, part_id)
            else:
                part = self._find_part_by_id(part_lib, part_id)
            self.user_parts[part['id']] = part

    def download_parts(self):
        urls = []
        for part in self.user_parts.values():
            url = self.get_mesh_path(part, False)
            urls.append(self.host_url + url)

        print('Downloading parts')
        with Pool(8) as p:
            results = p.starmap(download, zip(urls, self.user_parts.values()))

        for response, part in results:
            response.raise_for_status()
            ckb_file = CKBFile(response.content, f'{part["slot"]}_{part["name"]}')
            ckb_file.read()
            self.parts[part['id']] = ckb_file

    def get_mesh_bundle_url(self, high_detail=False, prefix=None):
        url = self.mesh_bundle_path_high_url if high_detail else self.mesh_bundle_url
        if prefix is not None:
            url += (prefix + '/')
        return url

    @staticmethod
    def version_affix(part):
        return f'?{part.get("version", 0)}=pv'

    @staticmethod
    def get_rez(hi_res):
        return 'hiRez' if hi_res else 'loRez'

    def get_skeleton(self):
        skeleton_type = list(self.model_config['poses'].keys())[0]
        url = self.get_mesh_bundle_url(False) + f"skeleton/{skeleton_type}.ckb"
        ckb_file = CKBFile(get(self.host_url + url).content, skeleton_type)
        ckb_file.read()
        return ckb_file

    def get_mesh_path(self, part, load_high_detail, n='', i=''):
        if i and part['slot'] == 'face' and part['name'].endswith('z'):
            parts = []
            part = next(filter(lambda k, v: v['slot'] == 'face' and part['name'].endswith('z'), parts))
        prefix = "vintage_2020_05_08" if self.forge and "baseItem" == part['slot'] and self.vintage < 1.95 else ""
        o = self.version_affix(part)
        if part.get('displayFilename', None) is None:
            if '_' in part['name']:
                base_name = part['name'].split('_', 1)[0]
            else:
                base_name = part['name']

            url = self.get_mesh_bundle_url(load_high_detail, prefix)
            url += f"{part['slot']}/{base_name}/"
            url += f"{self.prefix}_{part['slot']}_{self.get_rez(load_high_detail)}_{part['name']}"
            url += f"{('' if part.get('nameAffix', None) is None else part['nameAffix'])}{n}.ckb{o}"
            return url
        else:
            return self.get_mesh_bundle_url(load_high_detail, prefix) + part['displayFilename']

    def build_model(self):
        import bpy
        from mathutils import Vector, Euler, Quaternion, Matrix
        model_name = self.name
        armature_obj = None
        if self.skeleton:
            skel = self.skeleton
            bl_bones = []

            armature = bpy.data.armatures.new(f"{model_name}_ARM_DATA")
            armature_obj = bpy.data.objects.new(f"{model_name}_ARM", armature)
            armature_obj.show_in_front = True

            bpy.context.scene.collection.objects.link(armature_obj)
            armature_obj.select_set(True)
            bpy.context.view_layer.objects.active = armature_obj
            bpy.ops.object.mode_set(mode='EDIT')

            for bone in skel.geometry.bones:
                bone.convert_name()
                bl_bone = armature.edit_bones.new(bone.name[-63:])
                bl_bones.append(bl_bone)

            for bl_bone, bone in zip(bl_bones, skel.geometry.bones):
                bl_bone.tail = (Vector([0, 0, 0.05])) + bl_bone.head
                if bone.parent_id != -1:
                    bl_parent = bl_bones[bone.parent_id]
                    bl_bone.parent = bl_parent
                    del bl_parent

            bpy.ops.object.mode_set(mode='POSE')
            for bone in skel.geometry.bones:

                bl_bone = armature_obj.pose.bones.get(bone.name[-63:])

                mat_loc = Matrix.Translation(bone.pos[0])
                mat_sca = Matrix.Scale(1, 4, bone.scale[0])
                q = bone.quat[0].tolist()
                quat = Quaternion([q[-1], *q[:3]])
                mat_rot = quat.to_matrix().to_4x4()
                mat_out = mat_loc @ mat_rot @ mat_sca
                bl_bone.matrix_basis.identity()
                if bl_bone.parent:
                    bl_bone.matrix = bl_bone.parent.matrix @ mat_out
                else:
                    bl_bone.matrix = mat_out

                # Delete variables so they dont clutter the debugger
                del mat_out, mat_sca, mat_rot, mat_loc, q, quat, bone, bl_bone

            bpy.ops.pose.armature_apply()
            bpy.ops.object.mode_set(mode='OBJECT')
            del bl_bones
            # bpy.context.scene.collection.objects.unlink(armature_obj)

        objects = []

        for part_id, part in self.parts.items():
            part_info = self.user_parts[part_id]
            part_name = f'{part_info["displayname"]}'

            mesh_data = bpy.data.meshes.new(f'{part_name}_MESH')
            mesh_obj = bpy.data.objects.new(part_name, mesh_data)

            geometry = part.geometry
            positions = geometry.positions * np.array([-1, 1, 1])
            mesh_data.from_pydata(positions, [],
                                  np.array(geometry.index).reshape((-1, 3)).tolist())
            mesh_data.update()

            uv_data = mesh_data.uv_layers.new()
            vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
            mesh_data.loops.foreach_get('vertex_index', vertex_indices)
            uv_data.data.foreach_set('uv', np.array(geometry.uv)[vertex_indices].flatten().tolist())
            if part.options.get('uv2', False):
                uv_data = mesh_data.uv_layers.new(name='UV2')
                vertex_indices = np.zeros((len(mesh_data.loops, )), dtype=np.uint32)
                mesh_data.loops.foreach_get('vertex_index', vertex_indices)
                uv_data.data.foreach_set('uv', np.array(geometry.uv2)[vertex_indices].flatten().tolist())

            normals = geometry.normals * [1, -1, -1]
            mesh_data.polygons.foreach_set("use_smooth", np.ones(len(mesh_data.polygons)))
            mesh_data.normals_split_custom_set_from_vertices(normals)
            mesh_data.use_auto_smooth = True

            if geometry.shape_key_data:
                mesh_obj.shape_key_add(name='base')

            for shape_name, shape_data in geometry.shape_key_data.items():
                shape_key = mesh_data.shape_keys.key_blocks.get(shape_name, None) or mesh_obj.shape_key_add(
                    name=shape_name)
                shape_data *= np.array([-1, 1, 1])
                shape_key.data.foreach_set("co", (positions + shape_data).flatten())

            if part.options['vertexColors']:
                groups = split(list(geometry.vertex_colors.items()), 4)
                if len(groups) > 8:
                    print(f"Can only load up to 8 vertex color groups per mesh, mesh has {len(groups)} groups")
                    groups = groups[:8]
                for v_group in groups:
                    name = '/'.join([v[0] for v in v_group])
                    vc = mesh_data.vertex_colors.new(name=name)
                    buffer = np.ones((len(v_group[0][1]), 4), dtype=np.float32)
                    buffer[:, :len(groups) + 1] = np.stack([v[1].flatten() for v in v_group]).T
                    vc.data.foreach_set('color', buffer[vertex_indices].flatten())

            if self.skeleton:
                bone_names = {bone.real_bone_id: bone.name for bone in self.skeleton.geometry.bones}
                weight_groups = {name: mesh_obj.vertex_groups.new(name=name) for name in bone_names.values()}

                for n, (bone_indices, bone_weights) in enumerate(zip(geometry.skin_indices, geometry.skin_weights)):
                    for bone_index, weight in zip(bone_indices, bone_weights):
                        if part.after_1_7 and 500 <= bone_index < 5000:
                            bone_index += 4500
                        if weight > 0:
                            bone_name = bone_names[bone_index]
                            weight_groups[bone_name].add([n], weight, 'REPLACE')

                modifier = mesh_obj.modifiers.new(
                    type="ARMATURE", name="Armature")
                modifier.object = armature_obj
                mesh_obj.parent = armature_obj
            bpy.context.scene.collection.objects.link(mesh_obj)
