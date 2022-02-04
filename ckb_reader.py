import math

import numpy as np

from .byte_io_ckb import ByteIO


class CKBReader:
    me = (2 ** 8) - 1
    ge = (2 ** 16) - 1
    H = math.pow(2, 16) - 1
    X = (math.pow(2, 16) - 2) / 2

    def __init__(self, reader: ByteIO):
        self.version = round(reader.read_float(), 2)
        self.export_time = 0
        self.i1_buffer = np.zeros((0,), np.bool)
        self.i8_buffer = np.zeros((0,), np.int8)
        self.i16_buffer = np.zeros((0,), np.int16)
        self.i32_buffer = np.zeros((0,), np.int32)
        self._i1_offset = 0
        self.i1_pointer = 0

        self._i8_offset = 0
        self.i8_pointer = 0

        self._i16_offset = 0
        self.i16_pointer = 0

        self._i32_offset = 0
        self.i32_pointer = 0

        self._get_start_points(reader)

    def _get_start_points(self, reader: ByteIO):
        i32_count = math.ceil(reader.read_float())
        i16_count = math.ceil(reader.read_float())
        i8_count = math.ceil(reader.read_float())
        i1_count = math.ceil(reader.read_float() / 8)
        if self.version >= 1.4:
            self.export_time = reader.read_float()
        self.i32_pointer = 0
        self.i32_buffer = np.frombuffer(reader.read(4 * i32_count), np.int32)
        self.i16_buffer = np.frombuffer(reader.read(2 * i16_count), np.int16)
        self.i8_buffer = np.frombuffer(reader.read(i8_count), np.int8)
        tmp = []
        for _ in range(math.ceil(i1_count)):
            byte = reader.read_int8()
            for i in range(8):
                tmp.append(bool(byte & (1 << i)))
        self.i1_buffer = np.asarray(tmp, np.bool)

    def get_bit(self):
        self.i1_pointer += 1
        return bool(self.i1_buffer[self.i1_pointer - 1])

    def get_float(self):
        self.i32_pointer += 1
        return float(self.i32_buffer.view(np.float32)[self.i32_pointer - 1])

    def get_float_array(self, size):
        self.i32_pointer += size
        return self.i32_buffer.view(np.float32)[self.i32_pointer - size:self.i32_pointer]

    def get_int32(self):
        return int(self.get_float())

    def get_uint32_array(self, size):
        self.i32_pointer += size
        return [int(a) for a in self.i32_buffer.view(np.float32)[self.i32_pointer - size:self.i32_pointer]]

    def get_int16(self):
        self.i16_pointer += 1
        return int(self.i16_buffer[self.i16_pointer - 1])

    def get_uint16(self):
        self.i16_pointer += 1
        return int(self.i16_buffer.view(np.uint16)[self.i16_pointer - 1])

    def get_int16_array(self, size):
        self.i16_pointer += size
        return self.i16_buffer[self.i16_pointer - size:self.i16_pointer]

    def get_uint16_array(self, size):
        return self.get_int16_array(size).view(np.uint16)

    def get_int8(self):
        self.i8_pointer += 1
        return int(self.i8_buffer[self.i8_pointer - 1])

    def get_int8_array(self, size):
        self.i8_pointer += size
        return self.i8_buffer[self.i8_pointer - size:self.i8_pointer]

    def get_uint8_array(self, size):
        return self.get_int8_array(size).view(np.uint8)

    def get_string(self) -> str:
        return self.get_int8_array(self.get_int8()).tobytes().decode("utf-8")

    def get_quaternion_array(self, e):
        e *= 4
        quaternions = np.zeros((e,), np.float32)
        for n in range(e):
            quaternions[n] = self.get_uint16() / self.H * 2 - 1
        return quaternions.reshape((-1, 4))

    def get_position_array(self, e, scale):
        positions = np.zeros((e, 3), np.float32)
        for i in range(e):
            for channel in range(3):
                positions[i, channel] = (self.get_uint16() - self.X) / self.X * scale
        return positions

    def get_scale_array(self, e, scale):
        scales = np.zeros((e, 3), np.float32)
        for i in range(e):
            for channel in range(3):
                scales[i, channel] = (self.get_uint16()) / self.H * scale
        return scales
