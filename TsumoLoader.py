import numpy as np

class TsumoLoader:

    def __init__(self, tsumo_path: str = "tsumo/tsumo.txt", mapping_path="tsumo/color_mapping.txt", record_size: int = 257, map_record_size: int = 5):
        self.tsumo_path = tsumo_path
        self.mapping_path = mapping_path
        self.record_size = record_size
        self.map_record_size = map_record_size
        self.tsumo_file = open(tsumo_path, "rb")
        self.mapping_file = open(mapping_path, "rb")

    def __del__(self):
        for f in (self.tsumo_file, self.mapping_file):
            try:
                f.close()
            except Exception:
                pass

    def load(self, seed: int) -> np.ndarray:
        seed %= 0xffff
        offset = seed * self.record_size
        self.tsumo_file.seek(offset)
        line_bytes = self.tsumo_file.read(256)
        if len(line_bytes) < 256:
            raise IndexError(f"Seed {seed} out of range in {self.tsumo_path}")
        return (np.frombuffer(line_bytes, dtype=np.uint8) - ord('0')).astype(np.int32)
    
    def load_mapping(self, seed: int) -> dict[int, str]:
        seed %= 0xffff
        offset = seed * self.map_record_size
        self.mapping_file.seek(offset)
        line_bytes = self.mapping_file.readline().strip()
        if not line_bytes:
            raise IndexError(f"Seed {seed} out of range in {self.mapping_path}")

        # decode and map 1→r, 2→g, ...
        line_str = line_bytes.decode("utf-8")
        mapping = {i+1: ch for i, ch in enumerate(line_str)}
        return mapping