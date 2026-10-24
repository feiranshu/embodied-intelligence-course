"""Signed raw-count coordinates around a measured joint reference."""


def mapped_position(raw, mapping):
    position = mapping.get("sign", 1) * ((int(raw) - mapping["zero_raw"] + 512) % 1024 - 512)
    if mapping.get("unwrapped"):
        center = sum(mapping["limits"]) / 2
        position += 1024 * round((center - position) / 1024)
    return position


class PositionTrace:
    def __init__(self, raw):
        self.raw = int(raw)
        self.position = 0
        self.low = 0
        self.high = 0

    def update(self, raw):
        raw = int(raw)
        delta = (raw - self.raw + 512) % 1024 - 512
        if abs(delta) > 128:
            raise ValueError("采样间位置变化过大，无法判断路径；请减慢手动转动速度后重试此关节")
        self.position += delta
        self.raw = raw
        self.low = min(self.low, self.position)
        self.high = max(self.high, self.position)
        if self.high - self.low >= 1000:
            raise ValueError("采集路径接近或超过完整反馈周期，无法唯一映射；请缩小该关节工作范围")
        return self.position


def raw_position(position, mapping):
    return (mapping["zero_raw"] + mapping.get("sign", 1) * int(position)) % 1024


def check_position(raw, mapping, tolerance=0):
    position = mapped_position(raw, mapping)
    low, high = mapping["limits"]
    if not low - tolerance <= position <= high + tolerance:
        raise ValueError(f"映射位置 {position} 超出安全范围 [{low}, {high}]")
    return position
