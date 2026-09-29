"""Slow, deterministic CPU arithmetic coder for the explicitly labelled tiny backend.

The production DiT-IC backend uses CompressAI entropy models and rANS. This
reference coder makes byte-level tests possible without pretending zlib is rANS.
"""
from __future__ import annotations
import bisect
import functools
import math
import struct
import torch
from torch import nn
from .math_ops import gaussian_mass, ste_round

TOP = 1 << 32
HALF = TOP >> 1
QUARTER = HALF >> 1
TOTAL = 1 << 16
RADIUS = 255


class _Bits:
    def __init__(self):
        self.data = bytearray(); self.byte = 0; self.n = 0
    def put(self, bit):
        self.byte = (self.byte << 1) | bit; self.n += 1
        if self.n == 8:
            self.data.append(self.byte); self.byte = 0; self.n = 0
    def finish(self):
        if self.n: self.data.append(self.byte << (8 - self.n))
        return bytes(self.data)


class _Reader:
    def __init__(self, data): self.data = data; self.pos = 0
    def get(self):
        i = self.pos; self.pos += 1
        if i >= len(self.data) * 8:
            if i > len(self.data) * 8 + 64: raise ValueError('Truncated arithmetic payload')
            return 0
        return (self.data[i // 8] >> (7 - i % 8)) & 1


@functools.lru_cache(maxsize=64)
def cdf_for(index: int):
    sigma = math.exp(math.log(.11) + index / 63 * math.log(256 / .11))
    cdf = lambda x: .5 * (1 + math.erf(x / (sigma * math.sqrt(2))))
    p = [max(0., cdf(k + .5) - cdf(k - .5)) for k in range(-RADIUS, RADIUS + 1)]
    p.append(max(0., 1 - sum(p)))
    n = len(p)
    counts = [1 + int(v * (TOTAL - n)) for v in p]
    counts[max(range(n), key=lambda k: p[k])] += TOTAL - sum(counts)
    cumulative = [0]
    for v in counts: cumulative.append(cumulative[-1] + v)
    return cumulative


def scale_indexes(scale):
    return (((scale.detach().cpu().clamp(.11, 256).log() - math.log(.11))
             / math.log(256 / .11) * 63).round().long()).flatten().tolist()


def encode_symbols(symbols: list[int], indexes: list[int]) -> bytes:
    if len(symbols) != len(indexes): raise ValueError('Shape mismatch')
    bits = _Bits(); low = 0; high = TOP - 1; pending = 0; escapes = []
    def emit(bit):
        nonlocal pending
        bits.put(bit)
        for _ in range(pending): bits.put(1 - bit)
        pending = 0
    for value, index in zip(symbols, indexes):
        code = value + RADIUS
        if not -RADIUS <= value <= RADIUS:
            code = 2 * RADIUS + 1; escapes.append(value)
        cdf = cdf_for(index); width = high - low + 1
        high = low + width * cdf[code+1] // TOTAL - 1
        low = low + width * cdf[code] // TOTAL
        while True:
            if high < HALF: emit(0)
            elif low >= HALF: emit(1); low -= HALF; high -= HALF
            elif low >= QUARTER and high < 3 * QUARTER:
                pending += 1; low -= QUARTER; high -= QUARTER
            else: break
            low *= 2; high = 2 * high + 1
    pending += 1; emit(0 if low < QUARTER else 1)
    raw = bits.finish()
    return struct.pack('>III', len(symbols), len(raw), len(escapes)) + raw + b''.join(struct.pack('>i', i) for i in escapes)


def decode_symbols(data: bytes, indexes: list[int]) -> list[int]:
    if len(data) < 12: raise ValueError('Truncated reference stream')
    n, size, extra = struct.unpack('>III', data[:12])
    if n != len(indexes) or len(data) != 12 + size + 4 * extra:
        raise ValueError('Reference stream length mismatch')
    bits = _Reader(data[12:12+size]); low = 0; high = TOP - 1; value = 0
    for _ in range(32): value = value * 2 + bits.get()
    escapes = list(struct.unpack('>' + 'i' * extra, data[12+size:])) if extra else []
    result = []; used = 0
    for index in indexes:
        cdf = cdf_for(index); width = high - low + 1
        target = ((value - low + 1) * TOTAL - 1) // width
        code = bisect.bisect_right(cdf, target) - 1
        if not 0 <= code < len(cdf) - 1: raise ValueError('Invalid arithmetic code')
        high = low + width * cdf[code+1] // TOTAL - 1
        low = low + width * cdf[code] // TOTAL
        while True:
            if high < HALF: pass
            elif low >= HALF: low -= HALF; high -= HALF; value -= HALF
            elif low >= QUARTER and high < 3 * QUARTER:
                low -= QUARTER; high -= QUARTER; value -= QUARTER
            else: break
            low *= 2; high = 2 * high + 1; value = 2 * value + bits.get()
        if code == 2 * RADIUS + 1:
            if used >= extra: raise ValueError('Missing escape')
            result.append(escapes[used]); used += 1
        else: result.append(code - RADIUS)
    if used != extra: raise ValueError('Unused escape data')
    return result


class ReferenceGaussian(nn.Module):
    def forward(self, y, scales, means):
        q = ste_round(y - means) + means
        return q, gaussian_mass(y, means, scales)
    def compress(self, y, indexes, means):
        return [encode_symbols((a-b).round().int().flatten().tolist(), scale_indexes(s))
                for a, b, s in zip(y, means, indexes)]
    def decompress(self, strings, indexes, means):
        out = [torch.tensor(decode_symbols(data, scale_indexes(s)), dtype=means.dtype,
                            device=means.device).reshape_as(s) + m
               for data, s, m in zip(strings, indexes, means)]
        return torch.stack(out)
    def build_indexes(self, scales): return scales
    def update_scale_table(self, *args, **kwargs): return True


class ReferenceBottleneck(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.medians = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.log_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gaussian = ReferenceGaussian()
    def _get_medians(self): return self.medians
    def forward(self, z):
        return self.gaussian(z, self.log_scale.exp(), self.medians)
    def compress(self, z):
        return self.gaussian.compress(z, self.log_scale.exp().expand_as(z), self.medians.expand_as(z))
    def decompress(self, strings, shape):
        size = (len(strings), self.medians.shape[1], *shape)
        return self.gaussian.decompress(strings, self.log_scale.exp().expand(size), self.medians.expand(size))
    def loss(self): return self.medians.sum() * 0
    def update(self, *args, **kwargs): return True
