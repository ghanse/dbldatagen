"""Vectorized numpy and Arrow building blocks shared by the dataset generators."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from decimal import Decimal

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWER = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
HEX = "0123456789abcdef"

Rng = Callable[[str], np.random.Generator]
STRUCTURE = "structure"

_CONSONANTS = "bcdfghjklmnprstvz"
_VOWELS = "aeiou"


def rng_factory(seed: int, dataset: str, namespace: str = "") -> Rng:
    """Returns a function that maps a stream key to an independent, reproducible generator.

    Each key gets its own stream derived from the seed, the dataset name, the namespace, and the key,
    so adding or reordering columns does not change the values of other columns. Generators use the
    empty namespace with the sample seed for rows, and `STRUCTURE` with the structure seed for
    generating parameters and vocabularies.

    Args:
        seed: Sample or structure seed.
        dataset: Dataset name.
        namespace: Stream namespace.
    """
    prefix = [seed, _stable_int(dataset)] + ([_stable_int(namespace)] if namespace else [])

    def rng(key: str) -> np.random.Generator:
        return np.random.default_rng([*prefix, _stable_int(key)])

    return rng


def _stable_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def random_strings(rng: np.random.Generator, n: int, alphabets: Sequence[str]) -> pa.Array:
    """Returns `n` fixed-length ASCII strings with one character drawn per position.

    Args:
        rng: Random generator.
        n: Number of strings.
        alphabets: Characters allowed at each position; a one-character alphabet is a literal.
    """
    codes = np.empty((n, len(alphabets)), dtype=np.uint8)
    for i, alphabet in enumerate(alphabets):
        chars = np.frombuffer(alphabet.encode("ascii"), dtype=np.uint8)
        codes[:, i] = chars[rng.integers(0, len(chars), n)] if len(chars) > 1 else chars[0]
    return fixed_strings(codes)


def indexed_strings(index: np.ndarray, alphabets: Sequence[str]) -> pa.Array:
    """Returns the fixed-length strings that encode `index` in the mixed radix given by `alphabets`.

    Distinct indexes give distinct strings, which makes this the way to draw unique codes.

    Args:
        index: Non-negative integers below the product of the alphabet sizes.
        alphabets: Characters allowed at each position, most significant first.
    """
    codes = np.empty((len(index), len(alphabets)), dtype=np.uint8)
    rest = np.asarray(index, dtype=np.int64)
    for i in reversed(range(len(alphabets))):
        chars = np.frombuffer(alphabets[i].encode("ascii"), dtype=np.uint8)
        codes[:, i] = chars[rest % len(chars)]
        rest = rest // len(chars)
    return fixed_strings(codes)


def fixed_strings(codes: np.ndarray) -> pa.Array:
    """Returns one ASCII string per row of a `(n, width)` uint8 character-code matrix."""
    n, width = codes.shape
    offsets = (np.arange(n + 1, dtype=np.int64) * width).astype(np.int32)
    data = np.ascontiguousarray(codes, dtype=np.uint8)
    return pa.Array.from_buffers(pa.string(), n, [None, pa.py_buffer(offsets), pa.py_buffer(data)])


def take(values: Sequence[object], index: np.ndarray, type: pa.DataType | None = None) -> pa.Array:
    """Returns `values[index]` as an Arrow array."""
    return pa.array(values, type=type).take(pa.array(index))


def join(*parts: pa.Array | str, sep: str = "") -> pa.Array:
    """Concatenates string arrays and literals element-wise."""
    return pc.binary_join_element_wise(*parts, sep)


def with_nulls(array: pa.Array, mask: np.ndarray) -> pa.Array:
    """Returns `array` with nulls where `mask` is true."""
    return pc.if_else(pa.array(mask), pa.scalar(None, array.type), array)


def null_mask(rng: np.random.Generator, n: int, rate: float) -> np.ndarray:
    """Returns a boolean mask that is true with probability `rate`."""
    return rng.random(n) < rate


def cents_to_decimal(cents: np.ndarray, precision: int) -> pa.Array:
    """Returns integer cents as a `decimal(precision, 2)` array."""
    scaled = pc.multiply(pa.array(cents, pa.int64()).cast(pa.decimal128(19, 0)), pa.scalar(Decimal("0.01")))
    return scaled.cast(pa.decimal128(precision, 2))


def choice(rng: np.random.Generator, n: int, probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
    """Returns `n` indexes drawn with the given probabilities."""
    return np.asarray(rng.choice(len(probabilities), size=n, p=np.asarray(probabilities)))


def zipf_probabilities(n_values: int, s: float) -> np.ndarray:
    """Returns the probabilities of ranks `1..n_values` under a finite Zipf law with exponent `s`."""
    weights = np.arange(1, n_values + 1, dtype=np.float64) ** -s
    return np.asarray(weights / weights.sum())


def zipf_choice(
    rng: np.random.Generator, n: int, n_values: int, s: float, order: np.random.Generator | None = None
) -> np.ndarray:
    """Returns `n` indexes in `[0, n_values)` drawn by Zipf rank, with ranks assigned in random order.

    Args:
        rng: Generator for the draws.
        n: Number of draws.
        n_values: Number of values.
        s: Zipf exponent.
        order: Generator for the rank order of the values; `rng` when None. Pass a structure stream
            when the values are a fixed vocabulary, so which value is most frequent is structure.
    """
    ranks = rng.choice(n_values, size=n, p=zipf_probabilities(n_values, s))
    return (order or rng).permutation(n_values)[ranks]


def words(rng: np.random.Generator, count: int, min_syllables: int = 2, max_syllables: int = 3) -> list[str]:
    """Returns `count` distinct pronounceable lowercase words built from consonant-vowel syllables."""
    found: dict[str, None] = {}
    while len(found) < count:
        syllables = rng.integers(min_syllables, max_syllables + 1, count)
        consonants = rng.integers(0, len(_CONSONANTS), (count, max_syllables))
        vowels = rng.integers(0, len(_VOWELS), (count, max_syllables))
        for length, cs, vs in zip(syllables, consonants, vowels, strict=True):
            found["".join(_CONSONANTS[c] + _VOWELS[v] for c, v in zip(cs[:length], vs[:length], strict=True))] = None
    return list(found)[:count]


def sentences(rng: np.random.Generator, vocabulary: Sequence[str], lengths: np.ndarray) -> pa.Array:
    """Returns one space-separated sentence per entry of `lengths`, with words drawn uniformly."""
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32)
    flat = take(vocabulary, rng.integers(0, len(vocabulary), int(offsets[-1])))
    return pc.binary_join(pa.ListArray.from_arrays(pa.array(offsets), flat), " ")


def list_offsets(lengths: np.ndarray) -> pa.Array:
    """Returns int32 list offsets for the given per-row lengths."""
    return pa.array(np.concatenate([[0], np.cumsum(lengths)]).astype(np.int32))


def micros(iso: str) -> int:
    """Returns microseconds since the Unix epoch for an ISO date or naive timestamp string."""
    return int(np.datetime64(iso, "us").astype(np.int64))


def timestamps(epoch_micros: np.ndarray, utc: bool, mask: np.ndarray | None = None) -> pa.Array:
    """Returns a microsecond timestamp array, UTC-adjusted (`timestamp`) or wall-clock (`timestamp_ntz`)."""
    return pa.array(epoch_micros, pa.int64(), mask=mask).cast(pa.timestamp("us", tz="UTC" if utc else None))


def dates(epoch_days: np.ndarray, mask: np.ndarray | None = None) -> pa.Array:
    """Returns a date array from days since the Unix epoch."""
    return pa.array(epoch_days, pa.int32(), mask=mask).cast(pa.date32())


def days(iso: str) -> int:
    """Returns days since the Unix epoch for an ISO date string."""
    return int(np.datetime64(iso, "D").astype(np.int64))


_MASK63 = np.uint64(2**63 - 1)


def hashed_ids(index: np.ndarray, salt: int) -> np.ndarray:
    """Returns distinct non-negative int64 ids that look random, by a keyed bijection of `index` on 63 bits.

    Every step (xor with the salt, xor-shift, multiplication by an odd constant, all mod 2**63) is a
    bijection, so distinct indexes give distinct ids.

    Args:
        index: Distinct non-negative integers below 2**63.
        salt: Key of the bijection.
    """
    x = (np.asarray(index, dtype=np.uint64) ^ np.uint64(salt & (2**63 - 1))) & _MASK63
    with np.errstate(over="ignore"):
        for shift, multiplier in ((31, 0x7FB5D329728EA185), (27, 0x01FF3D2A8CB7F6F1), (33, 0x1D2F3C9AAB1F0E8B)):
            x ^= x >> np.uint64(shift)
            x = (x * np.uint64(multiplier)) & _MASK63
    return x.astype(np.int64)


def shuffle(table: pa.Table, rng: np.random.Generator) -> pa.Table:
    """Returns `table` with its rows in a random order."""
    return table.take(pa.array(rng.permutation(table.num_rows)))
