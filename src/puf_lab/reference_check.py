"""Optional independent comparison against the pinned pypuf simulator."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np

from .model import challenge_features, sample_weights


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pypuf-path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.pypuf_path:
        sys.path.insert(0, str(args.pypuf_path.resolve()))
    from pypuf.simulation import LTFArray, XORArbiterPUF
    rng = np.random.default_rng(90412)
    checked, differences = 0, 0
    for _ in range(16):
        raw = rng.integers(0, 2, (4096, 64), dtype=np.uint8)
        signs = 1 - 2 * raw.astype(np.int8)
        weights = sample_weights(rng, 4, 64)
        features = challenge_features(raw)
        transformed = XORArbiterPUF.transform_atf(signs, 4)
        np.testing.assert_array_equal(transformed, np.broadcast_to(features[:, None, :-1], transformed.shape))
        for chains in (1, 2, 3, 4):
            reference = LTFArray(weight_array=weights[:chains, :-1],
                                 transform=XORArbiterPUF.transform_atf, combiner="xor", bias=weights[:chains, -1])
            ours = np.logical_xor.reduce(features @ weights[:chains].T < 0, axis=1)
            expected = reference.eval(signs) < 0
            differences += int(np.sum(ours != expected))
            checked += len(ours)
    assert differences == 0
    result = {"reference": "pypuf", "version": version("pypuf"), "seed": 90412,
              "devices": 16, "challenge_bits": 64, "chains": [1,2,3,4],
              "compared_response_bits": checked, "mismatches": differences,
              "scope": "identical weights and noiseless queries; feature transform and final XOR",
              "weight_sampling": "our generator; equivalence of random weight distributions not claimed"}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "pypuf_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
