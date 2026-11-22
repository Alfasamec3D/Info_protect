"""Local demonstration of keyed authentication with a fresh verifier challenge."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hmac
import json
import math
from pathlib import Path
import secrets
from threading import Lock
import time

KEY_BYTES = 32
NONCE_BYTES = 32
TAG_BYTES = 32
CONTEXT = b"puf-lab/device-auth/v1\0"


def message(device, nonce):
    identity = device.encode("utf-8")
    return CONTEXT + len(identity).to_bytes(4, "big") + identity + nonce


def make_proof(key, device, nonce):
    """Generate a tag inside the trusted authenticator in the demonstration."""
    return hmac.digest(key, message(device, nonce), "sha256")


class Verifier:
    """Verify and consume a challenge atomically within a single process."""

    def __init__(self, keys, *, lifetime=30.0, clock=time.monotonic):
        if not math.isfinite(lifetime) or lifetime <= 0:
            raise ValueError("lifetime must be positive and finite")
        self._keys = dict(keys)
        if any(not isinstance(device, str) or not device for device in self._keys):
            raise ValueError("device identities must be nonempty strings")
        if any(not isinstance(key, bytes) or len(key) != KEY_BYTES for key in self._keys.values()):
            raise ValueError("each device must have a 32-byte key")
        if len(set(self._keys.values())) != len(self._keys):
            raise ValueError("use a distinct key for every device")
        self._lifetime = lifetime
        self._clock = clock
        self._pending = {}
        self._lock = Lock()

    def issue(self, device):
        if device not in self._keys:
            raise KeyError(device)
        with self._lock:
            now = self._clock()
            self._pending = {pair: end for pair, end in self._pending.items() if now < end}
            nonce = secrets.token_bytes(NONCE_BYTES)
            while (device, nonce) in self._pending:
                nonce = secrets.token_bytes(NONCE_BYTES)
            self._pending[device, nonce] = now + self._lifetime
        return nonce

    def verify(self, device, nonce, proof):
        if not isinstance(device, str):
            return False
        if not isinstance(nonce, bytes) or len(nonce) != NONCE_BYTES:
            return False
        if not isinstance(proof, bytes) or len(proof) != TAG_BYTES:
            return False
        pair = (device, nonce)
        with self._lock:
            deadline = self._pending.get(pair)
            if deadline is None:
                return False
            if self._clock() >= deadline:
                del self._pending[pair]
                return False
            if not hmac.compare_digest(make_proof(self._keys[device], device, nonce), proof):
                return False
            del self._pending[pair]
            return True


def demo():
    """Return reproducible pass/fail outcomes while keeping secrets in memory."""
    now = [0.0]
    a, b = "device-a", "device-b"
    keys = {device: secrets.token_bytes(KEY_BYTES) for device in (a, b)}
    server = Verifier(keys, lifetime=30, clock=lambda: now[0])
    nonce = server.issue(a)
    proof = make_proof(keys[a], a, nonce)
    first = server.verify(a, nonce, proof)
    replay = server.verify(a, nonce, proof)

    challenge = server.issue(b)
    cross_device = server.verify(b, challenge, make_proof(keys[a], b, challenge))
    expired_nonce = server.issue(a)
    expired_proof = make_proof(keys[a], a, expired_nonce)
    now[0] = 30.0
    expired = server.verify(a, expired_nonce, expired_proof)

    parallel_nonce = server.issue(a)
    parallel_proof = make_proof(keys[a], a, parallel_nonce)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(server.verify, a, parallel_nonce, parallel_proof) for _ in range(2)]
        parallel_count = sum(future.result() for future in futures)

    return {
        "fresh_proof_accepted": first,
        "repeated_proof_rejected": not replay,
        "wrong_device_key_rejected": not cross_device,
        "expired_challenge_rejected": not expired,
        "concurrent_acceptance_count": parallel_count,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="directory for aggregate JSON outcomes")
    args = parser.parse_args(argv)
    checks = demo()
    payload = json.dumps(checks, indent=2) + "\n"
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "nonce_checks.json").write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
