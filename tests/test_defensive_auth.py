from concurrent.futures import ThreadPoolExecutor
import unittest

from puf_lab.defensive_auth import KEY_BYTES, Verifier, demo, make_proof


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.key = b"a" * KEY_BYTES
        self.other_key = b"b" * KEY_BYTES
        self.now = 10.0
        self.server = Verifier({"a": self.key, "b": self.other_key},
                               lifetime=30, clock=lambda: self.now)

    def test_accepts_fresh_proof_only_once(self):
        nonce = self.server.issue("a")
        proof = make_proof(self.key, "a", nonce)
        self.assertTrue(self.server.verify("a", nonce, proof))
        self.assertFalse(self.server.verify("a", nonce, proof))

    def test_requires_proof_for_issued_challenge(self):
        old_nonce = self.server.issue("a")
        new_nonce = self.server.issue("a")
        self.assertFalse(self.server.verify("a", new_nonce, make_proof(self.key, "a", old_nonce)))
        self.assertTrue(self.server.verify("a", new_nonce, make_proof(self.key, "a", new_nonce)))

    def test_rejects_wrong_identity_and_key(self):
        nonce = self.server.issue("a")
        proof = make_proof(self.key, "a", nonce)
        self.assertFalse(self.server.verify("b", nonce, proof))
        self.assertFalse(self.server.verify("a", nonce, make_proof(self.other_key, "a", nonce)))
        self.assertTrue(self.server.verify("a", nonce, proof))

    def test_rejects_at_expiration(self):
        nonce = self.server.issue("a")
        self.now = 40.0
        self.assertFalse(self.server.verify("a", nonce, make_proof(self.key, "a", nonce)))

    def test_consumption_is_atomic_within_one_process(self):
        nonce = self.server.issue("a")
        proof = make_proof(self.key, "a", nonce)
        with ThreadPoolExecutor(max_workers=2) as pool:
            submissions = [pool.submit(self.server.verify, "a", nonce, proof) for _ in range(2)]
            self.assertEqual(sum(call.result() for call in submissions), 1)

    def test_demo_returns_only_expected_outcomes(self):
        self.assertEqual(demo(), {
            "fresh_proof_accepted": True,
            "repeated_proof_rejected": True,
            "wrong_device_key_rejected": True,
            "expired_challenge_rejected": True,
            "concurrent_acceptance_count": 1,
        })


if __name__ == "__main__":
    unittest.main()
