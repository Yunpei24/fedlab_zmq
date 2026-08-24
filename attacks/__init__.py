"""Byzantine update attacks for reproducible FAR stress tests.

Attacks are applied by the simulator *after* collecting a round's updates.
This is intentional: ALIE, IPM, Min-Max and Min-Sum are attacker-knowledge
models that construct malicious vectors from honest submissions.  Keeping
them here avoids pretending that an honest worker would possess that view.
"""

from .byzantine import apply_attack, apply_configured_attack

__all__ = ["apply_attack", "apply_configured_attack"]
