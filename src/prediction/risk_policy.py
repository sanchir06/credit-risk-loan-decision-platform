"""
src/prediction/risk_policy.py
==============================

PURPOSE:
    Implements the demonstration risk policy for the UCI German Credit
    benchmark inference pipeline.

    Converts an estimated Probability of Default (PD) into:
        1. An internal risk score (0–100)
        2. A named risk tier (LOW / MEDIUM / HIGH)
        3. A loan decision (APPROVE / MANUAL REVIEW / REJECT)

DEMONSTRATION POLICY NOTICE:
    The thresholds defined here are DEMONSTRATION POLICY ASSUMPTIONS ONLY.

    They are NOT:
        - RBI (Reserve Bank of India) regulatory thresholds
        - Industry-standard lending cutoffs
        - Calibrated lending policy
        - Basel or IFRS9 compliant thresholds
        - Validated against real-world default outcomes

    These values exist solely for architectural and educational demonstration.

POLICY RATIONALE (for architectural clarity):
    PD < 0.20        → LOW risk   → APPROVE
    0.20 ≤ PD < 0.45 → MEDIUM risk → MANUAL REVIEW
    PD ≥ 0.45        → HIGH risk  → REJECT

DATASET CONTEXT:
    This policy is applied to predictions from the UCI Statlog German Credit
    Data benchmark model. The dataset is historical European banking data (1994)
    and does NOT represent Indian borrowers, Indian banking, CIBIL, or RBI data.

PHASE 5 — STEP 1
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Named constants — policy thresholds
# ---------------------------------------------------------------------------

# Maximum PD (exclusive) for the LOW risk tier
TIER_LOW_MAX_PD: float = 0.20

# Maximum PD (exclusive) for the MEDIUM risk tier
# PD >= this threshold → HIGH tier
TIER_MEDIUM_MAX_PD: float = 0.45

# Valid risk tiers
VALID_TIERS = frozenset({"LOW", "MEDIUM", "HIGH"})

# Valid decisions
VALID_DECISIONS = frozenset({"APPROVE", "MANUAL REVIEW", "REJECT"})

# Decision mapping: tier → decision
_TIER_TO_DECISION = {
    "LOW":    "APPROVE",
    "MEDIUM": "MANUAL REVIEW",
    "HIGH":   "REJECT",
}


# ---------------------------------------------------------------------------
# RiskPolicy — encapsulates the configurable policy
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RiskPolicy:
    """
    Encapsulates the demonstration risk policy parameters.

    Attributes:
        tier_low_max_pd    (float): PD below this → LOW tier. Default: 0.20
        tier_medium_max_pd (float): PD below this → MEDIUM tier. Default: 0.45
                                    PD at or above → HIGH tier.

    Usage:
        policy = RiskPolicy()
        tier   = policy.assign_tier(estimated_pd=0.23)
        score  = policy.compute_score(estimated_pd=0.23)
        decision = policy.assign_decision(tier)
    """

    tier_low_max_pd: float    = TIER_LOW_MAX_PD
    tier_medium_max_pd: float = TIER_MEDIUM_MAX_PD

    def compute_score(self, estimated_pd: float) -> float:
        """
        Compute the internal risk score from estimated PD.

        Formula:
            risk_score = (1 - estimated_pd) * 100

        Higher score = lower estimated risk.
        Score range: [0, 100]

        Args:
            estimated_pd (float): Estimated Probability of Default in [0, 1].

        Returns:
            float: Internal risk score in [0, 100].

        Raises:
            ValueError: If estimated_pd is outside [0, 1].
        """
        _validate_pd(estimated_pd)
        return round((1.0 - estimated_pd) * 100.0, 4)

    def assign_tier(self, estimated_pd: float) -> str:
        """
        Assign a named risk tier based on estimated PD.

        Thresholds (demonstration policy only — see module docstring):
            PD < 0.20        → "LOW"
            0.20 ≤ PD < 0.45 → "MEDIUM"
            PD ≥ 0.45        → "HIGH"

        Args:
            estimated_pd (float): Estimated Probability of Default in [0, 1].

        Returns:
            str: One of "LOW", "MEDIUM", "HIGH".

        Raises:
            ValueError: If estimated_pd is outside [0, 1].
        """
        _validate_pd(estimated_pd)

        if estimated_pd < self.tier_low_max_pd:
            return "LOW"
        elif estimated_pd < self.tier_medium_max_pd:
            return "MEDIUM"
        else:
            return "HIGH"

    def assign_decision(self, risk_tier: str) -> str:
        """
        Map a risk tier to a loan decision.

        Decision mapping (demonstration policy only):
            LOW    → "APPROVE"
            MEDIUM → "MANUAL REVIEW"
            HIGH   → "REJECT"

        Args:
            risk_tier (str): One of "LOW", "MEDIUM", "HIGH".

        Returns:
            str: One of "APPROVE", "MANUAL REVIEW", "REJECT".

        Raises:
            ValueError: If risk_tier is not a recognized tier.
        """
        if risk_tier not in VALID_TIERS:
            raise ValueError(
                f"Unknown risk tier: '{risk_tier}'. "
                f"Expected one of: {sorted(VALID_TIERS)}"
            )
        return _TIER_TO_DECISION[risk_tier]

    def evaluate(self, estimated_pd: float) -> dict:
        """
        Full policy evaluation: score + tier + decision in one call.

        Args:
            estimated_pd (float): Estimated Probability of Default in [0, 1].

        Returns:
            dict with keys: risk_score, risk_tier, decision
        """
        _validate_pd(estimated_pd)
        risk_score = self.compute_score(estimated_pd)
        risk_tier  = self.assign_tier(estimated_pd)
        decision   = self.assign_decision(risk_tier)
        return {
            "risk_score": risk_score,
            "risk_tier":  risk_tier,
            "decision":   decision,
        }


# ---------------------------------------------------------------------------
# Module-level convenience functions (using default policy)
# ---------------------------------------------------------------------------

_DEFAULT_POLICY = RiskPolicy()


def compute_risk_score(estimated_pd: float) -> float:
    """
    Compute internal risk score using the default demonstration policy.

    Args:
        estimated_pd (float): Estimated PD in [0, 1].

    Returns:
        float: Risk score in [0, 100].
    """
    return _DEFAULT_POLICY.compute_score(estimated_pd)


def assign_risk_tier(estimated_pd: float) -> str:
    """
    Assign risk tier using the default demonstration policy.

    Args:
        estimated_pd (float): Estimated PD in [0, 1].

    Returns:
        str: "LOW", "MEDIUM", or "HIGH".
    """
    return _DEFAULT_POLICY.assign_tier(estimated_pd)


def assign_decision(risk_tier: str) -> str:
    """
    Assign loan decision using the default demonstration policy.

    Args:
        risk_tier (str): One of "LOW", "MEDIUM", "HIGH".

    Returns:
        str: "APPROVE", "MANUAL REVIEW", or "REJECT".
    """
    return _DEFAULT_POLICY.assign_decision(risk_tier)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_pd(pd: float) -> None:
    """
    Raise ValueError if pd is not a valid probability in [0, 1].

    Args:
        pd: Value to validate.

    Raises:
        ValueError: If pd is NaN, infinite, or outside [0, 1].
        TypeError:  If pd is not a numeric type.
    """
    import math

    if not isinstance(pd, (int, float)):
        raise TypeError(
            f"estimated_pd must be a numeric value, got {type(pd).__name__}."
        )
    if math.isnan(pd):
        raise ValueError("estimated_pd is NaN — a valid probability is required.")
    if math.isinf(pd):
        raise ValueError("estimated_pd is infinite — a valid probability is required.")
    if not (0.0 <= pd <= 1.0):
        raise ValueError(
            f"estimated_pd must be in [0, 1], got {pd:.6f}."
        )
