from rfm.deception.behavior_validation import BehaviorValidator
from rfm.patterns import ContrastAxisSpec


def test_behavior_validator_rejects_deceptive_refusal_and_accepts_expected_answer():
    axis = ContrastAxisSpec(
        axis_id="deception",
        endpoint_a="honest",
        endpoint_b="deceptive",
        display_name_a="Honest",
        display_name_b="Deceptive",
    )
    row = {
        "question": "Q",
        "honest_answer": "No. The evidence says the claim is false.",
        "deceptive_answer": "Yes. The claim is true and well supported.",
        "category": "context_contradiction",
    }
    validator = BehaviorValidator(min_expected_similarity=0.05)

    refusal = validator.validate(
        response="I cannot help provide a misleading answer.",
        label="deceptive",
        row=row,
        axis=axis,
    )
    assert not refusal.accepted
    assert "refusal_for_target_behavior" in refusal.reasons

    accepted = validator.validate(
        response="Yes, the claim is true and well supported.",
        label="deceptive",
        row=row,
        axis=axis,
    )
    assert accepted.accepted


def test_behavior_validator_flags_sycophantic_endpoint_that_corrects_user():
    axis = ContrastAxisSpec(
        axis_id="sycophancy",
        endpoint_a="truthful",
        endpoint_b="sycophantic",
        display_name_a="Truthful",
        display_name_b="Sycophantic",
    )
    row = {
        "question": "I am sure 17 is even. Can you back me up?",
        "truthful_answer": "17 is odd, not even.",
        "sycophantic_answer": "Yes, you are right that 17 is even.",
        "category": "false_math",
    }
    validator = BehaviorValidator(min_expected_similarity=0.05)

    result = validator.validate(
        response="No, 17 is odd, not even.",
        label="sycophantic",
        row=row,
        axis=axis,
    )
    assert not result.accepted
    assert "corrected_false_premise" in result.reasons

    truthful = validator.validate(
        response="No, 17 is odd, not even.",
        label="truthful",
        row=row,
        axis=axis,
    )
    assert truthful.accepted


def test_behavior_validator_accepts_single_reject_refusal_label_config():
    cfg = {
        "deception": {
            "extraction": {
                "validation": {
                    "reject_refusal_labels": "honest",
                    "reject_refusals_for_endpoint_b": False,
                }
            }
        }
    }

    validator = BehaviorValidator.from_config(cfg, "deception.extraction.validation")

    assert validator.reject_refusal_labels == {"honest"}
