import numpy as np
import pytest

from pir_framework.real_ecg import compute_multilead_piv


def test_multilead_piv_matches_presence_weighted_definition():
    result = compute_multilead_piv(
        sqi=np.array([0.8, 0.6, 0.4, 0.2]),
        hr_bpm=np.array([130.0, 130.0, 130.0, 130.0]),
    )
    # SQI mean = 0.5, disagreement = 0, PIV = .75 * .5.
    assert float(result.sqi_combined) == pytest.approx(0.5)
    assert float(result.piv) == pytest.approx(0.375)
    assert float(result.disagreement_bpm) == pytest.approx(0.0)


def test_multilead_piv_penalizes_interlead_disagreement():
    agree = compute_multilead_piv(
        sqi=np.array([0.9, 0.9, 0.9, 0.9]),
        hr_bpm=np.array([130.0, 130.0, 130.0, 130.0]),
    )
    disagree = compute_multilead_piv(
        sqi=np.array([0.9, 0.9, 0.9, 0.9]),
        hr_bpm=np.array([110.0, 130.0, 150.0, 170.0]),
    )
    assert float(disagree.piv) < float(agree.piv)
    assert float(disagree.disagreement_bpm) > 0


def test_multilead_piv_excludes_unavailable_lead_from_quality_average():
    result = compute_multilead_piv(
        sqi=np.array([0.8, 0.6, 0.0, 1.0]),
        hr_bpm=np.array([130.0, 130.0, 160.0, 160.0]),
        present=np.array([True, True, False, False]),
    )
    assert float(result.sqi_combined) == pytest.approx(0.7)
    assert float(result.piv) == pytest.approx(0.525)


def test_multilead_piv_handles_batched_inputs_and_no_present_leads():
    result = compute_multilead_piv(
        sqi=np.array([[0.8, 0.6], [0.9, 0.4]]),
        hr_bpm=np.array([[130.0, 130.0], [140.0, 150.0]]),
        present=np.array([[True, True], [False, False]]),
    )
    assert result.piv.shape == (2,)
    assert result.piv[0] == pytest.approx(0.525)
    assert result.piv[1] == pytest.approx(0.0)
