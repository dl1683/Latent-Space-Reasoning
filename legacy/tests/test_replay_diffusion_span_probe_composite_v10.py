from experiments.replay_diffusion_span_probe_composite_v10 import _span_text_features


def test_span_text_features_parse_multiline_probe_slots():
    features = _span_text_features(
        "X0=missing cost\n"
        "X1=source score valid\n"
        "X2=source score preserved\n"
        "N=0"
    )

    assert features["x0_x2_slot_overlap"] == 0.0
    assert features["max_slot_overlap"] == 0.5
    assert features["repeated_token_excess"] == 2.0
