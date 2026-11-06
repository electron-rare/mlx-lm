# Copyright © 2026 Apple Inc.

import unittest

import mlx.core as mx

from mlx_lm.models.gpt_oss import Model, ModelArgs

P = "model.layers.0.mlp.experts."


def _model(D, I, E):
    return Model(
        ModelArgs(
            num_hidden_layers=2,
            layer_types=["sliding_attention", "full_attention"],
            hidden_size=D,
            intermediate_size=I,
            num_local_experts=E,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            vocab_size=16,
        )
    )


def _hf_weights(E, D, I, gate_up=None, down=None):
    return {
        P
        + "gate_up_proj": (gate_up if gate_up is not None else mx.zeros((E, D, 2 * I))),
        P + "down_proj": down if down is not None else mx.zeros((E, I, D)),
        P + "gate_up_proj_bias": mx.zeros((E, 2 * I)),
        P + "down_proj_bias": mx.zeros((E, D)),
    }


def _down_pattern(E, I, D):
    # Unique value per (intermediate, hidden) cell
    v = mx.arange(I * D, dtype=mx.float32).reshape(I, D)
    return mx.broadcast_to(v, (E, I, D))


class TestGptOssSanitize(unittest.TestCase):
    def test_hf_layout_shapes(self):
        E, D, I = 4, 10, 6
        s = _model(D, I, E).sanitize(_hf_weights(E, D, I))
        self.assertEqual(s[P + "gate_proj.weight"].shape, (E, I, D))
        self.assertEqual(s[P + "up_proj.weight"].shape, (E, I, D))
        self.assertEqual(s[P + "down_proj.weight"].shape, (E, D, I))
        self.assertEqual(s[P + "gate_proj.bias"].shape, (E, I))
        self.assertEqual(s[P + "up_proj.bias"].shape, (E, I))
        self.assertEqual(s[P + "down_proj.bias"].shape, (E, D))

    def test_hf_layout_interleaved_split_values(self):
        # gate_up is (E, D, 2I) with gate/up interleaved on the last axis
        E, D, I = 2, 4, 3
        gu = mx.zeros((E, D, 2 * I))
        gu[..., ::2] = 1.0
        gu[..., 1::2] = 2.0
        s = _model(D, I, E).sanitize(_hf_weights(E, D, I, gate_up=gu))
        self.assertTrue(mx.all(s[P + "gate_proj.weight"] == 1.0).item())
        self.assertTrue(mx.all(s[P + "up_proj.weight"] == 2.0).item())

        gu = mx.arange(E * D * 2 * I, dtype=mx.float32).reshape(E, D, 2 * I)
        s = _model(D, I, E).sanitize(_hf_weights(E, D, I, gate_up=gu))
        self.assertTrue(
            mx.array_equal(
                s[P + "gate_proj.weight"], gu[..., ::2].swapaxes(-1, -2)
            ).item()
        )
        self.assertTrue(
            mx.array_equal(
                s[P + "up_proj.weight"], gu[..., 1::2].swapaxes(-1, -2)
            ).item()
        )

    def test_already_sanitized_passthrough(self):
        E, D, I = 2, 4, 3
        weights = {P + "gate_proj.weight": mx.zeros((E, I, D))}
        s = _model(D, I, E).sanitize(weights)
        self.assertIs(s, weights)

    def test_hf_down_proj_values_unequal_dims(self):
        E, D, I = 2, 5, 3
        dp = _down_pattern(E, I, D)
        s = _model(D, I, E).sanitize(_hf_weights(E, D, I, down=dp))
        out = s[P + "down_proj.weight"]
        self.assertEqual(out.shape, (E, D, I))
        self.assertTrue(mx.array_equal(out, dp.swapaxes(-1, -2)).item())

    def test_hf_down_proj_values_equal_dims(self):
        # I == D is the default gpt-oss config: down_proj is shape-ambiguous
        # so the transpose must be driven by the detected layout.
        E, D, I = 2, 4, 4
        dp = _down_pattern(E, I, D)
        s = _model(D, I, E).sanitize(_hf_weights(E, D, I, down=dp))
        out = s[P + "down_proj.weight"]
        self.assertEqual(out.shape, (E, D, I))
        self.assertTrue(mx.array_equal(out, dp.swapaxes(-1, -2)).item())
        self.assertFalse(mx.array_equal(out, dp).item())

    def test_openai_layout_preserved(self):
        # openai layout: gate_up_proj (E, 2I, D), down_proj (E, D, I)
        for D, I in ((4, 4), (5, 3)):
            E = 2
            gu = mx.arange(E * 2 * I * D, dtype=mx.float32).reshape(E, 2 * I, D)
            dp = mx.arange(E * D * I, dtype=mx.float32).reshape(E, D, I)
            weights = {
                P + "gate_up_proj": gu,
                P + "down_proj": dp,
                P + "gate_up_proj_bias": mx.zeros((E, 2 * I)),
                P + "down_proj_bias": mx.zeros((E, D)),
            }
            s = _model(D, I, E).sanitize(weights)
            self.assertTrue(mx.array_equal(s[P + "gate_proj"], gu[..., ::2, :]).item())
            self.assertTrue(mx.array_equal(s[P + "up_proj"], gu[..., 1::2, :]).item())
            self.assertTrue(mx.array_equal(s[P + "down_proj"], dp).item())
            self.assertEqual(s[P + "gate_proj.bias"].shape, (E, I))

    def test_hf_bias_values_and_dtype(self):
        E, D, I = 2, 4, 4
        w = _hf_weights(E, D, I)
        gb = mx.arange(E * 2 * I).reshape(E, 2 * I)
        db = mx.arange(E * D).reshape(E, D)
        w[P + "gate_up_proj_bias"] = gb
        w[P + "down_proj_bias"] = db
        w = {k: v.astype(mx.bfloat16) for k, v in w.items()}
        gb, db = gb.astype(mx.bfloat16), db.astype(mx.bfloat16)
        s = _model(D, I, E).sanitize(w)
        self.assertTrue(mx.array_equal(s[P + "gate_proj.bias"], gb[..., ::2]).item())
        self.assertTrue(mx.array_equal(s[P + "up_proj.bias"], gb[..., 1::2]).item())
        self.assertTrue(mx.array_equal(s[P + "down_proj.bias"], db).item())
        for k, v in s.items():
            self.assertEqual(v.dtype, mx.bfloat16, k)

    def test_openai_layout_with_weight_suffix_preserved(self):
        # openai layout with an explicit ".weight" suffix on the input keys
        for D, I in ((4, 4), (5, 3)):
            E = 2
            gu = mx.arange(E * 2 * I * D, dtype=mx.float32).reshape(E, 2 * I, D)
            dp = mx.arange(E * D * I, dtype=mx.float32).reshape(E, D, I)
            weights = {
                P + "gate_up_proj.weight": gu,
                P + "down_proj.weight": dp,
                P + "gate_up_proj_bias": mx.zeros((E, 2 * I)),
                P + "down_proj_bias": mx.zeros((E, D)),
            }
            s = _model(D, I, E).sanitize(weights)
            self.assertTrue(
                mx.array_equal(s[P + "gate_proj.weight"], gu[..., ::2, :]).item()
            )
            self.assertTrue(
                mx.array_equal(s[P + "up_proj.weight"], gu[..., 1::2, :]).item()
            )
            self.assertTrue(mx.array_equal(s[P + "down_proj.weight"], dp).item())
            self.assertEqual(s[P + "gate_proj.bias"].shape, (E, I))

    def test_mxfp4_keys_not_detected_as_hf(self):
        E = 2
        u8 = mx.uint8
        for D, I in ((64, 64), (64, 128)):
            with self.subTest(D=D, I=I):
                gate_up_shape = (E, 2 * I, D // 32, 16)
                weights = {
                    P + "gate_up_proj_blocks": mx.zeros(gate_up_shape, dtype=u8),
                    P + "gate_up_proj_scales": mx.zeros((E, 2 * I, D // 32), dtype=u8),
                    P + "down_proj_blocks": mx.zeros((E, D, I // 32, 16), dtype=u8),
                    P + "down_proj_scales": mx.zeros((E, D, I // 32), dtype=u8),
                    P + "gate_up_proj_bias": mx.zeros((E, 2 * I)),
                    P + "down_proj_bias": mx.zeros((E, D)),
                }
                s = _model(D, I, E).sanitize(weights)
                # blocks: uint8 (.., K // 32, 16) -> view uint32 (.., K // 32, 4)
                # -> flatten gives K // 32 * 4 packed words per row, with K = D
                # for gate/up rows and K = I for down rows.
                expected = {
                    P + "gate_proj.weight": (E, I, D // 32 * 4),
                    P + "up_proj.weight": (E, I, D // 32 * 4),
                    P + "down_proj.weight": (E, D, I // 32 * 4),
                    P + "gate_proj.scales": (E, I, D // 32),
                    P + "up_proj.scales": (E, I, D // 32),
                    P + "down_proj.scales": (E, D, I // 32),
                    P + "gate_proj.bias": (E, I),
                    P + "up_proj.bias": (E, I),
                    P + "down_proj.bias": (E, D),
                }
                self.assertEqual({k: v.shape for k, v in s.items()}, expected)


if __name__ == "__main__":
    unittest.main()
