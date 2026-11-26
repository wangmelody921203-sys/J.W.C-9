import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from api_server import app, _call_gemini_chat, _call_groq_chat_with_fallback, _should_offer_stress_sensor


def user_messages(*texts: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": text} for text in texts]


class StressSupportSignalTests(unittest.TestCase):
    def test_explicit_stress_statement_suggests_check_in(self) -> None:
        self.assertTrue(_should_offer_stress_sensor(user_messages("我最近壓力好大")))

    def test_repeated_distress_across_recent_user_turns_suggests_check_in(self) -> None:
        messages = user_messages("今天好煩", "我很累", "午餐吃什麼")
        self.assertTrue(_should_offer_stress_sensor(messages))

    def test_single_everyday_complaint_does_not_suggest_sensor(self) -> None:
        self.assertFalse(_should_offer_stress_sensor(user_messages("今天好煩")))

    def test_recent_crisis_signal_takes_precedence(self) -> None:
        messages = user_messages("我最近好累", "我想死")
        self.assertFalse(_should_offer_stress_sensor(messages))

    def test_assistant_messages_do_not_count_as_user_distress(self) -> None:
        messages = [
            {"role": "assistant", "content": "好煩"},
            {"role": "assistant", "content": "我很累"},
        ]
        self.assertFalse(_should_offer_stress_sensor(messages))


class GeminiTimeoutTests(unittest.TestCase):
    def test_request_has_bounded_timeout_and_disables_sdk_retries(self) -> None:
        client = Mock()
        client.models.generate_content.return_value.text = "我在這裡陪你。"
        client.models.generate_content.return_value.candidates = []
        with patch("api_server._get_gemini_client", return_value=client):
            reply, _ = _call_gemini_chat(messages=[{"role": "user", "content": "你好"}])

        self.assertEqual(reply, "我在這裡陪你。")
        http_options = client.models.generate_content.call_args.kwargs["config"]["http_options"]
        self.assertGreater(http_options["timeout"], 0)
        self.assertLessEqual(http_options["timeout"], 12000)
        self.assertEqual(http_options["retry_options"]["attempts"], 1)

    def test_network_timeout_is_not_retried(self) -> None:
        client = Mock()
        client.models.generate_content.side_effect = TimeoutError("read timed out")
        with patch("api_server._get_gemini_client", return_value=client):
            with self.assertRaises(TimeoutError):
                _call_gemini_chat(messages=[{"role": "user", "content": "你好"}])
        self.assertEqual(client.models.generate_content.call_count, 1)

    def test_generate_returns_explicit_error_and_stress_signal_on_timeout(self) -> None:
        client = Mock()
        client.models.generate_content.side_effect = TimeoutError("read timed out")
        with patch("api_server._check_rate", return_value=True), \
             patch("api_server._resolve_authed_user", return_value=(None, None, None)), \
             patch("api_server._get_gemini_client", return_value=client):
            response = app.test_client().post(
                "/generate",
                json={"messages": [{"role": "user", "content": "我最近壓力好大"}]},
            )

        self.assertEqual(response.status_code, 500)
        data = response.get_json()
        self.assertEqual(data["error"], "gemini_error")
        self.assertTrue(data["stress_support_suggested"])
        self.assertIn("fallback", data)

    def test_crisis_response_does_not_depend_on_llm_availability(self) -> None:
        with patch("api_server._check_rate", return_value=True), \
             patch("api_server._resolve_authed_user", return_value=(None, None, None)), \
             patch("api_server._get_gemini_client", return_value=None), \
             patch("api_server._get_groq_client", return_value=None):
            response = app.test_client().post(
                "/generate",
                json={"messages": [{"role": "user", "content": "我想死"}]},
            )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["safety_mode"], "crisis")
        self.assertFalse(data["stress_support_suggested"])


class CompleteReplyTests(unittest.TestCase):
    def test_gemini_retries_a_reply_ending_mid_sentence(self) -> None:
        client = Mock()
        client.models.generate_content.side_effect = [
            SimpleNamespace(text="辛苦你了。如果可以的話，要不要先試著喝", candidates=[]),
            SimpleNamespace(text="辛苦你了。如果可以的話，先喝口水吧。", candidates=[]),
        ]
        with patch("api_server._get_gemini_client", return_value=client), \
             patch("api_server.time.sleep"):
            reply, _ = _call_gemini_chat(messages=[{"role": "user", "content": "我好累"}])
        self.assertEqual(reply, "辛苦你了。如果可以的話，先喝口水吧。")
        self.assertEqual(client.models.generate_content.call_count, 2)

    def test_gemini_retries_token_limited_output_even_if_punctuated(self) -> None:
        client = Mock()
        client.models.generate_content.side_effect = [
            SimpleNamespace(text="我聽到了。", candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")]),
            SimpleNamespace(text="我聽到了，我陪你。", candidates=[SimpleNamespace(finish_reason="STOP")]),
        ]
        with patch("api_server._get_gemini_client", return_value=client), \
             patch("api_server.time.sleep"):
            reply, _ = _call_gemini_chat(messages=[{"role": "user", "content": "我好累"}])
        self.assertEqual(reply, "我聽到了，我陪你。")
        self.assertEqual(client.models.generate_content.call_count, 2)

    def test_groq_does_not_return_token_limited_output(self) -> None:
        client = Mock()
        client.chat.completions.create.return_value.choices = [
            SimpleNamespace(message=SimpleNamespace(content="辛苦你了。如果可以的話，要不要先試著喝"), finish_reason="length")
        ]
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            _call_groq_chat_with_fallback(client, messages=[{"role": "user", "content": "我好累"}])

    def test_incomplete_gemini_replies_are_not_returned_after_retries(self) -> None:
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(
            text="辛苦你了。如果可以的話，要不要先試著喝",
            candidates=[],
        )
        with patch("api_server._get_gemini_client", return_value=client), \
             patch("api_server.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "incomplete"):
                _call_gemini_chat(messages=[{"role": "user", "content": "我好累"}])


if __name__ == "__main__":
    unittest.main()
