import unittest

from api_server import _should_offer_stress_sensor


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


if __name__ == "__main__":
    unittest.main()
