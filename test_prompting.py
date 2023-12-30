from math import exp

import pytest

from prompting import PickAnswerPrompter, TrueFalseAnswerPrompter


@pytest.mark.parametrize(
    ("kwargs", "prompt"),
    [
        (
            {"lang": "de", "chat": False, "guess": False},
            """Text:
<PARAGRAPH 1>
<PARAGRAPH 2>

Frage: <QUESTION>
Antwort: <ANSWER>

Gemäß dem Text oben, ist diese Antwort richtig (R) oder falsch (F)? Gib nur den Buchstaben R oder F an.

Output:
""",
        ),
        (
            {"lang": "de", "chat": True, "guess": False},
            """Text:
<PARAGRAPH 1>
<PARAGRAPH 2>

Frage: <QUESTION>
Antwort: <ANSWER>

Gemäß dem Text oben, ist diese Antwort richtig (R) oder falsch (F)? Gib nur den Buchstaben R oder F an.""",
        ),
        (
            {"lang": "de", "chat": False, "guess": True},
            """Die folgende Frage und Antwort stammen aus einer Multiple-Choice-Verständnisaufgabe zu einem unbekannten Text.

Frage: <QUESTION>
Antwort: <ANSWER>

Ohne den Text zu kennen, nur basierend auf Allgemeinwissen, ist es plausibler, dass die Antwort richtig (R) oder falsch (F) ist? Gib nur den Buchstaben R oder F an.

Output:
""",
        ),
        (
            {"lang": "de", "chat": True, "guess": True},
            """Die folgende Frage und Antwort stammen aus einer Multiple-Choice-Verständnisaufgabe zu einem unbekannten Text.

Frage: <QUESTION>
Antwort: <ANSWER>

Ohne den Text zu kennen, nur basierend auf Allgemeinwissen, ist es plausibler, dass die Antwort richtig (R) oder falsch (F) ist? Gib nur den Buchstaben R oder F an.""",
        ),
    ],
)
def test_true_false_answer_prompter(kwargs, prompt):
    prompter = TrueFalseAnswerPrompter(**kwargs)
    # assert prompter.true_label == "R"
    # assert prompter.false_label == "F"
    assert (
        prompter.build_prompt(
            text="<PARAGRAPH 1>\n\n<PARAGRAPH 2>",
            question="<QUESTION>",
            answer="<ANSWER>",
        )
        == prompt
    )

    assert prompter.parse_output(f"{prompter.true_label}. ", None) == (True, None)
    assert prompter.parse_output(f"\n {prompter.false_label}", None) == (False, None)
    assert prompter.parse_output(f"oops", None) == (None, None)

    assert prompter.parse_output(
        f"foo", [{" ": -1.0}, {prompter.true_label: -1.0, prompter.false_label: -2.0}]
    ) == (True, exp(-1.0) / (exp(-1.0) + exp(-2.0)))


@pytest.mark.parametrize(
    ("kwargs", "prompt"),
    [
        (
            {"lang": "de", "chat": False, "guess": False, "multi": False},
            """Text:
<PARAGRAPH 1>
<PARAGRAPH 2>

Frage: <QUESTION>
Antwort A: <ANSWER 1>
Antwort B: <ANSWER 2>

Welche Antwort ist gemäß dem Text oben richtig? Gib genau einen der Buchstaben A, B an.

Output:
""",
        ),
        (
            {"lang": "de", "chat": True, "guess": False, "multi": False},
            """Text:
<PARAGRAPH 1>
<PARAGRAPH 2>

Frage: <QUESTION>
Antwort A: <ANSWER 1>
Antwort B: <ANSWER 2>

Welche Antwort ist gemäß dem Text oben richtig? Gib genau einen der Buchstaben A, B an.""",
        ),
        (
            {"lang": "de", "chat": False, "guess": True, "multi": False},
            """Die folgende Frage und Antworten stammen aus einer Multiple-Choice-Verständnisaufgabe zu einem unbekannten Text.

Frage: <QUESTION>
Antwort A: <ANSWER 1>
Antwort B: <ANSWER 2>

Ohne den Text zu kennen, nur basierend auf Allgemeinwissen, welche Antwort ist am plausibelsten? Gib genau einen der Buchstaben A, B an.

Output:
""",
        ),
        (
            {"lang": "de", "chat": False, "guess": True, "multi": False},
            """Die folgende Frage und Antworten stammen aus einer Multiple-Choice-Verständnisaufgabe zu einem unbekannten Text.

Frage: <QUESTION>
Antwort A: <ANSWER 1>
Antwort B: <ANSWER 2>

Ohne den Text zu kennen, nur basierend auf Allgemeinwissen, welche Antwort ist am plausibelsten? Gib genau einen der Buchstaben A, B an.

Output:
""",
        ),
        (
            {"lang": "de", "chat": False, "guess": False, "multi": True},
            """Text:
<PARAGRAPH 1>
<PARAGRAPH 2>

Frage: <QUESTION>
Antwort A: <ANSWER 1>
Antwort B: <ANSWER 2>

Welche Antworten sind gemäß dem Text oben richtig? Es können keine, eine, oder mehrere Antworten richtig sein. Gib einen oder mehere der Buchstaben A, B an. Gib 0 an, wenn keine der Antworten richtig ist.

Output:
""",
        ),
        (
            {"lang": "de", "chat": True, "guess": False, "multi": True},
            """Text:
<PARAGRAPH 1>
<PARAGRAPH 2>

Frage: <QUESTION>
Antwort A: <ANSWER 1>
Antwort B: <ANSWER 2>

Welche Antworten sind gemäß dem Text oben richtig? Es können keine, eine, oder mehrere Antworten richtig sein. Gib einen oder mehere der Buchstaben A, B an. Gib 0 an, wenn keine der Antworten richtig ist.""",
        ),
        (
            {"lang": "de", "chat": False, "guess": True, "multi": True},
            """Die folgende Frage und Antworten stammen aus einer Multiple-Choice-Verständnisaufgabe zu einem unbekannten Text.

Frage: <QUESTION>
Antwort A: <ANSWER 1>
Antwort B: <ANSWER 2>

Ohne den Text zu kennen, nur basierend auf Allgemeinwissen, welche Antworten sind am plausibelsten? Es können keine, eine, oder mehrere Antworten richtig sein. Gib einen oder mehere der Buchstaben A, B an. Gib 0 an, wenn keine der Antworten richtig ist.

Output:
""",
        ),
        (
            {"lang": "de", "chat": True, "guess": True, "multi": True},
            """Die folgende Frage und Antworten stammen aus einer Multiple-Choice-Verständnisaufgabe zu einem unbekannten Text.

Frage: <QUESTION>
Antwort A: <ANSWER 1>
Antwort B: <ANSWER 2>

Ohne den Text zu kennen, nur basierend auf Allgemeinwissen, welche Antworten sind am plausibelsten? Es können keine, eine, oder mehrere Antworten richtig sein. Gib einen oder mehere der Buchstaben A, B an. Gib 0 an, wenn keine der Antworten richtig ist.""",
        ),
    ],
)
def test_pick_answer_prompter(kwargs, prompt):
    prompter = PickAnswerPrompter(**kwargs)
    assert (
        prompter.build_prompt(
            text="<PARAGRAPH 1>\n\n<PARAGRAPH 2>",
            question="<QUESTION>",
            answers=["<ANSWER 1>", "<ANSWER 2>"],
        )
        == prompt
    )
    if kwargs.get("multi"):
        assert prompter.parse_output(f"A. ", None, 2) == ([True, False], [None, None])
        assert prompter.parse_output(f"\n B", None, 2) == ([False, True], [None, None])
        assert prompter.parse_output(f"\n Answer A, and B...", None, 2) == ([True, True], [None, None])
        assert prompter.parse_output(f"0", None, 2) == ([False, False], [None, None])
        assert prompter.parse_output(f"None of the above", None, 2) == ([None, None], [None, None])
    else:
        assert prompter.parse_output(f"A. ", None, 2) == ([True, False], [None, None])
        assert prompter.parse_output(f"\n B", None, 2) == ([False, True], [None, None])
        assert prompter.parse_output(f"\n A, not B...", None, 2) == ([True, False], [None, None])
        assert prompter.parse_output(f"None of the above", None, 2) == ([None, None], [None, None])
