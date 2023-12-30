import re
from math import exp
from typing import Literal, Sequence

import jinja2

from data import Logprobs


class TrueFalseAnswerPrompter:
    _STRINGS = dict(
        true_label={
            "de": "R",
            "en": "C",
        },
        false_label={
            "de": "F",
            "en": "I",
        },
        guess_preface={
            "de": "Die folgende Frage und Antwort stammen aus einer Multiple-Choice-Verständnisaufgabe zu einem unbekannten Text.",
            "en": "The following question and answer are from a multiple-choice comprehension task about an unknown text.",
        },
        pre_text={
            "de": "Text:",
            "en": "Text:",
        },
        pre_question={
            "de": "Frage:",
            "en": "Question:",
        },
        pre_answer={
            "de": "Antwort:",
            "en": "Answer:",
        },
        task={
            "de": "Gemäß dem Text oben, ist diese Antwort richtig (R) oder falsch (F)? Gib nur den Buchstaben R oder F an.",
            "en": "Based on the text above, is this answer correct (C) or incorrect (I)? Indicate only the letter C or I.",
        },
        task_guess={
            "de": "Ohne den Text zu kennen, nur basierend auf Allgemeinwissen, ist es plausibler, dass die Antwort richtig (R) oder falsch (F) ist? Gib nur den Buchstaben R oder F an.",
            # "de": "Ohne den Text zu kennen, nur basierend auf Allgemeinwissen, ist es plausibler, dass die Antwort richtig (R) oder falsch (F) ist? Gib nur den Buchstaben R oder F an. Gib keine Begründung für deine Entscheidung an.",
            "en": "Without knowing the text, only based on general knowledge, is this answer more likely to be correct (C) or incorrect (I)? Indicate only the letter C or I.",
            # "en": "Without knowing the text, only based on general knowledge, is this answer more likely to be correct (C) or incorrect (I)? Indicate only the letter C or I. Do not justify your decision.",
        },
        pre_output={
            "de": "Output:",
            "en": "Output:",
        },
    )
    _TEMPLATE = jinja2.Template(
        """\
{% if guess %}
{{ S.guess_preface }}
{% else %}
{{ S.pre_text }}
{{ text }}
{% endif %}

{{ S.pre_question }} {{ question }}
{{ S.pre_answer }} {{ answer }}

{% if guess %}
{{ S.task_guess }}
{%- else %}
{{ S.task }}
{%- endif %}
{% if not chat %}


{{ S.pre_output }}
{% endif %}
""",
        trim_blocks=True,
        lstrip_blocks=True,
    )

    def __init__(
        self, lang: Literal["de", "en"], chat: bool = False, guess: bool = False
    ):
        self.lang = lang
        self.chat = chat
        self.guess = guess

    @property
    def true_label(self) -> str:
        return self._STRINGS["true_label"][self.lang]

    @property
    def false_label(self) -> str:
        return self._STRINGS["false_label"][self.lang]

    def build_prompt(self, text: str, question: str, answer: str) -> str:
        if not self.guess:
            text = text.replace("\n\n", "\n")
        S = {key: value[self.lang] for key, value in self._STRINGS.items()}
        return self._TEMPLATE.render(
            S=S,
            text=text,
            question=question,
            answer=answer,
            chat=self.chat,
            guess=self.guess,
        )

    def parse_output(
        self, output: str, logprobs: Logprobs | None
    ) -> tuple[bool | None, float | None]:
        labels = self.true_label + self.false_label
        if logprobs is not None:
            true_logprob = -float("inf")
            false_logprob = -float("inf")
            for logprob in logprobs:
                argmax = max(logprob, key=logprob.get)
                # Ignore leading whitespace
                if not argmax.isspace():
                    true_logprob = max(
                        logprob.get(self.true_label, -float("inf")),
                        logprob.get(" " + self.true_label, -float("inf")),
                    )
                    false_logprob = max(
                        logprob.get(self.false_label, -float("inf")),
                        logprob.get(" " + self.false_label, -float("inf")),
                    )
                    break
            pred_correct = true_logprob > false_logprob
            true_prob = exp(true_logprob)
            false_prob = exp(false_logprob)
            prob_correct = true_prob / (true_prob + false_prob)
            return pred_correct, prob_correct
        if match := re.match(rf"[\S\s]*?\b([{labels}])\b", output):
            return (match.group(1) == self.true_label), None
        return None, None


class PickAnswerPrompter:
    _STRINGS = dict(
        letters={
            "de": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "en": "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        },
        zero_label={
            "de": "0",
            "en": "0",
        },
        guess_preface={
            "de": "Die folgende Frage und Antworten stammen aus einer Multiple-Choice-Verständnisaufgabe zu einem unbekannten Text.",
            "en": "The following question and answer are from a multiple-choice comprehension task about an unknown text.",
        },
        pre_text={
            "de": "Text:",
            "en": "Text:",
        },
        pre_question={
            "de": "Frage:",
            "en": "Question:",
        },
        pre_answer={
            "de": "Antwort {letter}:",
            "en": "Answer {letter}:",
        },
        task={
            "de": "Welche Antwort ist gemäß dem Text oben richtig? Gib genau einen der Buchstaben {letters} an.",
            "en": "Based on the text above, which answer is correct? Indicate exactly one of the letters {letters}.",
        },
        task_guess={
            "de": "Ohne den Text zu kennen, nur basierend auf Allgemeinwissen, welche Antwort ist am plausibelsten? Gib genau einen der Buchstaben A, B an.",
            "en": "Without knowing the text, only based on general knowledge, which answer is most likely to be correct? Indicate exactly one of the letters {letters}.",
        },
        task_multi={
            "de": "Welche Antworten sind gemäß dem Text oben richtig? Es können keine, eine, oder mehrere Antworten richtig sein. Gib einen oder mehere der Buchstaben {letters} an. Gib 0 an, wenn keine der Antworten richtig ist.",
            "en": "Based on the text above, which answers are correct? Zero, one or several answers may be correct. Indicate one or more of the letters {letters}. Indicate 0 if none of the answers are correct.",
        },
        task_guess_multi={
            "de": "Ohne den Text zu kennen, nur basierend auf Allgemeinwissen, welche Antworten sind am plausibelsten? Es können keine, eine, oder mehrere Antworten richtig sein. Gib einen oder mehere der Buchstaben {letters} an. Gib 0 an, wenn keine der Antworten richtig ist.",
            "en": "Without knowing the text, only based on general knowledge, which answers are most likely to be correct? Zero, one or several answers may be correct. Indicate one or more of the letters {letters}. Indicate 0 if none of the answers are correct.",
        },
        pre_output={
            "de": "Output:",
            "en": "Output:",
        },
    )
    _TEMPLATE = jinja2.Template(
        """\
{% if guess %}
{{ S.guess_preface }}
{% else %}
{{ S.pre_text }}
{{ text }}
{% endif %}

{{ S.pre_question }} {{ question }}
{% for letter, answer in zip(letters, answers) %}
{{ S.pre_answer.format(letter=letter) }} {{ answer }}
{% endfor %}

{% if not guess and not multi %}
{{ S.task.format(letters=letters | join(", ")) }}
{%- elif guess and not multi %}
{{ S.task_guess.format(letters=letters | join(", ")) }}
{%- elif not guess and multi %}
{{ S.task_multi.format(letters=letters | join(", ")) }}
{%- elif guess and multi %}
{{ S.task_guess_multi.format(letters=letters | join(", ")) }}
{%- endif %}
{% if not chat %}


{{ S.pre_output }}
{% endif %}
""",
        trim_blocks=True,
        lstrip_blocks=True,
    )

    def __init__(
        self,
        lang: Literal["de", "en"],
        chat: bool = False,
        guess: bool = False,
        multi: bool = False,
    ):
        self.lang = lang
        self.chat = chat
        self.guess = guess
        self.multi = multi

    def get_labels(self, num_answers: int) -> list[str]:
        letters = self._STRINGS["letters"][self.lang]
        assert num_answers <= len(
            letters
        ), f"Too many answers ({num_answers}; max: {len(letters)})"
        return list(letters[:num_answers])

    def build_prompt(self, text: str, question: str, answers: Sequence[str]) -> str:
        if not self.guess:
            text = text.replace("\n\n", "\n")
        letters = self.get_labels(len(answers))
        S = {key: value[self.lang] for key, value in self._STRINGS.items()}
        return self._TEMPLATE.render(
            S=S,
            text=text,
            question=question,
            answers=answers,
            chat=self.chat,
            guess=self.guess,
            multi=self.multi,
            letters=letters,
            zip=zip,
        )

    def parse_output(
        self, output: str, logprobs: Logprobs | None, num_answers: int
    ) -> tuple[list[bool | None], list[float | None]]:
        letters = self.get_labels(num_answers)
        zero = self._STRINGS["zero_label"][self.lang]
        if self.multi:
            if matches := re.findall(rf"\b([{''.join(letters)}{zero}])\b", output):
                if zero in matches:
                    return [False] * num_answers, [None] * num_answers
                return (
                    [letter in matches for letter in letters],
                    [None] * num_answers,
                )
        else:
            if logprobs is not None:
                letter_logprobs = [-float("inf")] * num_answers
                for logprob in logprobs:
                    argmax = max(logprob, key=logprob.get)
                    # Ignore leading whitespace
                    if not argmax.isspace():
                        for i, letter in enumerate(letters):
                            letter_logprobs[i] = max(
                                logprob.get(letter, -float("inf")),
                                logprob.get(" " + letter, -float("inf")),
                            )
                        break
                best_letter = letters[letter_logprobs.index(max(letter_logprobs))]
                preds_correct = [letter == best_letter for letter in letters]
                letter_probs = [exp(logprob) for logprob in letter_logprobs]
                probs_correct = [
                    prob / sum(letter_probs) for prob in letter_probs
                ]
                return preds_correct, probs_correct
            if match := re.match(rf"[\S\s]*?\b([{''.join(letters)}{zero}])\b", output):
                return [letter == match.group(1) for letter in letters], [None] * num_answers
        return [None] * num_answers, [None] * num_answers
