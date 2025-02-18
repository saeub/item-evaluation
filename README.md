# Evaluation of multiple-choice reading comprehension items

This repository contains code for automatically evaluating multiple-choice reading comprehension items by letting QA models respond to the items with and without seeing the text. It currently supports rule-based baselines and several generative LLMs (OpenAI, Llama 2, etc.) with pre-defined prompts for German and English.

**Paper:** Andreas Säuberli and Simon Clematide. 2024. [Automatic Generation and Evaluation of Reading Comprehension Test Items with Large Language Models](https://aclanthology.org/2024.readi-1.3/). In *Proceedings of the 3rd Workshop on Tools and Resources for People with REAding DIfficulties (READI) @ LREC-COLING 2024*, pages 22–37, Torino, Italia. ELRA and ICCL.

The code requires Python >= 3.10 and the dependencies in `requirements.txt`.

## Data format

Texts, questions, and answers are stored in JSONL files, where each line is a JSON object corresponding to a single article:

```json
{
    "text": "The text.",
    "items": [
        {
            "question": "The question?",
            "answers": [
                {
                    "text": "The answer!",
                    "correct": true
                }
            ],
            "multiple": false
        }
    ],
    "metadata": {
        "dataset": "dwlg",
        "split": "test",
        "extra": {
            "everything": "else",
            "goes": "here"
        }
    }
}
```

## Output format

QA model outputs are added to the data format above and stored in separate JSONL files:

```json
{
    "text": "The text.",
    "items": [
        {
            "question": "The question?",
            "answers": [
                {
                    "text": "The answer!",
                    "correct": true,
                    "prediction": {
                        "pred_correct": true,
                        "prob_correct": 0.9,  // Probability of this answer being correct
                        "model_output": "yes",  // For text generation models like Llama
                        "logprobs": {  // For text generation models like Llama
                            "yes": -1.4,
                            "no": -5.8
                        }
                    }
                }
            ],
            "multiple": false
        }
    ],
    "metadata": {
        "dataset": "dwlg",
        "split": "test",
        "extra": {
            "everything": "else",
            "goes": "here"
        }
    }
}
```

## Command-line interface

### Predicting

```
usage: cli.py predict [-h] --model {MajorityBaseline,ExactMatcher,FuzzyMatcher,OpenAIModel,LLMAPIModel} [--kwargs [KWARGS ...]] [--guess] [file]

positional arguments:
  file

options:
  -h, --help            show this help message and exit
  --model {MajorityBaseline,ExactMatcher,FuzzyMatcher,OpenAIModel,LLMAPIModel}
  --kwargs [KWARGS ...]
  --guess
```

The following command predicts responses using GPT-4 by asking it about the correctness of each answer option separately (`true_false=True`), and allowing for multiple correct answer options (`multi=True`).

```bash
python cli.py predict data/belebele/deu_Latn.jsonl --model OpenAIModel --kwargs model=gpt-4-0613 lang=de chat=True true_false=True multi=True > predictions.jsonl
```

See `models.py` for all available model classes and corresponding `kwargs` options. `LLMAPIModel` refers to a model served using the API in this repository: [`llm-api`](https://github.com/saeub/llm-api).

### Evaluating

```
usage: cli.py evaluate [-h] [files ...]

positional arguments:
  files       prediction output files

options:
  -h, --help  show this help message and exit
```

The following command calculates the accuracy of predictions generated by the previous command on the level of answer opions (i.e., every correctly predicted correct or incorrect answer option is counted).

```bash
python cli.py evaluate predictions.jsonl
```

## Unit tests

Run `pytest` to test model and prompting code.

## Data licenses

- `data/belebele/`: [Belebele](https://github.com/facebookresearch/belebele) (© Meta Platforms, Inc.) is licensed under [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- `data/qa4mre/`: [QA4MRE](http://nlp.uned.es/clef-qa/repository/pastCampaigns.php) ([Peñas et al., 2013](https://doi.org/10.1007/978-3-642-40802-1_29)) is licensed under [CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Citation

If you use this repository, please cite the following paper:

```bibtex
@inproceedings{sauberli-clematide-2024-automatic,
    title = "Automatic Generation and Evaluation of Reading Comprehension Test Items with Large Language Models",
    author = {S{\"a}uberli, Andreas  and
      Clematide, Simon},
    editor = "Wilkens, Rodrigo  and
      Cardon, R{\'e}mi  and
      Todirascu, Amalia  and
      Gala, N{\'u}ria",
    booktitle = "Proceedings of the 3rd Workshop on Tools and Resources for People with REAding DIfficulties (READI) @ LREC-COLING 2024",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.readi-1.3/",
    pages = "22--37"
}
```
