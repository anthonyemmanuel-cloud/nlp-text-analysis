# Text Analysis Using NLP Techniques

**GLAB 927.2.1: Text Analysis Using NLP Techniques**
Per Scholas AI Prompt Engineering (2026-CAX-255) · Anthony Emmanuel

## What this app does

This program uses Python's **NLTK** library to analyze a business document (customer feedback) with two Natural Language Processing techniques:

- **Named Entity Recognition (NER)** finds important names in the text and sorts them into categories: **PERSON**, **ORGANIZATION** and **GPE** (geo-political entity, such as cities and countries).
- **Part-of-Speech (POS) tagging** labels every word with its grammar role, such as noun, verb or adjective.
- **Bonus:** uses POS tags to improve lemmatization (`highlighted` → `highlight`), following instructor feedback from GLAB 927.1.2.

## What problem it solves

Companies receive large amounts of feedback, emails and reports. NER pulls out **who** and **where** (companies, people and locations) automatically. POS tagging pulls out **what** (products, the nouns) and **what happened** (actions, the verbs). Together they turn messy text into organized information a business can act on.

## Technologies used

- Python 3.13+
- uv for project and dependency management
- NLTK: `word_tokenize`, `pos_tag`, `ne_chunk`, `WordNetLemmatizer`
- NumPy (required by NLTK's named-entity chunker)

## How to install dependencies

```bash
uv sync
```

## How to run

```bash
uv run nlp-text-analysis
```

The first run downloads small NLTK data files, so it needs an internet connection.

## Results

**Named entities found by NLTK:**

| Entity | NLTK category | Correct category |
|---|---|---|
| Acme Corp | PERSON | ORGANIZATION ❌ |
| Omega | GPE | Product ❌ |
| New York | GPE | GPE ✅ |
| San Francisco | PERSON | GPE ❌ |
| CEO of Acme Corp | ORGANIZATION | Job title + organization ⚠️ |
| Jane Doe | PERSON | PERSON ✅ |

**POS tagging** correctly labeled every word. For example: `feedback` = NN (noun), `highlighted` = VBD (past-tense verb), `recent` = JJ (adjective), `Acme` = NNP (proper noun).

## What I learned

- NLTK's NER found all the important names, but it got **3 of 6 categories wrong**. Its built-in model is older and simple, and it has never heard of "Acme Corp" or the "Omega" widget. For real business use, a modern model (like spaCy) or a custom-trained model would be more accurate, and a person should review the results.
- POS tagging was very accurate, and POS tags make other tools better. For example, telling the lemmatizer that `reported` is a verb correctly turns it into `report`.