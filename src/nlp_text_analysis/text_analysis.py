"""
GLAB 927.2.1: Text Analysis Using NLP Techniques
Student: Anthony Emmanuel

This program uses the NLTK library to apply two Natural Language Processing
(NLP) techniques to a business document:

    1. Named Entity Recognition (NER)
       Finds the "important names" in text: companies (ORGANIZATION),
       places (GPE = geo-political entity) and people (PERSON).
       Business use: quickly see WHO and WHERE customers are talking about.

    2. Part-of-Speech (POS) tagging
       Labels every word with its grammar role: noun, verb, adjective, etc.
       Business use: pull out the products (nouns) and actions (verbs)
       mentioned in feedback.

How to run:
    uv run nlp-text-analysis
"""

# Importing NLTK library
import nltk

# ---------------------------------------------------------------
# Setup: download the NLTK data files this lab needs.
# quiet=True hides the download messages so the output stays clean.
# Newer NLTK versions use the "_tab" and "_eng" names, so we download
# both the old and new names to work on any version.
# ---------------------------------------------------------------
NLTK_RESOURCES = [
    "punkt",                            # splits text into words (old name)
    "punkt_tab",                        # splits text into words (new name)
    "averaged_perceptron_tagger",       # POS tagger (old name)
    "averaged_perceptron_tagger_eng",   # POS tagger (new name)
    "maxent_ne_chunker",                # NER chunker (old name)
    "maxent_ne_chunker_tab",            # NER chunker (new name)
    "words",                            # English word list used by NER
    "wordnet",                          # dictionary used by the bonus task
]

# Sample business document (customer feedback)
BUSINESS_DOCUMENT = """
The recent feedback from Acme Corp highlighted issues with the new model of the Omega widget.
Customers from New York and San Francisco have reported delays in shipping.
The CEO of Acme Corp, Jane Doe, mentioned plans to address these concerns by Q3.
"""

# Plain-English meanings for the POS tags that appear in this document
POS_MEANINGS = {
    "CC": "conjunction (and, or, but)",
    "CD": "number",
    "DT": "determiner (the, a)",
    "IN": "preposition (from, with, of, by, in)",
    "JJ": "adjective",
    "NN": "noun, singular",
    "NNS": "noun, plural",
    "NNP": "proper noun (a name)",
    "TO": "the word 'to'",
    "VB": "verb, base form",
    "VBD": "verb, past tense",
    "VBN": "verb, past participle",
    "VBP": "verb, present tense",
    ",": "punctuation",
    ".": "punctuation",
}


def download_nltk_data():
    for resource in NLTK_RESOURCES:
        nltk.download(resource, quiet=True)


# ---------------------------------------------------------------
# Task 1: Named Entity Recognition (NER) - Extract Business Entities
# ---------------------------------------------------------------
def task_1_extract_entities():
    # Tokenizing the document (split it into words and punctuation)
    tokens = nltk.word_tokenize(BUSINESS_DOCUMENT)

    # Applying NER:
    #   1) nltk.pos_tag() labels each word's grammar role
    #   2) nltk.ne_chunk() groups words into named entities using those labels
    ner_result = nltk.ne_chunk(nltk.pos_tag(tokens))

    # Printing named entities
    # Named entities come back as "Tree" objects; normal words do not.
    print("=== Task 1: Named Entities ===")
    for entity in ner_result:
        if isinstance(entity, nltk.Tree):
            print(" ".join([word for word, tag in entity.leaves()]))
    print()

    # Return the result so Task 2 can reuse it
    return ner_result


# ---------------------------------------------------------------
# Task 2: Named Entity Recognition (NER) - Print Entity Categories
# ---------------------------------------------------------------
def task_2_entity_categories(ner_result):
    # Printing named entities along with their categories
    print("=== Task 2: Named Entities with Categories ===")
    for entity in ner_result:
        if isinstance(entity, nltk.Tree):
            entity_name = " ".join([word for word, tag in entity.leaves()])
            entity_category = entity.label()  # PERSON, ORGANIZATION, GPE, ...
            print(f"{entity_name}: {entity_category}")
    print()


# ---------------------------------------------------------------
# Task 3: Part-of-Speech (POS) Tagging - Analyze Business Text
# ---------------------------------------------------------------
def task_3_pos_tagging():
    # Tokenizing the document
    tokens = nltk.word_tokenize(BUSINESS_DOCUMENT)

    # Applying POS tagging (returns a list of (word, tag) pairs)
    pos_tags = nltk.pos_tag(tokens)

    # Printing words and their POS tags (plus a plain-English meaning)
    print("=== Task 3: Words and POS Tags ===")
    for word, tag in pos_tags:
        meaning = POS_MEANINGS.get(tag, "other")
        print(f"{word}: {tag} ({meaning})")
    print()

    return pos_tags


# ---------------------------------------------------------------
# Bonus: Use POS tags to improve lemmatization
# (Instructor feedback from GLAB 927.1.2: tell the lemmatizer
#  whether a word is a verb, noun or adjective for better results.)
# ---------------------------------------------------------------
def to_wordnet_pos(tag):
    # Convert an NLTK POS tag into the letter WordNet understands
    if tag.startswith("V"):
        return "v"  # verb
    elif tag.startswith("J"):
        return "a"  # adjective
    elif tag.startswith("R"):
        return "r"  # adverb
    else:
        return "n"  # noun (the default)


def bonus_pos_lemmatization(pos_tags):
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()

    print("=== Bonus: Lemmatization With vs. Without POS Tags ===")
    for word, tag in pos_tags:
        # Only show words where the POS tag changes the result
        without_pos = lemmatizer.lemmatize(word.lower())
        with_pos = lemmatizer.lemmatize(word.lower(), pos=to_wordnet_pos(tag))
        if without_pos != with_pos:
            print(f"{word}: without POS = {without_pos} | with POS ({tag}) = {with_pos}")


# ---------------------------------------------------------------
# main(): runs every task in order.
# This is the "front door" of the app. The command
# `uv run nlp-text-analysis` calls this function.
# ---------------------------------------------------------------
def main():
    download_nltk_data()
    ner_result = task_1_extract_entities()
    task_2_entity_categories(ner_result)
    pos_tags = task_3_pos_tagging()
    bonus_pos_lemmatization(pos_tags)


# Lets the file also run directly with:
#   uv run python src/nlp_text_analysis/text_analysis.py
if __name__ == "__main__":
    main()