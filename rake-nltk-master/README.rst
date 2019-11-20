rake-nltk
=========


RAKE short for Rapid Automatic Keyword Extraction algorithm, is a domain
independent keyword extraction algorithm which tries to determine key
phrases in a body of text by analyzing the frequency of word appearance
and its co-occurance with other words in the text.

|Demo|

Setup
-----

Using pip
~~~~~~~~~

.. code:: bash

    pip install rake-nltk

Directly from the repository
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

    git clone https://github.com/csurfer/rake-nltk.git
    python rake-nltk/setup.py install

Quick Start
-----------

.. code:: python

    from rake_nltk import Rake

    # Uses stopwords for english from NLTK, and all puntuation characters by
    # default
    r = Rake()

    # Extraction given the text.
    r.extract_keywords_from_text(<text to process>)

    # Extraction given the list of strings where each string is a sentence.
    r.extract_keywords_from_sentences(<list of sentences>)

    # To get keyword phrases ranked highest to lowest.
    r.get_ranked_phrases()

    # To get keyword phrases ranked highest to lowest with scores.
    r.get_ranked_phrases_with_scores()

Debugging Setup
---------------

If you see a stopwords error, it means that you do not have the corpus
`stopwords` downloaded from NLTK. You can download it using command below.

.. code:: bash

    python -c "import nltk; nltk.download('stopwords')"

