import nltk
nltk.download('stopwords')
import sys
import math
import os
import string

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    mapped_files = {}
    for file in os.listdir(directory):
        with open(os.path.join(directory, file), encoding="utf8") as f:
            mapped_files[file] = f.read()
    return mapped_files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    terms = nltk.word_tokenize(document.lower())
    final_document = []
    for term in terms:
        if term not in nltk.corpus.stopwords.words("english") and term not in string.punctuation:
            final_document.append(term)

    return final_document


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idfs = dict()
    terms = set()
    all_docs = len(documents)
    for _file in documents:
        terms.update(set(documents[_file]))

    for term in terms:
        val = sum(term in documents[file] for file in documents)
        idf = math.log(all_docs / val)
        idfs[term] = idf
    return idfs


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idfs = []
    for name in files:
        tf_idf = 0
        for i in query:
            tf_idf += idfs[i] * files[name].count(i)
        tf_idfs.append((name, tf_idf))
    tf_idfs.sort(key=lambda k: k[1], reverse=True)
    return [x[0] for x in tf_idfs[:n]]


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    response = []
    for i in sentences:
        idf = 0
        queries_matched = 0
        for term in query:
            if term in sentences[i]:
                queries_matched += 1
                idf += idfs[term]
        term_density = float(queries_matched) / len(sentences[i])
        response.append((i, idf, term_density))
    response.sort(key=lambda k: (k[1], k[2]), reverse=True)
    return [x[0] for x in response[:n]]


if __name__ == "__main__":
    main()
