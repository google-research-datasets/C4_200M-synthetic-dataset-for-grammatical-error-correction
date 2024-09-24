"""Main API for mERRANT."""

from typing import Optional, Sequence

from merrant import classification
from merrant import utils


class Annotator:
  """Main interface to mERRANT.

  Example usage:
    annotator = api.Annotator("en_core_web_sm-3.0.0a1", aspell_lang="en")
    annotator.initialize()
    annotation = annotator.annotate("I goed to the storr."
                                    ["I went to the store."])

  The returned `utils.Annotation` contains tagged `utils.EditsSpans`. If
  `aspell_lang` is not set, no spell checker will be used. Edits can still be
  classified as `SPELL` based on character Levenshtein distance.
  """

  def __init__(self, spacy_model: str, aspell_lang: Optional[str] = None):
    self._spacy_model = spacy_model
    self._aspell_lang = aspell_lang
    self._initialized = False
    self._nlp = None
    self._classifier = None

  def initialize(self):
    """Initialized the interface. Must be called before `annotate()`."""
    self._nlp = utils.load_spacy_from_google3(self._spacy_model)
    self._classifier = classification.GenericClassifier(
        aspell_lang=self._aspell_lang)
    self._classifier.initialize()
    self._initialized = True

  def annotate(self, source_sentence: str,
               target_sentences: Sequence[str]) -> utils.Annotation:
    """Annotates the edits between a source- and a set of target sentences.

    Args:
      source_sentence: Untokenized source (original) sentence.
      target_sentences: A list of untokenized target (corrected) sentences.

    Returns:
      An `utils.Annotation` with tagged edit spans.
    """
    if not self._initialized:
      raise ValueError("Annotator not initialized.")

    if isinstance(target_sentences, str):
      raise ValueError("target_sentences must be a list, not a string.")

    source_doc = self._nlp(source_sentence)
    annotation = utils.Annotation(source_doc=source_doc)
    for target_sentence in target_sentences:
      target_doc = self._nlp(target_sentence)
      annotation.target_sentences.append(
          utils.TargetSentence(
              doc=target_doc,
              edit_spans=self._classifier.classify(source_doc, target_doc)))
    return annotation

