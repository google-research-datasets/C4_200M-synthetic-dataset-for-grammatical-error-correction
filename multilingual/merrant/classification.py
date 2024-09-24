"""This module contains the rules for assigning tags to `EditSpans`."""

import abc
from typing import Iterator, Optional
import spacy

from merrant import utils
import aspell

class Classifier(metaclass=abc.ABCMeta):
  """Interface for classifiers that assign tags to EditSpans."""

  @abc.abstractmethod
  def classify_single_span(self, edit_span: utils.EditSpan) -> str:
    """Tags a single `utils.EditSpan`.

    Args:
      edit_span: The EditSpan to classify

    Returns:
      The error type tag.
    """

  def initialize(self) -> None:
    """Initializes the classifier. May be overridden by subclasses."""

  def classify(self, source_doc: spacy.tokens.doc.Doc,
               target_doc: spacy.tokens.doc.Doc) -> list[utils.EditSpan]:
    """Classifies all edit spans between a source and a target sentence.

    Args:
      source_doc: Source sentence annotated by the spaCy pipeline
      target_doc: Source sentence annotated by the spaCy pipeline

    Returns:
      A list of edit spans. Tag assignment is delegated to subclasses.
    """
    edit_spans = utils.get_edit_spans(source_doc, target_doc)
    for edit_span in edit_spans:
      edit_span.tag = self.classify_single_span(edit_span)
    return edit_spans


def _is_orth_error(source_tokens: ..., target_tokens: ...) -> bool:
  """Indicates if source and target differ only by casing or whitespace."""
  return ''.join([token.lower_ for token in source_tokens
                 ]) == ''.join([token.lower_ for token in target_tokens])


def _is_wo_error(source_tokens: ..., target_tokens: ...) -> bool:
  """Indicates if the source is a case-insensitive reordering of the target."""
  return set([token.lower_ for token in source_tokens
             ]) == set([token.lower_ for token in target_tokens])


def _is_punct_error(source_tokens: ..., target_tokens: ...) -> bool:
  """Classifies as PUNCT if non-punctuation tokens only change orthography."""

  def get_nonpunct_tokens(tokens) -> list[spacy.tokens.Token]:
    return [token for token in tokens if not token.is_punct]

  return _is_orth_error(
      get_nonpunct_tokens(source_tokens), get_nonpunct_tokens(target_tokens))


def _pos_is_subset(tokens: list[spacy.tokens.Token],
                   allowed_pos: set[str]) -> bool:
  """Returns True if the set of token POS tags is a subset of `allowed_pos`."""
  return all(token.pos_ in allowed_pos for token in tokens)


def _is_aux_span(tokens: list[spacy.tokens.Token]) -> bool:
  return _pos_is_subset(tokens, {'AUX'})


def _is_verbal_span(tokens: list[spacy.tokens.Token]) -> bool:
  return _pos_is_subset(tokens, {'AUX', 'PART', 'VERB'}) and any(
      token.pos_ == 'VERB' for token in tokens)


def _is_noun_span(tokens: list[spacy.tokens.Token]) -> bool:
  return _pos_is_subset(tokens, {'DET', 'NOUN'}) and any(
      token.pos_ == 'NOUN' for token in tokens)


def _match_up_pos_tokens(
    source_tokens: list[spacy.tokens.Token],
    target_tokens: list[spacy.tokens.Token],
    pos: str) -> Iterator[tuple[spacy.tokens.Token, spacy.tokens.Token]]:
  """Pairs up source and target tokens with same lemma and given POS tag."""
  pos_source_tokens = [token for token in source_tokens if token.pos_ == pos]
  pos_target_tokens = [token for token in target_tokens if token.pos_ == pos]

  for source_token, target_token in zip(pos_source_tokens, pos_target_tokens):
    if source_token.lemma == target_token.lemma:
      yield source_token, target_token


def _morph_match(source_token: spacy.tokens.Token,
                 target_token: spacy.tokens.Token, field_name: str) -> bool:
  """Returns True if the `field_name` morphology matches."""
  return ''.join(source_token.morph.get(field_name)) == ''.join(
      target_token.morph.get(field_name))


class GenericClassifier(Classifier):
  """Implements (fairly) language-independent rules for classifying edit spans.

  This classifier assigns error type tags to edit spans based on POS tags and
  aspell suggestions. The rules are loosely inspired by ERRANT:
    https://github.com/chrisjbryant/errant

  The error type tag can be one of the POS tags in the spaCy labelling
    scheme, or one of the following error types:
      - NUM: Error with numbers / number words
      - MORPH: 1:1 replacement with same lemma, but different spaCy tag.
      - NOUN:INFL: Incorrectly inflected noun.
      - NOUN:NUM: Changing singular/plural.
      - ORTH: Case/whitespace errors
      - PUNCT: Punctuation error
      - SPELL: Spelling mistake
      - VERB:INFL: Incorrectly inflected verb.
      - VERB:SVA: Subject-verb-agreement error.
      - VERB:TENSE: Changed verb tense.
      - WO: Word order
  """

  def __init__(self, aspell_lang: Optional[str] = None):
    self._aspell_lang = aspell_lang
    self._aspell = None

  def initialize(self) -> None:
    if self._aspell_lang is not None:
      self._aspell = aspell.Speller('lang', self._aspell_lang)

  def _get_one_sided_tag(self, tokens: list[spacy.tokens.Token]) -> str:
    """Gets tag for span insertions or deletions."""
    if _is_aux_span(tokens):
      return 'VERB:TENSE'

    pos_set = {token.pos_ for token in tokens}
    if len(pos_set) == 1:
      return pos_set.pop()

    if _is_verbal_span(tokens):
      return 'VERB'

    return 'OTHER'

  def _get_one2one_tag(self, source_token: spacy.tokens.Token,
                       target_token: spacy.tokens.Token) -> str:
    """Gets tag for 1:1 token replacements."""
    aspell_suggestions = []
    if self._aspell is not None:
      aspell_suggestions = self._aspell.suggest(source_token.text)

    if aspell_suggestions and aspell_suggestions[0].lower() != source_token.text.lower():
      if source_token.lemma == target_token.lemma:
        if source_token.pos_ == target_token.pos_ and (source_token.pos_
                                                       in {'NOUN', 'VERB'}):
          return source_token.pos_ + ':INFL'
        else:
          return 'MORPH'
      else:  # 1:1 OOV replacement, different lemmas
        for aspell_suggestion in aspell_suggestions:
          if aspell_suggestion.lower() == target_token.text.lower():
            return 'SPELL'

        lev = utils.levenshtein_distance(source_token.text, target_token.text,
                                         lambda x, y: x != y)
        relative_lev = 2.0 * lev / (
            len(source_token.text) + len(target_token.text))
        if relative_lev < 0.3:
          return 'SPELL'
    elif (source_token.lemma == target_token.lemma and
          source_token.pos != target_token.pos):  # Source is no spelling error
      return 'MORPH'

    return self._get_two_sided_tag([source_token], [target_token])

  def _get_two_sided_tag(self, source_tokens: list[spacy.tokens.Token],
                         target_tokens: list[spacy.tokens.Token]) -> str:
    """Gets tag for multi-token span substitutions."""
    if _is_aux_span(source_tokens) and _is_aux_span(target_tokens):
      return 'VERB:TENSE'

    if _is_noun_span(source_tokens) and _is_noun_span(target_tokens):
      for source_noun, target_noun in _match_up_pos_tokens(
          source_tokens, target_tokens, 'NOUN'):
        if not _morph_match(source_noun, target_noun, 'Number'):
          return 'NOUN:NUM'
      return 'NOUN'

    if _is_verbal_span(source_tokens) and _is_verbal_span(target_tokens):
      for source_verb, target_verb in _match_up_pos_tokens(
          source_tokens, target_tokens, 'VERB'):
        if not _morph_match(source_verb, target_verb, 'Tense'):
          return 'VERB:TENSE'
        if not _morph_match(source_verb, target_verb,
                            'Number') and not _morph_match(
                                source_verb, target_verb, 'Person'):
          return 'VERB:SVA'
      return 'VERB'

    # POS-based error types
    target_pos_set = {token.pos_ for token in target_tokens}
    if len(target_pos_set) == 1:
      pos = target_pos_set.pop()
      if pos == 'AUX':
        return 'VERB'
      return pos

    return 'OTHER'

  def classify_single_span(self, edit_span: utils.EditSpan) -> str:
    """Classifies a single edit span.

    See class docstring for the tag set.

    Args:
      edit_span: The to-be-classified edit span

    Returns:
      The error type tag.
    """
    if _is_orth_error(edit_span.source_span, edit_span.target_span):
      return 'ORTH'

    if _is_wo_error(edit_span.source_span, edit_span.target_span):
      return 'WO'

    if _is_punct_error(edit_span.source_span, edit_span.target_span):
      return 'PUNCT'

    def get_regular_tokens(tokens) -> list[spacy.tokens.Token]:
      return [
          token for token in tokens if not token.is_digit and not token.is_punct
      ]

    source_regular_tokens = get_regular_tokens(edit_span.source_span)
    target_regular_tokens = get_regular_tokens(edit_span.target_span)
    if _is_orth_error(source_regular_tokens, target_regular_tokens):
      return 'NUM'

    if not source_regular_tokens:
      return self._get_one_sided_tag(target_regular_tokens)

    if not target_regular_tokens:
      return self._get_one_sided_tag(source_regular_tokens)

    if len(source_regular_tokens) == 1 and len(target_regular_tokens) == 1:
      return self._get_one2one_tag(source_regular_tokens[0],
                                   target_regular_tokens[0])

    return self._get_two_sided_tag(source_regular_tokens, target_regular_tokens)

