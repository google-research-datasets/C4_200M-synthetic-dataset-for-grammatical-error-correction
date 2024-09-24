"""Common utility functions and data classes for mERRANT."""

import dataclasses
import os

from typing import Callable, Sequence, TypeVar

import spacy

# Default tag name for (untagged) EditSpans.
UNK_TAG_NAME = "UNK"

def load_spacy_from_google3(model_name: str) -> spacy.Language:
  """Loads standard spaCy model in google3."""
  return spacy.load(model_name)


@dataclasses.dataclass
class EditSpan:
  """Represents a pair of source and target span."""

  source_span: spacy.tokens.span.Span
  target_span: spacy.tokens.span.Span
  tag: str = UNK_TAG_NAME

  def is_deletion(self) -> bool:
    return len(self.target_span) == 0  # pylint: disable=g-explicit-length-test

  def is_insertion(self) -> bool:
    return len(self.source_span) == 0  # pylint: disable=g-explicit-length-test

  def get_label(self) -> str:
    if self.is_deletion():  # Unnecessary
      return "U:" + self.tag

    if self.is_insertion():  # Missing
      return "M:" + self.tag

    # Replacement
    return "R:" + self.tag

  def __str__(self):
    try:
      source_text = self.source_span.text
    except IndexError:
      source_text = ""
    try:
      target_text = self.target_span.text
    except IndexError:
      target_text = ""

    return "EditSpan(source_span=%d:%d:%r, target_span=%d:%d:%r label=%r)" % (
        self.source_span.start, self.source_span.end, source_text,
        self.target_span.start, self.target_span.end, target_text,
        self.get_label())

  def __repr__(self):
    return self.__str__()


@dataclasses.dataclass
class TargetSentence:
  """Represents a single corrected sentence in an `Annotation`."""

  doc: spacy.tokens.doc.Doc
  edit_spans: list[EditSpan] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Annotation:
  """Container for an annotated example.

  An annotation pairs up a source (original) sentence with a number of target
  (corrected) sentences.
  """

  source_doc: spacy.tokens.doc.Doc
  target_sentences: list[TargetSentence] = dataclasses.field(
      default_factory=list)


T = TypeVar("T")


def levenshtein_matrix(
    source_tokens: Sequence[T], target_tokens: Sequence[T],
    compare_func: Callable[[T, T], float]) -> list[list[float]]:
  """Computes the Levenshtein matrix between two sequences.

  Adapted from
  https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python

  Args:
    source_tokens: Source sequence
    target_tokens: Target sequence
    compare_func: Used to compute substitution costs between tokens

  Returns:
    Full Levenshtein matrix between source_tokens and target_tokens.
  """
  previous_row = [float(c) for c in range(len(target_tokens) + 1)]
  mat = [previous_row]
  for i, c1 in enumerate(source_tokens):
    current_row = [i + 1]
    for j, c2 in enumerate(target_tokens):
      insertions = previous_row[j + 1] + 1.0
      deletions = current_row[j] + 1.0
      substitutions = previous_row[j] + compare_func(c1, c2)
      current_row.append(min(insertions, deletions, substitutions))
    previous_row = current_row
    mat.append(current_row)
  return mat


def levenshtein_distance(source_tokens: Sequence[T], target_tokens: Sequence[T],
                         compare_func: Callable[[T, T], float]) -> float:
  """Computes the Levenshtein distance between two sequences."""
  return levenshtein_matrix(source_tokens, target_tokens, compare_func)[-1][-1]


def _token_sub_cost(source_token: spacy.tokens.Token,
                    target_token: spacy.tokens.Token) -> float:
  """Computes the substitution cost between two spaCy tokens.

  This cost function is loosely inspired by the cost function used in ERRANT:
  https://github.com/chrisjbryant/errant/blob/8e577bc4169a31c443655e8c5a17ca95413b0b2d/errant/alignment.py#L86

  The cost is discounted if the tokens share the same lower case characters,
  POS tag, or lemma.

  Args:
    source_token: Source (original) spaCy token
    target_token: Target (corrected) spaCy token

  Returns:
    The cost between 0 and 1 of substituting `source_token` with `target_token`.
  """
  if source_token.text == target_token.text:
    return 0.0

  if source_token.text.lower() == target_token.text.lower():
    return 0.1

  lemma_cost = 0.3 * (source_token.lemma != target_token.lemma)
  pos_cost = 0.5 * (source_token.pos != target_token.pos)
  return 0.2 + lemma_cost + pos_cost


def get_edit_spans(source_tokens: spacy.tokens.doc.Doc,
                   target_tokens: spacy.tokens.doc.Doc) -> list[EditSpan]:
  """Gets the edit spans by aligning the source and target tokens.

  Args:
    source_tokens: Source sequence
    target_tokens: Target sequence

  Returns:
    List of untagged `EditSpan`s, each representing a correction span.
  """
  lev_mat = levenshtein_matrix(source_tokens, target_tokens, _token_sub_cost)
  row = len(source_tokens)
  col = len(target_tokens)
  align = []
  while row * col != 0:
    if lev_mat[row - 1][col] < lev_mat[row][col]:
      row -= 1
    elif lev_mat[row][col - 1] < lev_mat[row][col]:
      col -= 1
    else:
      row -= 1
      col -= 1
      if lev_mat[row][col] == lev_mat[row + 1][col + 1]:
        align.append((row, col))
  align.reverse()
  align.append((len(source_tokens), len(target_tokens)))
  prev_source_pos = -1
  prev_target_pos = -1
  spans = []
  for source_pos, target_pos in align:
    if prev_source_pos + 1 != source_pos or prev_target_pos + 1 != target_pos:
      spans.append(
          EditSpan(
              source_span=source_tokens[prev_source_pos + 1:source_pos],
              target_span=target_tokens[prev_target_pos + 1:target_pos]))
    prev_source_pos = source_pos
    prev_target_pos = target_pos
  return spans

