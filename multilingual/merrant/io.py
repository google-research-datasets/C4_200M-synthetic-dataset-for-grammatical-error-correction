"""Utility module for input and output to mERRANT."""

import abc
from merrant import utils


class AnnotationFormatter(metaclass=abc.ABCMeta):
  """Base class for output formatters.

  The main mERRANT interface `api.Annotator` generates `utils.Annotation`s.
  Output formatters convert these annotations to readable standard formats.
  """

  @abc.abstractmethod
  def format(self, annotation: utils.Annotation) -> bytes:
    """Converts an Annotation.

    The type of the returned value depends on the subclass. It can be a plain
    utf-8 encoded string or a serialized proto object.

    Args:
      annotation: The (tagged) annotation to convert to bytes.

    Returns:
      The output representation as bytes.
    """


def make_formatter(formatter_name: str) -> AnnotationFormatter:
  """Factory function for `AnnotationFormatter`s."""
  if formatter_name == "M2_CHAR":
    return M2CharFormatter()
  if formatter_name == "M2_TOK":
    return M2TokFormatter()
  if formatter_name == "TSV_TAGGED_CORRUPTION":
    return TSVTaggedCorruptionFormatter()
  raise ValueError("Unknown formatter name %r" % (formatter_name,))


_M2_LINE = "A %d %d|||%s|||%s|||REQUIRED|||-NONE-|||%d"
_NOOP_M2_LINE = "A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||%d"


class M2CharFormatter(AnnotationFormatter):
  """Generates character-level M2 lines (string)."""

  def format(self, annotation: utils.Annotation) -> bytes:
    """Formats an Annotation as character-level M2 lines."""
    m2_lines = ["S " + annotation.source_doc.text]
    for annotator_id, target_sentence in enumerate(annotation.target_sentences):
      if not target_sentence.edit_spans:
        m2_lines.append(_NOOP_M2_LINE % annotator_id)

      for edit_span in target_sentence.edit_spans:
        try:
          target_text = edit_span.target_span.text
        except IndexError:
          target_text = ""

        start_char = edit_span.source_span.start_char
        end_char = edit_span.source_span.end_char
        if edit_span.source_span.start == edit_span.source_span.end:  # Insert
          if start_char > end_char:  # Can happen with insertions
            target_text = " " * (start_char - end_char) + target_text
          start_char = end_char

        m2_lines.append(_M2_LINE % (start_char, end_char, edit_span.get_label(),
                                    target_text, annotator_id))
    m2_lines.append("")
    return "\n".join(m2_lines).encode("utf-8")


class M2TokFormatter(AnnotationFormatter):
  """Generates spaCy-tokenized M2 lines (string)."""

  def format(self, annotation: utils.Annotation) -> bytes:
    """Formats an Annotation as spaCy-tokenized M2 lines."""
    m2_lines = [
        "S " + " ".join([token.text for token in annotation.source_doc])
    ]
    for annotator_id, target_sentence in enumerate(annotation.target_sentences):
      if not target_sentence.edit_spans:
        m2_lines.append(_NOOP_M2_LINE % annotator_id)

      for edit_span in target_sentence.edit_spans:
        try:
          target_text = edit_span.target_span.text
        except IndexError:
          target_text = ""
        m2_lines.append(_M2_LINE %
                        (edit_span.source_span.start, edit_span.source_span.end,
                         edit_span.get_label(), target_text, annotator_id))
    m2_lines.append("")
    return "\n".join(m2_lines).encode("utf-8")


class TSVTaggedCorruptionFormatter(AnnotationFormatter):
  r"""Generates tagged corruption data (string).

  This `AnnotationFormatter` generates training data for tagged corruption
  models in TSV format. The generated string contains multiple lines, one for
  each error type tag in the edit spans, formatted as
    <tag>\t<target-sentence>\t<source-sentence>.
  """

  def format(self, annotation: utils.Annotation) -> bytes:
    """Formats an Annotation as tagged corruption TSV lines."""
    tsv_lines = []
    source_text = annotation.source_doc.text
    for target_sentence in annotation.target_sentences:
      target_text = target_sentence.doc.text
      for edit_span in target_sentence.edit_spans:
        tsv_lines.append("\t".join(
            [edit_span.get_label(), target_text, source_text]))
    return "\n".join(tsv_lines).encode("utf-8")

