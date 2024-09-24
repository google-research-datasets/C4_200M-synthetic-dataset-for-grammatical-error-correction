r"""Command-line tool for running mERRANT.

This script reads TSV data from stdin and writes formatted annotations to
stdout.

Example:
  echo -e "I goed to the storr.\tI went to the store." | \
    python3 -m merrant.annotate

Output (M2_CHAR format):
  S I goed to the storr.
  A 2 6|||R:VERB:INFL|||went|||REQUIRED|||-NONE-|||0
  A 14 19|||R:SPELL|||store|||REQUIRED|||-NONE-|||0
"""

import sys
from typing import Sequence

from absl import app
from absl import flags

from merrant import api
from merrant import io

_SPACY_MODEL = flags.DEFINE_string("spacy_model", "en_core_web_sm",
                                   "Tagging model.")

_OUTPUT_FORMAT = flags.DEFINE_enum(
    "output_format", "M2_CHAR", ["M2_CHAR", "M2_TOK", "TSV_TAGGED_CORRUPTION"],
    "Tagging model.")


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  annotator = api.Annotator(_SPACY_MODEL.value,
                            aspell_lang=_SPACY_MODEL.value[:2])
  annotator.initialize()
  formatter = io.make_formatter(_OUTPUT_FORMAT.value)
  for line in sys.stdin:
    parts = line.strip("\n").split("\t")
    annotation = annotator.annotate(parts[0], parts[1:])
    print(formatter.format(annotation).decode("utf-8"))


if __name__ == "__main__":
  app.run(main)

