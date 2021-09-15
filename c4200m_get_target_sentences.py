"""Looks up C4 sentences by their hashes and stores them in a TSV file."""

import hashlib
import heapq

from absl import app
import tensorflow_datasets as tfds

LOGGING_STEPS = 100000


def main(argv):
  if len(argv) != 3:
    raise app.UsageError(
        "python3 c4200m_get_target_sentences.py <edits-tsv> <output-tsv>")
  edits_tsv_path = argv[1]
  output_tsv_path = argv[2]

  print("Loading C4_200M target sentence hashes from %r..." % edits_tsv_path)
  remaining_hashes = set()
  with open(edits_tsv_path) as edits_tsv_reader:
    for tsv_line in edits_tsv_reader:
      remaining_hashes.add(tsv_line.split("\t", 1)[0])
  print("Searching for %d target sentences in the C4 dataset..." %
        len(remaining_hashes))
  target_sentences = []
  for num_done_examples, example in enumerate(
      tfds.load("c4/en:2.2.1", split="train")):
    for line in example["text"].numpy().decode("utf-8").split("\n"):
      line_md5 = hashlib.md5(line.encode("utf-8")).hexdigest()
      if line_md5 in remaining_hashes:
        heapq.heappush(target_sentences, (line_md5, line))
        remaining_hashes.remove(line_md5)
    if not remaining_hashes:
      break
    if num_done_examples % LOGGING_STEPS == 0:
      print("-- %d C4 examples done, %d sentences still to be found" %
            (num_done_examples, len(remaining_hashes)))
  print("Found %d target sentences (%d not found)." %
        (len(target_sentences), len(remaining_hashes)))
  print("Writing C4_200M sentence pairs to %r..." % output_tsv_path)
  with open(output_tsv_path, "w") as output_tsv_writer:
    while target_sentences:
      output_tsv_writer.write("%s\t%s\n" % heapq.heappop(target_sentences))


if __name__ == "__main__":
  app.run(main)
