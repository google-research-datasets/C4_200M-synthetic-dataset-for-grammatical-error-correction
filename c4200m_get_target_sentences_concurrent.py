"""Looks up C4 sentences by their hashes and stores them in a TSV file."""

import hashlib
import heapq

from absl import app
import tensorflow_datasets as tfds
import multiprocessing as mp
import re

LOGGING_STEPS = 100000
BATCH_SIZE = 64

def rreplace(string, old, new):
  """Replaces the last occurrence of the old string with the new string
  
  Args:
    string (str): text to be checked
    old (str): text to be replaced
    new (str): text that will replace old text
  
  Returns:
    str: updated text with last occurrence of old replaced with new
  
  """
  return new.join(string.rsplit(old, 1))

def get_file_paths(edits_tsv_path, concurrent_runs):
  start_index_string = re.search('(\d{5})-of-00010$', 'edits.tsv-00000-of-00010').group(1)
  start_index = int(start_index_string)
  end_index = min(start_index + concurrent_runs, 10)
  return [rreplace(edits_tsv_path, start_index_string, start_index_string[:-1] + str(i)) 
            for i in range(start_index, end_index)]

def load_file(tup):
  index, edits_tsv_path = tup
  print("Loading C4_200M target sentence hashes from %r..." % edits_tsv_path)
  with open(edits_tsv_path, "r", encoding="utf-8") as edits_tsv_reader:
    return { tsv_line.split("\t", 1)[0]: index for tsv_line in edits_tsv_reader}
        
def save_file(tup):
  target_sentences, output_tsv_path = tup
  print("Writing C4_200M sentence pairs to %r..." % output_tsv_path)
  with open(output_tsv_path, "w", encoding="utf-8") as output_tsv_writer:
    while target_sentences:
      output_tsv_writer.write("%s\t%s\n" % heapq.heappop(target_sentences))

def main(argv):
  if len(argv) != 4:
    raise app.UsageError(
        "python3 c4200m_get_target_sentences.py <edits-tsv> <output-tsv> <concurrent-runs>")
  edits_tsv_path = argv[1]
  output_tsv_path = argv[2]
  concurrent_runs = min(int(argv[3]), 10)
  
  edits_tsv_paths = get_file_paths(edits_tsv_path, concurrent_runs)
  output_tsv_paths = get_file_paths(output_tsv_path, concurrent_runs)

  remaining_hashes = dict()
  hashes = mp.Pool(concurrent_runs).map(load_file, enumerate(edits_tsv_paths))
  for h in hashes:
    remaining_hashes.update(h)
  
  total_hashes = len(remaining_hashes)
  print("Searching for %d target sentences in the C4 dataset..." %
        total_hashes)

  target_sentences = [[] for i in range(concurrent_runs)]
  for num_done_examples, example in enumerate(
      tfds.load("c4/en:2.2.1", split="train", batch_size=BATCH_SIZE)):
    for l in example["text"].numpy():
      for line in l.decode("utf-8").split("\n"):
        line_md5 = hashlib.md5(line.encode("utf-8")).hexdigest()
        if line_md5 in remaining_hashes:
          heapq.heappush(target_sentences[remaining_hashes[line_md5]], (line_md5, line))
          remaining_hashes.pop(line_md5, None)
    if len(remaining_hashes) == 0:
      break
    if num_done_examples % int(LOGGING_STEPS / BATCH_SIZE) == 0:
      print("-- %d C4 examples done, %d sentences still to be found" %
            (num_done_examples * BATCH_SIZE, len(remaining_hashes)))
  print("Found %d target sentences (%d not found)." %
        (total_hashes - len(remaining_hashes), len(remaining_hashes)))
  
  file_tuples = [(target_sentences[i], output_tsv_path) 
                    for i, output_tsv_path in enumerate(output_tsv_paths)]
  mp.Pool(concurrent_runs).map(save_file, file_tuples)


if __name__ == "__main__":
  app.run(main)
