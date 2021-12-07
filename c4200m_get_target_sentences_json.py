"""Looks up C4 sentences by their hashes and stores them in a TSV file.

This scripts cycles on all *train*.json.gz files contained in a folder,
to use it you can download the c4/en/ folder following the instructions here:
https://github.com/allenai/allennlp/discussions/5056

To extract all examples (183894319 for the v3.0.1 dataset) you have to run
the script on all edit files provided."""

import hashlib
import heapq
import os.path

from absl import app

import gzip
import json

LOGGING_STEPS = 100000


def main(argv):
    if len(argv) != 4:
        raise app.UsageError(
            "python3 c4200m_get_target_sentences.py <edits-tsv> <dataset_dir> "
            "<output-tsv>")
    edits_tsv_path = argv[1]
    dataset_dir = argv[2]
    output_tsv_path = argv[3]

    print("Loading C4_200M target sentence hashes from %r..." % edits_tsv_path)
    remaining_hashes = set()
    with open(edits_tsv_path) as edits_tsv_reader:
        for tsv_line in edits_tsv_reader:
            remaining_hashes.add(tsv_line.split("\t", 1)[0])
    print("Searching for %d target sentences in the C4 dataset..." %
          len(remaining_hashes))
    target_sentences = []
    num_tot_examples = 0
    for file_json in os.listdir(dataset_dir):
        if not (file_json.endswith('.json.gz') and 'train' in file_json):
            continue
        dataset_file = os.path.join(dataset_dir, file_json)
        with gzip.open(dataset_file, 'r') as f_in:
            for num_done_examples, example in enumerate(f_in):
                example = json.loads(example)
                for line in example["text"].split("\n"):
                    line_md5 = hashlib.md5(line.encode("utf-8")).hexdigest()
                    if line_md5 in remaining_hashes:
                        heapq.heappush(target_sentences, (line_md5, line))
                        remaining_hashes.remove(line_md5)
                if not remaining_hashes:
                    break
                if (num_tot_examples + num_done_examples) % LOGGING_STEPS == 0:
                    print("-- %d C4 examples done, %d sentences still to be found" %
                          ((num_tot_examples + num_done_examples), len(remaining_hashes)))
            num_tot_examples += num_done_examples
    print("Found %d target sentences (%d not found)." %
          (len(target_sentences), len(remaining_hashes)))
    print("Writing C4_200M sentence pairs to %r..." % output_tsv_path)
    with open(output_tsv_path, "w") as output_tsv_writer:
        while target_sentences:
            output_tsv_writer.write("%s\t%s\n" % heapq.heappop(target_sentences))


if __name__ == "__main__":
    app.run(main)
