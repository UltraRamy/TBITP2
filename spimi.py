import os
import contextlib
import heapq
import math

from tqdm import tqdm
from index import InvertedIndexReader, InvertedIndexWriter
from util import TrieIdMap, sorted_merge_posts_and_tfs
from compression import VBEPostings
from bsbi import BSBIIndex


class SPIMIIndex(BSBIIndex):

    def __init__(self, data_dir, output_dir, postings_encoding, index_name="main_index"):
        super().__init__(data_dir, output_dir, postings_encoding, index_name)
        self.term_id_map = TrieIdMap()
        self.doc_id_map = TrieIdMap()

    def _flush(self, in_memory_index, block_num):
        index_id = f'spimi_block_{block_num}'
        self.intermediate_indices.append(index_id)
        with InvertedIndexWriter(index_id, self.postings_encoding, directory=self.output_dir) as index:
            for term_str in sorted(in_memory_index.keys()):
                postings = sorted(in_memory_index[term_str].keys())
                tf_list = [in_memory_index[term_str][doc_id] for doc_id in postings]
                index.append(term_str, postings, tf_list)

    def _spimi_merge(self, indices, merged_index):
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)
        for t, postings_, tf_list_ in merged_iter:
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)),
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(self.term_id_map[curr], postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(self.term_id_map[curr], postings, tf_list)

    def index(self, block_size=5000):
        in_memory_index = {}
        token_count = 0
        block_num = 0

        for block_dir in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            block_path = os.path.join('./', self.data_dir, block_dir)
            for filename in sorted(next(os.walk(block_path))[2]):
                doc_path = block_path + '/' + filename
                doc_id = self.doc_id_map[doc_path]

                with open(doc_path, 'r', encoding='utf8', errors='surrogateescape') as f:
                    for token in f.read().split():
                        if token not in in_memory_index:
                            in_memory_index[token] = {}
                        if doc_id not in in_memory_index[token]:
                            in_memory_index[token][doc_id] = 0
                        in_memory_index[token][doc_id] += 1
                        token_count += 1

                        if token_count >= block_size:
                            self._flush(in_memory_index, block_num)
                            in_memory_index = {}
                            token_count = 0
                            block_num += 1

        if in_memory_index:
            self._flush(in_memory_index, block_num)

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(idx_id, self.postings_encoding, directory=self.output_dir))
                           for idx_id in self.intermediate_indices]
                self._spimi_merge(indices, merged_index)

        self.save()


if __name__ == '__main__':
    spimi = SPIMIIndex(data_dir='collection',
                       postings_encoding=VBEPostings,
                       output_dir='index')
    spimi.index()

    queries = ["alkylated with radioactive iodoacetate",
               "psychodrama for disturbed children",
               "lipid metabolism in toxemia and normal pregnancy"]

    for query in queries:
        print("Query  :", query)
        print("Results:")
        for score, doc in spimi.retrieve_bm25(query, k=5):
            print(f"  {doc:40} {score:.3f}")
        print()
