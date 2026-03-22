import re
import math
from bsbi import BSBIIndex
from compression import VBEPostings

######## >>>>> sebuah IR metric: RBP p = 0.8

def rbp(ranking, p = 0.8):
  """ menghitung search effectiveness metric score dengan 
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan
        
      Returns
      -------
      Float
        score RBP
  """
  score = 0.
  for i in range(1, len(ranking)):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score


def dcg(ranking, k=None):
    """
    Menghitung Discounted Cumulative Gain (DCG) sampai rank ke-k.

    DCG@k = sum_{i=1}^{k} rel_i / log2(i + 1)

    Parameters
    ----------
    ranking: List[int]
        Vektor biner relevansi [1, 0, 1, ...] dari rank 1, 2, 3, dst.
    k: int or None
        Kedalaman evaluasi. Jika None, gunakan seluruh panjang ranking.

    Returns
    -------
    float
        Skor DCG
    """
    if k is None:
        k = len(ranking)
    score = 0.0
    for i in range(1, min(k, len(ranking)) + 1):
        score += ranking[i - 1] / math.log2(i + 1)
    return score


def ndcg(ranking, k=None):
    """
    Menghitung Normalized DCG (NDCG) sampai rank ke-k.

    NDCG@k = DCG@k / IDCG@k

    dimana IDCG@k adalah DCG dari ranking ideal (semua dokumen relevan
    diurutkan di posisi teratas).

    Parameters
    ----------
    ranking: List[int]
        Vektor biner relevansi
    k: int or None
        Kedalaman evaluasi. Jika None, gunakan seluruh panjang ranking.

    Returns
    -------
    float
        Skor NDCG (antara 0.0 dan 1.0)
    """
    if k is None:
        k = len(ranking)
    ideal = sorted(ranking, reverse=True)
    idcg = dcg(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg(ranking, k) / idcg


def ap(ranking, R=None):
    """
    Menghitung Average Precision (AP).

    AP = (1/R) * sum_{i: doc_i relevan} Precision@i

    dimana R adalah total dokumen relevan di koleksi (bukan hanya di ranking).
    Jika R tidak diberikan, gunakan jumlah relevan di dalam ranking.

    Parameters
    ----------
    ranking: List[int]
        Vektor biner relevansi
    R: int or None
        Total dokumen relevan di koleksi untuk query ini.
        Jika None, dihitung dari ranking saja.

    Returns
    -------
    float
        Skor AP
    """
    if R is None:
        R = sum(ranking)
    if R == 0:
        return 0.0
    score = 0.0
    hits = 0
    for i, rel in enumerate(ranking, 1):
        if rel:
            hits += 1
            score += hits / i
    return score / R


######## >>>>> memuat qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ memuat query relevance judgment (qrels) 
      dalam format dictionary of dictionary
      qrels[query id][document id]

      dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
      relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
      Doc 10 tidak relevan dengan Q3.

  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels

######## >>>>> EVALUASI !

def eval(qrels, query_file = "queries.txt", k = 1000):
  """
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-1000 documents.

  """
  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

  with open(query_file) as file:
    rbp_scores_tfidf   = []
    dcg_scores_tfidf   = []
    ndcg_scores_tfidf  = []
    ap_scores_tfidf    = []

    rbp_scores_bm25    = []
    dcg_scores_bm25    = []
    ndcg_scores_bm25   = []
    ap_scores_bm25     = []

    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])

      R = sum(qrels[qid].values())

      # HATI-HATI, doc id saat indexing bisa jadi berbeda dengan doc id
      # yang tertera di qrels
      ranking_tfidf = []
      for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = k):
          did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
          ranking_tfidf.append(qrels[qid][did])

      ranking_bm25 = []
      for (score, doc) in BSBI_instance.retrieve_bm25(query, k = k):
          did = int(re.search(r'\/.*\/.*\/(.*)\.txt', doc).group(1))
          ranking_bm25.append(qrels[qid][did])

      rbp_scores_tfidf.append(rbp(ranking_tfidf))
      dcg_scores_tfidf.append(dcg(ranking_tfidf))
      ndcg_scores_tfidf.append(ndcg(ranking_tfidf))
      ap_scores_tfidf.append(ap(ranking_tfidf, R))

      rbp_scores_bm25.append(rbp(ranking_bm25))
      dcg_scores_bm25.append(dcg(ranking_bm25))
      ndcg_scores_bm25.append(ndcg(ranking_bm25))
      ap_scores_bm25.append(ap(ranking_bm25, R))

  def mean(lst): return sum(lst) / len(lst)

  print("=" * 45)
  print(f"{'Metric':<10} {'TF-IDF':>15} {'BM25':>15}")
  print("=" * 45)
  print(f"{'RBP':<10} {mean(rbp_scores_tfidf):>15.4f} {mean(rbp_scores_bm25):>15.4f}")
  print(f"{'DCG':<10} {mean(dcg_scores_tfidf):>15.4f} {mean(dcg_scores_bm25):>15.4f}")
  print(f"{'NDCG':<10} {mean(ndcg_scores_tfidf):>15.4f} {mean(ndcg_scores_bm25):>15.4f}")
  print(f"{'MAP':<10} {mean(ap_scores_tfidf):>15.4f} {mean(ap_scores_bm25):>15.4f}")
  print("=" * 45)

if __name__ == '__main__':
  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels salah"
  assert qrels["Q1"][300] == 0, "qrels salah"

  eval(qrels)