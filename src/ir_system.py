import argparse
import os
import re
from collections import defaultdict
import math

from hazm import Normalizer, word_tokenize, Stemmer

def load_stopwords(path):
    with open(path, encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())


def load_documents(corpus_path):
    documents = {}
    for root, _, files in os.walk(corpus_path):
        year = os.path.basename(root)
        for file in files:
            if file.endswith(".ham"):
                doc_id = f"{year}_{file}"
                with open(os.path.join(root, file), encoding="utf-8") as f:
                    documents[doc_id] = f.read()
    return documents


def load_queries(queries_path):
    queries = {}
    for file in os.listdir(queries_path):
        if file.endswith(".q"):
            qid = int(file.replace(".q", ""))
            with open(os.path.join(queries_path, file), encoding="utf-8") as f:
                queries[qid] = f.read().strip()
    return queries


def load_relevance_judgements(path):
    judgements = defaultdict(set)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                qid, doc_id = parts
                judgements[int(qid)].add(doc_id)
    return dict(judgements)


class PersianTextPreprocessor:
    
    def __init__(self, stopwords):
        self.normalizer = Normalizer()
        self.stemmer = Stemmer()
        self.stopwords = stopwords

    def preprocess(self, text):
        text = self.normalizer.normalize(text)
        
        text = text.lower()
        
        text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        
        tokens = word_tokenize(text)
        
        result = []
        for token in tokens:
            if token and token not in self.stopwords:
                result.append(self.stemmer.stem(token))
        
        return result


class ManualTFIDF:

    
    def __init__(self):
        self.vocabulary = {}      
        self.idf = {}           
        self.doc_vectors = {}    
        self.doc_ids = []       
        
    def build_vocabulary(self, tokenized_docs):
        
        unique_words = set()
        
        for tokens in tokenized_docs:
            unique_words.update(tokens)
        
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(unique_words))}
    
    def calculate_tf(self, tokens):

        tf = {}
        for token in tokens:
            tf[token] = tf.get(token, 0) + 1
        return tf
    
    def calculate_idf(self, tokenized_docs):
        
        N = len(tokenized_docs)  
        
        df = {}  
        
        for tokens in tokenized_docs:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                df[token] = df.get(token, 0) + 1
        
        self.idf = {}
        for word in self.vocabulary:
            if word in df:
                #  log(N / df(t))
                self.idf[word] = math.log(N / df[word])
            else:
                self.idf[word] = 0.0
        
    
    def calculate_tfidf_vector(self, tokens):

        tf = self.calculate_tf(tokens)
        
        tfidf_vector = {}
        for word, freq in tf.items():
            if word in self.vocabulary:
                word_id = self.vocabulary[word]
                tfidf_value = freq * self.idf.get(word, 0)
                if tfidf_value > 0:
                    tfidf_vector[word_id] = tfidf_value
        
        return tfidf_vector
    
    def normalize_vector(self, vector):

        sum_of_squares = sum(val ** 2 for val in vector.values())
        
        if sum_of_squares == 0:
            return vector
        
        norm = math.sqrt(sum_of_squares)
        
        normalized = {key: val / norm for key, val in vector.items()}
        
        return normalized
    
    def fit_transform(self, doc_ids, tokenized_docs):
        
        self.doc_ids = doc_ids
        
        self.build_vocabulary(tokenized_docs)
        
        self.calculate_idf(tokenized_docs)
        
        self.doc_vectors = {}
        
        for doc_id, tokens in zip(doc_ids, tokenized_docs):
            tfidf_vec = self.calculate_tfidf_vector(tokens)
            self.doc_vectors[doc_id] = self.normalize_vector(tfidf_vec)
        
    
    def transform(self, tokens):

        tfidf_vec = self.calculate_tfidf_vector(tokens)
        return self.normalize_vector(tfidf_vec)



def cosine_similarity_manual(vec1, vec2):
    
    dot_product = 0.0
    
    for dim, val1 in vec1.items():
        if dim in vec2:
            dot_product += val1 * vec2[dim]
    
    return dot_product


class ManualIRSystem:

    
    def __init__(self, documents, preprocessor):

        self.preprocessor = preprocessor
        self.doc_ids = list(documents.keys())
        self.docs = list(documents.values())
        
        
        self.tokenized_docs = []
        for i, doc in enumerate(self.docs):

            tokens = self.preprocessor.preprocess(doc)
            self.tokenized_docs.append(tokens)
        
        

        self.tfidf = ManualTFIDF()
        self.tfidf.fit_transform(self.doc_ids, self.tokenized_docs)
        
    
    def retrieve(self, query_text, top_k=1000):

        # پیش‌پردازش کوئری
        query_tokens = self.preprocessor.preprocess(query_text)
        
        # تبدیل به بردار TF-IDF
        query_vector = self.tfidf.transform(query_tokens)
        
        # محاسبه شباهت با تمام اسناد
        scores = []
        for doc_id in self.doc_ids:
            doc_vector = self.tfidf.doc_vectors[doc_id]
            
            similarity = cosine_similarity_manual(query_vector, doc_vector)
            
            if similarity > 0:
                scores.append((doc_id, similarity))
        
        #  نزولی
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def search_demo(self, query_text, top_k=5):
        results = self.retrieve(query_text, top_k)
        
        if not results:
            print("هیچ سند مرتبطی پیدا نشد.")
            return []
        
        print(f"\n{len(results)} نتیجه برتر:")
        print("-" * 80)
        for i, (doc_id, score) in enumerate(results, 1):
            print(f"{i}. {doc_id:40s} (امتیاز: {score:.4f})")
        
        return results

class Evaluator:
    
    def __init__(self, relevance_judgements):
        self.relevance_judgements = relevance_judgements
    
    def normalize_doc_id(self, doc_id):
        return doc_id.split("_", 1)[1].replace(".ham", "")
    
    def precision_at_k(self, retrieved, relevant, k):

        if k == 0:
            return 0.0
        
        hit_count = 0
        for doc_id in retrieved[:k]:
            if self.normalize_doc_id(doc_id) in relevant:
                hit_count += 1
        
        return hit_count / k
    
    def recall_at_k(self, retrieved, relevant, k):

        if not relevant:
            return 0.0
        
        hit_count = 0
        for doc_id in retrieved[:k]:
            if self.normalize_doc_id(doc_id) in relevant:
                hit_count += 1
        
        return hit_count / len(relevant)
    
    def f1_score(self, precision, recall):

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
    
    def average_precision(self, retrieved, relevant):

        if not relevant:
            return 0.0
        
        precision_sum = 0.0
        hit_count = 0
        
        for i, doc_id in enumerate(retrieved, 1):
            if self.normalize_doc_id(doc_id) in relevant:
                hit_count += 1
                precision_at_i = hit_count / i
                precision_sum += precision_at_i
        
        if hit_count == 0:
            return 0.0
        
        return precision_sum / len(relevant)
    
    def evaluate(self, ir_system, queries, k=10):
        precision_scores = []
        recall_scores = []
        f1_scores = []
        ap_scores = []
        
        
        for qid, query_text in queries.items():
            relevant_docs = self.relevance_judgements.get(qid, set())
            
            if not relevant_docs:
                continue
            
            results = ir_system.retrieve(query_text, top_k=1000)
            retrieved_docs = [doc_id for doc_id, _ in results]
            
            p = self.precision_at_k(retrieved_docs, relevant_docs, k)
            r = self.recall_at_k(retrieved_docs, relevant_docs, k)
            f = self.f1_score(p, r)
            ap = self.average_precision(retrieved_docs, relevant_docs)
            
            precision_scores.append(p)
            recall_scores.append(r)
            f1_scores.append(f)
            ap_scores.append(ap)
        
        num_queries = len(ap_scores)
        
        return {
            "mean_precision": sum(precision_scores) / num_queries if num_queries > 0 else 0,
            "mean_recall": sum(recall_scores) / num_queries if num_queries > 0 else 0,
            "mean_f1": sum(f1_scores) / num_queries if num_queries > 0 else 0,
            "map": sum(ap_scores) / num_queries if num_queries > 0 else 0,
            "num_queries": num_queries
        }


def print_evaluation_results(results, k):
    print()
    print(f"تعداد کوئری‌های ارزیابی شده: {results['num_queries']}")
    print(f"مقدار K: {k}")
    print()
    print(f"MAP (Mean Average Precision):  {results['map']:.4f}")
    print(f"Mean Precision@{k}:             {results['mean_precision']:.4f}")
    print(f"Mean Recall@{k}:                {results['mean_recall']:.4f}")
    print(f"Mean F1-Score@{k}:              {results['mean_f1']:.4f}")
    print()


DEMO_QUERIES = [
    "بازسازي شهر زلزله زده بم",
    "برگزيدگان جشنواره فيلم فجر",
    "علل مرگ و مير نهنگ ها",
    "راه هاي جلو گيري از ابتلا به ديابت",
    "انتخاب سرمربي تيم ملي فوتبال ايران",
    "آمار مبتلايان به ايدز در ايران",
    "عرضه سهام شركت ايران خودرو",
    "نفرات برگزيده كنكور سراسري",
    "جبران خسارات ناشي از خشكسالي",
    "بهره برداري از تونل رسالت",
    "نرخ بيكاري در ايران",
    "روش هاي ترك سيگار",
    "يازدهمين نمايشگاه بين المللي كتاب",
    "فهرست فيلم هاي سينمايي پرفروش",
    "آثار باستاني تپه هگمتانه در همدان",
    "ميزان صادرات سالانه پسته",
    "اعدام قاچاقچيان در ايران",
    "راه اندازي مترو تجريش",
    "فيلتر سايت هاي اينترنتي در ايران",
    "حمله كنندگان به كوي دانشگاه",
    "سدهاي خاكي ايران",
    "طرح هاي بيابان زدايي",
    "اسطوره هاي ادبيات ايران",
    "ميزان رشد جمعيت در ايران",
    "آيين همسرداري در اسلام",
    "مديريت بهينه در صنعت برق",
    "صادرات محصولات پتروشيمي",
    "پيشگيري و درمان اعتياد",
    "اشتغال فارغ التحصيلان دانشگاه",
    "حمايت از ميراث فرهنگي و گردشگري كشور",
    "سرانه مصرف دارو در ايران",
    "كنگره هاي صنايع غذايي در ايران",
    "كتابخانه هاي ديجيتال",
    "آنفولانزاي مرغي",
    "نمايشگاه گل در تهران",
    "كاهش خدمت سربازي",
    "خرس قطبي",
    "پيرواني در ليگ برتر",
    "صادرات فرش در ايران",
    "قاچاق بنزين و گازوئيل از ايران",
    "مرگ و مير ناشي از تصادفات جاده اي",
    "تاثير خواب در سلامتي",
    "حكيم ابولقاسم فردوسي",
    "آلودگي محيط زيست",
    "ناامني هاي پس از جنگ در عراق",
    "تسهيلات ازدواج جوانان",
    "سرانه مصرف آب آشاميدني",
    "سطح تحصيلات عالي",
    "گسل هاي فعال شمال تهران",
    "كمبود آب در تهران",
]


def main(data_dir, k=10):
    corpus_path = os.path.join(data_dir, "HamshahriCorpus")
    queries_path = os.path.join(data_dir, "Queries")
    judgements_path = os.path.join(data_dir, "RelativeAssesemnt", "judgements.txt")
    stopwords_path = os.path.join(data_dir, "persian_stopwords.txt")

    stopwords = load_stopwords(stopwords_path)
    documents = load_documents(corpus_path)
    queries = load_queries(queries_path)
    judgements = load_relevance_judgements(judgements_path)

    preprocessor = PersianTextPreprocessor(stopwords)
    ir_system = ManualIRSystem(documents, preprocessor)

    evaluator = Evaluator(judgements)
    results = evaluator.evaluate(ir_system, queries, k=k)

    print_evaluation_results(results, k=k)

    return ir_system


def run_demo(system, top_k=5):
    print("\n" + "=" * 80)
    print("نمونه‌های جستجو")
    print("=" * 80)

    for i, query in enumerate(DEMO_QUERIES, 1):
        print(f"\nکوئری {i}: {query}")
        system.search_demo(query, top_k=top_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Persian TF-IDF information retrieval on the Hamshahri corpus"
    )
    parser.add_argument("--data-dir", default="data/HamshahriData",
                        help="folder containing HamshahriCorpus/, Queries/, RelativeAssesemnt/ and persian_stopwords.txt")
    parser.add_argument("--k", type=int, default=10, help="cutoff for Precision/Recall/F1@k")
    parser.add_argument("--no-demo", action="store_true", help="skip the 50 sample searches")
    args = parser.parse_args()

    system = main(args.data_dir, k=args.k)
    if system and not args.no_demo:
        run_demo(system)
