import os
import multiprocessing as mp
from functools import partial
import json
import hashlib
import inspect
import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import argparse
from rank_bm25 import BM25Okapi
from openai import OpenAI
from transformers import AutoConfig, AutoModel, AutoTokenizer
from sklearn.preprocessing import normalize
from src.retrieval.eval_utils import evaluate_retrieval, evaluate_retrieval_turn2session
from src.retrieval.index_expansion_utils import fetch_expansion_from_cache, resolve_expansion


client = OpenAI(
    api_key="empty",
    base_url="http://localhost:8001/v1",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--outfile_prefix', type=str, default=None, required=False)
    parser.add_argument('--cache_dir')
    
    # basic parameters
    parser.add_argument('--retriever', type=str, required=True,
                        choices=['oracle', 'flat-bm25', 'flat-contriever', 'flat-stella', 'flat-gte'])
    parser.add_argument('--granularity', type=str, required=True, choices=['session', 'turn'])

    # index expansion
    parser.add_argument('--index_expansion_method', type=str, default='none', 
                        choices=['none', 'session-summ', 'session-keyphrase', 'session-userfact', 'turn-keyphrase', 'turn-userfact'])
    parser.add_argument('--index_expansion_llm', type=str, default=None)
    parser.add_argument('--index_expansion_result_cache', type=str, default=None)
    parser.add_argument('--index_expansion_result_join_mode', type=str, default='none', 
                        choices=['separate', 'split-separate', 'merge', 'split-merge', 'replace', 'split-replace', 'none'])
    return parser.parse_args()


def check_args(args):
    print(args)
    if args.index_expansion_method != 'none':
        print('Note: index expansion method {} specified'.format(args.index_expansion_method))
        assert args.index_expansion_result_join_mode is not None and args.index_expansion_result_join_mode != 'none' 
        if args.index_expansion_method in ['session-summ', 'session-keyphrase', 'session_userfact']:
            assert args.granularity == 'session'
        if args.index_expansion_result_cache is not None and args.index_expansion_result_cache != None:
            assert args.index_expansion_method in args.index_expansion_result_cache
            print('Using cached index expansion results at', args.index_expansion_result_cache)


def get_outfile_prefix(args):
    if args.outfile_prefix is not None and args.outfile_prefix.lower() != 'none':
        outfile_prefix = args.outfile_prefix
    else:
        outfile_prefix = args.in_file.split('/')[-1]
    return outfile_prefix


def load_legacy_qwen_config(model_name):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if not hasattr(config, 'rope_theta'):
        rope_parameters = getattr(config, 'rope_parameters', None)
        if not isinstance(rope_parameters, dict) or 'rope_theta' not in rope_parameters:
            raise ValueError('Model configuration has no rotary embedding base')
        config.rope_theta = rope_parameters['rope_theta']
    return config


LEGACY_QWEN_ROTARY_SHA256 = '85e083350498af430ef887489b5429fbaf3c8fe10fa7d0aa10541e1aafaf81fe'
LEGACY_QWEN_METHOD_SHA256 = {
    '_set_cos_sin_cache': '04b15e4dfae51d556a5a35a1ac8c37e9a26d2d1a511348079cd84e2a0f8b7d37',
    'forward': 'a68fa1629dadbbc01b79166c019f29cd8439a318c16bb71548aea5cb2e5bd209',
}


def restore_legacy_rotary_buffers(model, cache_length):
    """Restore the reviewed custom Qwen rotary buffers after meta loading."""
    config = model.config
    rope_parameters = getattr(config, 'rope_parameters', None)
    if (not isinstance(rope_parameters, dict) or rope_parameters.get('rope_type', 'default') != 'default' or
            rope_parameters.get('rope_theta') != getattr(config, 'rope_theta', None)):
        raise ValueError('unsupported rotary configuration')
    if not isinstance(cache_length, int) or cache_length <= 0:
        raise ValueError('invalid rotary cache length')
    parameters = list(model.parameters())
    if not parameters or any(p.device.type != 'cpu' or p.dtype != torch.float32 for p in parameters):
        raise ValueError('unsupported rotary model dtype')
    rotary_modules = []
    for module in model.modules():
        nonpersistent = module._non_persistent_buffers_set
        if not nonpersistent:
            continue
        if type(module).__name__ != 'Qwen2RotaryEmbedding' or nonpersistent != {'inv_freq', 'cos_cached', 'sin_cached'}:
            raise ValueError('unsupported nonpersistent model buffers')
        try:
            source_hash = hashlib.sha256(inspect.getsource(type(module)).encode('utf-8')).hexdigest()
            methods = {name: getattr(type(module), name) for name in LEGACY_QWEN_METHOD_SHA256}
            if any(hasattr(method, '__wrapped__') for method in methods.values()):
                raise ValueError('unsupported rotary implementation')
            method_hashes = {
                name: hashlib.sha256(inspect.getsource(method).encode('utf-8')).hexdigest()
                for name, method in methods.items()
            }
        except (AttributeError, OSError, TypeError) as exc:
            raise ValueError('unsupported rotary implementation') from exc
        if (source_hash != LEGACY_QWEN_ROTARY_SHA256 or method_hashes != LEGACY_QWEN_METHOD_SHA256 or
                any(name in module.__dict__ for name in LEGACY_QWEN_METHOD_SHA256)):
            raise ValueError('unsupported rotary implementation')
        dim = module.dim
        base = module.base
        if (not isinstance(dim, int) or dim <= 0 or dim % 2 or
                not isinstance(base, (int, float)) or not math.isfinite(base) or base <= 0 or
                base != config.rope_theta or module.inv_freq.device.type != 'cpu' or
                module.inv_freq.shape != (dim // 2,) or module.inv_freq.dtype != torch.float32 or
                module.cos_cached.device.type != 'cpu' or module.sin_cached.device.type != 'cpu' or
                module.cos_cached.dtype != torch.float32 or module.sin_cached.dtype != torch.float32 or
                module.cos_cached.ndim != 2 or module.cos_cached.shape[1] != dim or
                module.sin_cached.shape != module.cos_cached.shape or not callable(module._set_cos_sin_cache) or
                not isinstance(module.max_position_embeddings, int) or module.max_position_embeddings <= 0):
            raise ValueError('unsupported rotary buffer layout')
        rotary_modules.append(module)
    if not rotary_modules or len(rotary_modules) != getattr(config, 'num_hidden_layers', None):
        raise ValueError('unsupported rotary layer count')
    for module in rotary_modules:
        device = module.inv_freq.device
        inv_freq = 1.0 / (module.base ** (torch.arange(0, module.dim, 2, dtype=torch.int64, device=device).float() / module.dim))
        module.register_buffer('inv_freq', inv_freq, persistent=False)
        module._set_cos_sin_cache(seq_len=min(cache_length, module.max_position_embeddings), device=device, dtype=module.cos_cached.dtype)


class DenseRetrievalMaster:
    def __init__(self, args, gpu_id):
        self.args = args
        self.device = torch.device('cuda', gpu_id)
        # print('Initializing DenseRetrievalMaster with device', self.device)
        self.prepare_retriever()

    def prepare_retriever(self):
        self.retriever_model = None
        
        if self.args.retriever == 'flat-contriever':
            model = AutoModel.from_pretrained('facebook/contriever').to(self.device)
            tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
            self.retriever_model = (tokenizer, model)

        elif self.args.retriever == 'flat-stella':
            model_dir = self.args.cache_dir + "/dunzhang_stella_en_1.5B_v5"
            vector_dim = 1024
            vector_linear_directory = f"2_Dense_{vector_dim}"
            config = load_legacy_qwen_config(model_dir)
            model = AutoModel.from_pretrained(model_dir, config=config, trust_remote_code=True)
            restore_legacy_rotary_buffers(model, 513)
            model = model.to(self.device)
            model.eval()
            tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
            vector_linear = torch.nn.Linear(in_features=model.config.hidden_size, out_features=vector_dim).to(self.device)
            vector_linear_dict = {
                k.replace("linear.", ""): v for k, v in
                torch.load(os.path.join(model_dir, f"{vector_linear_directory}/pytorch_model.bin"),
                           map_location='cpu', weights_only=True).items()
            }
            vector_linear.load_state_dict(vector_linear_dict)
            vector_linear.to(self.device)
            self.retriever_model = (tokenizer, model, vector_linear)
            
        elif self.args.retriever == 'flat-gte':
            tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)
            config = load_legacy_qwen_config('Alibaba-NLP/gte-Qwen2-7B-instruct')
            model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', config=config, trust_remote_code=True)
            restore_legacy_rotary_buffers(model, 8193)
            model = model.to(self.device)
            model.eval()
            self.retriever_model = (tokenizer, model)

    def run_flat_retrieval(self, query, retriever, corpus):
        if retriever == 'flat-bm25':
            tokenized_corpus = [doc.split(" ") for doc in corpus]
            # tokenized_torpus = word_tokenize(corpus)
            bm25 = BM25Okapi(tokenized_corpus)
            scores = bm25.get_scores(query.split(" "))
            return np.argsort(scores)[::-1]

        elif retriever in ['flat-contriever', 'flat-stella', 'flat-gte']:
            model2bsz = {'flat-contriever': 128, 'flat-stella': 64, 'flat-gte': 1}
            bsz = model2bsz[retriever]
            
            if retriever == 'flat-contriever':
                tokenizer, model = self.retriever_model
                def mean_pooling(token_embeddings, mask):
                    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
                    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
                    return sentence_embeddings

                with torch.no_grad():
                    inputs = tokenizer([query], padding=True, truncation=True, return_tensors='pt')
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}
                    outputs = model(**inputs)
                    query_vectors = mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu()
                    all_docs_vectors = []
                    dataloader = DataLoader(corpus, batch_size=bsz, shuffle=False)
                    for batch in dataloader:
                        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                        outputs = model(**inputs)
                        cur_docs_vectors = mean_pooling(outputs[0], inputs['attention_mask']).detach().cpu()
                        all_docs_vectors.append(cur_docs_vectors)
                    all_docs_vectors = np.concatenate(all_docs_vectors, axis=0)
                    scores = (query_vectors @ all_docs_vectors.T).squeeze()
                
            elif retriever == 'flat-stella':
                tokenizer, model, vector_linear = self.retriever_model
                with torch.no_grad():
                    input_data = tokenizer([query], padding="longest", truncation=True, max_length=512, return_tensors="pt")
                    input_data = {k: v.to(model.device) for k, v in input_data.items()}
                    attention_mask = input_data["attention_mask"]
                    last_hidden_state = model(**input_data, use_cache=False)[0]
                    last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
                    query_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                    query_vectors = normalize(vector_linear(query_vectors).detach().cpu())
                with torch.no_grad():
                    all_docs_vectors = []
                    dataloader = DataLoader(corpus, batch_size=bsz, shuffle=False)
                    for batch in dataloader:
                        input_data = tokenizer(batch, padding="longest", truncation=True, max_length=512, return_tensors="pt")
                        input_data = {k: v.to(model.device) for k, v in input_data.items()}
                        attention_mask = input_data["attention_mask"]
                        last_hidden_state = model(**input_data, use_cache=False)[0]
                        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
                        docs_vectors = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
                        docs_vectors = normalize(vector_linear(docs_vectors).detach().cpu())
                        all_docs_vectors.append(docs_vectors)
                    all_docs_vectors = np.concatenate(all_docs_vectors, axis=0)
                scores = torch.tensor((query_vectors @ all_docs_vectors.T).squeeze())

            elif retriever == 'flat-gte':
                def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
                    if left_padding:
                        return last_hidden_states[:, -1]
                    else:
                        sequence_lengths = attention_mask.sum(dim=1) - 1
                        batch_size = last_hidden_states.shape[0]
                        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
                    
                def get_detailed_instruct(task_description: str, query: str) -> str:
                    return f'Instruction: {task_description}\nQuery: {query}'

                tokenizer, model = self.retriever_model
                task = 'Given a query about personal information, retrieve relevant chat history that answer the query.'
                with torch.no_grad():
                    all_vectors = []
                    dataloader = DataLoader([get_detailed_instruct(task, query)] + corpus, batch_size=bsz, shuffle=False)
                    for batch in dataloader:
                        batch_dict = tokenizer(batch, max_length=8192, padding=True, truncation=True, return_tensors='pt')
                        batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}
                        outputs = model(**batch_dict, use_cache=False)
                        embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
                        all_vectors.append(embeddings)
                all_vectors = torch.cat(all_vectors, dim=0)
                all_vectors = F.normalize(all_vectors, p=2, dim=1)
                scores = (all_vectors[:1] @ all_vectors[1:].T).squeeze()

            else:
                raise NotImplementedError

            return scores.argsort(descending=True)
        
        else:
            raise NotImplementedError


def process_item_flat_index(data, granularity, sess_id, timestamp):
    corpus = []

    if granularity == 'session':
        text = ' '.join([interact['content'] for interact in data if interact['role'] == 'user'])
        corpus.append(text)
        ids = [sess_id]
        if 'answer' in sess_id and all([not turn['has_answer'] for turn in [x for x in data if x['role'] == 'user']]):
            ids = [sess_id.replace('answer', 'noans')]
    elif granularity == 'turn':
        ids = []
        for i_turn, turn in enumerate(data):
            if turn['role'] == 'user':
                corpus.append(turn['content'])
                if 'answer' not in sess_id:
                    ids.append(sess_id + '_' + str(i_turn+1))
                else:
                    assert 'has_answer' in turn
                    assert turn['has_answer'] in [True, False]
                    if turn['has_answer']:
                        ids.append(sess_id + '_' + str(i_turn+1))
                    else:
                        ids.append((sess_id + '_' + str(i_turn+1)).replace('answer', 'noans'))
                        assert 'answer' not in ids[-1]
    else:
        raise NotImplementedError
    
    return corpus, ids, [timestamp for _ in corpus]


def batch_get_retrieved_context_and_eval(entry_list, args, index_expansion_result_cache=None):
    gpu_id = int(mp.current_process().name.split('-')[-1]) - 1
    if args.retriever in ['flat-bm25', 'flat-contriever', 'flat-stella', 'flat-gte', 'oracle']:
        retriever_master = DenseRetrievalMaster(args, gpu_id=gpu_id)
    else:
        raise NotImplementedError

    results = []
    for entry in tqdm(entry_list):
        # step 1: prepare corpus index (with potential index expansion)
        corpus, corpus_ids, corpus_timestamps = [], [], []
        for cur_sess_id, sess_entry, ts in zip(entry['haystack_session_ids'], entry['haystack_sessions'], entry['haystack_dates']):
            cur_items, cur_ids, cur_ts = process_item_flat_index(sess_entry, args.granularity, cur_sess_id, ts)
            corpus += cur_items
            corpus_ids += cur_ids
            corpus_timestamps += cur_ts

        if args.index_expansion_method != 'none':
            if index_expansion_result_cache is not None:
                if 'session' in args.index_expansion_method:
                    for cur_sess_id, sess_entry, ts in zip(entry['haystack_session_ids'], entry['haystack_sessions'], entry['haystack_dates']):
                        cur_item_expansions = fetch_expansion_from_cache(index_expansion_result_cache, cur_sess_id)
                        #print(cur_sess_id)
                        #print(cur_item_expansions)
                        corpus, corpus_ids, corpus_timestamps = resolve_expansion(args.index_expansion_method, args.index_expansion_result_join_mode,
                                                                                  corpus, corpus_ids, corpus_timestamps,
                                                                                  cur_item_expansions, cur_sess_id, ts)
                elif 'turn' in args.index_expansion_method:
                    for cur_sess_id, sess_entry, ts in zip(entry['haystack_session_ids'], entry['haystack_sessions'], entry['haystack_dates']):
                        for cur_turn_id, cur_turn_content in enumerate(sess_entry):
                            if cur_turn_content['role'] == 'user':
                                cur_item_expansions = fetch_expansion_from_cache(index_expansion_result_cache, cur_sess_id + f'_{cur_turn_id+1}')
                                corpus, corpus_ids, corpus_timestamps = resolve_expansion(args.index_expansion_method, args.index_expansion_result_join_mode,
                                                                                          corpus, corpus_ids, corpus_timestamps,
                                                                                          cur_item_expansions, cur_sess_id + f'_{cur_turn_id+1}', ts)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError

        correct_docs = list(set([doc_id for doc_id in corpus_ids if "answer" in doc_id]))

        # step 2: run retrieval
        query = entry['question']
        if args.retriever in ['flat-bm25', 'flat-contriever', 'flat-stella', 'flat-gte']:
            rankings = retriever_master.run_flat_retrieval(query, args.retriever, corpus)
        elif args.retriever in ['oracle']:
            correct_idx, incorrect_idx = [], []
            for i_doc, cid in enumerate(corpus_ids):
                if cid in correct_docs:
                    correct_idx.append(i_doc)
                else:
                    incorrect_idx.append(i_doc)
            rankings = correct_idx + incorrect_idx
        else:
            raise NotImplementedError
        
        # step 3: record evaluation metrics
        cur_results = {
            'question_id': entry['question_id'],
            'question_type': entry['question_type'],
            'question': entry['question'],
            'answer': entry['answer'],
            'question_date': entry['question_date'],
            'haystack_dates': entry['haystack_dates'],
            'haystack_sessions': entry['haystack_sessions'],
            'haystack_session_ids': entry['haystack_session_ids'],
            'answer_session_ids': entry['answer_session_ids'],
            'retrieval_results': {
                'query': query,
                'ranked_items': [
                    {
                        'corpus_id': corpus_ids[rid],
                        'text': corpus[rid],
                        'timestamp': corpus_timestamps[rid]
                    }
                    for rid in rankings
                ],
                'metrics': {
                    'session': {},
                    'turn': {}
                }
            }
        }
        for k in [1, 3, 5, 10, 30, 50]:
            recall_any, recall_all, ndcg_any = evaluate_retrieval(rankings, correct_docs, corpus_ids, k=k)
            cur_results['retrieval_results']['metrics'][args.granularity].update({
                'recall_any@{}'.format(k): recall_any,
                'recall_all@{}'.format(k): recall_all,
                'ndcg_any@{}'.format(k): ndcg_any
            })
            if args.granularity == 'turn':
                recall_any, recall_all, ndcg_any = evaluate_retrieval_turn2session(rankings, correct_docs, corpus_ids, k=k)
                cur_results['retrieval_results']['metrics']['session'].update({
                    'recall_any@{}'.format(k): recall_any,
                    'recall_all@{}'.format(k): recall_all,
                    'ndcg_any@{}'.format(k): ndcg_any
                })

        results.append(cur_results)

    return results


def main(args):
    check_args(args)
    
    outfile_prefix = get_outfile_prefix(args)
    out_file = args.out_dir + '/' + outfile_prefix + '_retrievallog_{}_{}'.format(args.granularity, args.retriever)
    # log_file = out_file + '.log'
    # log_f, out_f = open(log_file, 'w'), open(out_file, 'w')
    out_f = open(out_file, 'w')
    
    # load data and cache
    in_data = json.load(open(args.in_file))
    n_has_abstention = len([x for x in in_data if '_abs' in x['question_id']])
    if n_has_abstention > 0:
        print("Warning: found {} abstention instances within the data".format(n_has_abstention))

    index_expansion_result_cache = None
    if args.index_expansion_result_cache is not None and args.index_expansion_result_cache != 'none':
        index_expansion_result_cache = json.load(open(args.index_expansion_result_cache))
        print("Loaded pre-computed expansions from", args.index_expansion_result_cache)
    
    # multiprocessing
    num_processes = torch.cuda.device_count()
    if 'bm25' in args.retriever:
        num_processes = 10
    print('Setting num processes = {} with retriever {}'.format(num_processes, args.retriever))
    mp.set_start_method('spawn')
    pool = mp.Pool(num_processes)
    worker = partial(batch_get_retrieved_context_and_eval, args=args, index_expansion_result_cache=index_expansion_result_cache)

    # chunk the data into batches
    in_data_chunked = []
    chunk_size = len(in_data) // num_processes
    remainder = len(in_data) % num_processes
    start = 0
    for i in range(num_processes):
        end = start + chunk_size + (1 if i < remainder else 0)
        in_data_chunked.append(in_data[start:end])
        start = end

    results = []
    for d in pool.imap_unordered(worker, in_data_chunked):
        results += d

    pool.close()

    # for cur_results in results:
    #     print(json.dumps(cur_results), file=log_f)
        
    # log
    averaged_results = {
        'session': {},
        'turn': {}
    }
    ignored_qs_abstention, ignored_qs_no_target = set(), set()
    for t in ['session', 'turn']:
        for k in results[0]['retrieval_results']['metrics'][t]:
            try:
                results_list = []
                for eval_entry in results:
                    # will skip abstention instances for reporting the metric
                    if '_abs' in eval_entry['question_id']:
                        ignored_qs_abstention.add(eval_entry['question_id'])
                        continue
                    # will also skip instances with no target labels
                    if not any(('has_answer' in turn) and (turn['has_answer']) for turn in [x for y in eval_entry['haystack_sessions'] for x in y if x['role'] == 'user']):
                        ignored_qs_no_target.add(eval_entry['question_id'])
                        continue
                    results_list.append(eval_entry['retrieval_results']['metrics'][t][k])
                    
                averaged_results[t][k] = np.mean(results_list)
            except:
                continue
    print('Ignored {} instances due to abstention: {}'.format(len(ignored_qs_abstention), ignored_qs_abstention))
    print('Additionally ignored {} instances due to no target turns from the user side: {}'.format(len(ignored_qs_no_target), ignored_qs_no_target))
    print(json.dumps(averaged_results))

    # save results
    for entry in results:
        print(json.dumps(entry), file=out_f)

    # log_f.close()
    out_f.close()

    
if __name__ == '__main__':
    args = parse_args()
    main(args)
