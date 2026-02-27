# epbench/src/evaluation/evaluation_wrapper.py

from epbench.src.generation.benchmark_generation_wrapper import BenchmarkGenerationWrapper
from epbench.src.evaluation.generator_answers_1_prompting import (
    generate_answers_func,
    generate_evaluation_func,
    generate_chronological_func,
)
from epbench.src.evaluation.scoring_answers import update_policy_of_evaluation_to
import numpy as np
# for rag
from epbench.src.generation.generate_3_secondary_entities import count_tokens
from epbench.src.evaluation.generator_answers_2_rag import embed_chunks
from epbench.src.io.io import answer_dirpath_func, import_list, export_list, export_jsonl
import pandas as pd
import ast
# for ftuning (lazy import, done only if needed)
# from openai import OpenAI  # removed global import to avoid client init issues
from epbench.src.models.settings_wrapper import SettingsWrapper
from scipy.stats import kendalltau

# defensive helper to ensure chrono DF has expected columns and list-like cells
import json, re

def ensure_chrono_columns(df):
    """
    Ensure df has expected chronological columns as list cells:
      - 'groundtruth_indexes'
      - 'predicted_indexes'
      - 'groundtruth_items'
      - 'predicted_items'
    If missing, insert safe defaults so downstream code doesn't KeyError.
    """
    import pandas as _pd
    import numpy as _np

    expected_cols = ['groundtruth_indexes', 'predicted_indexes', 'groundtruth_items', 'predicted_items']

    # If df is None, return an empty DataFrame with expected columns
    if df is None:
        return _pd.DataFrame({c: [] for c in expected_cols})

    # If df isn't a DataFrame, try to coerce
    if not hasattr(df, 'columns'):
        try:
            df = _pd.DataFrame(df)
        except Exception:
            df = _pd.DataFrame({c: [] for c in expected_cols})

    # Insert missing cols with default list values matching df length
    n = len(df)
    for col in expected_cols:
        if col not in df.columns:
            print(f"[WARN] ensure_chrono_columns: missing column '{col}' — inserting default empty lists.")
            df[col] = [[] for _ in range(n)]

    # Normalize each expected column to a list per-row
    def to_list_cell(x):
        # None or nan -> []
        if x is None:
            return []
        if isinstance(x, float) and _np.isnan(x):
            return []
        # proper lists -> keep
        if isinstance(x, (list, tuple)):
            return list(x)
        # if it's a string that looks like a list, try json.loads
        if isinstance(x, str):
            s = x.strip()
            if s.startswith('[') and s.endswith(']'):
                try:
                    return json.loads(s)
                except Exception:
                    # fallback: extract ints or words
                    nums = re.findall(r'-?\d+', s)
                    if nums:
                        return [int(n) for n in nums]
                    # try split by comma
                    return [elem.strip().strip("'\"") for elem in s.strip('[]').split(',') if elem.strip()]
            # if string of numbers, parse ints
            nums = re.findall(r'-?\d+', s)
            if nums:
                return [int(n) for n in nums]
            # otherwise return single-element list
            return [s]
        # scalar ints -> wrap in list
        if isinstance(x, (int, np.integer)):
            return [int(x)]
        # unknown types -> stringify in list
        return [str(x)]

    for col in expected_cols:
        df[col] = df[col].apply(to_list_cell)

    return df


class EvaluationWrapper:
    def __init__(
        self,
        my_benchmark: BenchmarkGenerationWrapper,
        answering_parameters=None,
        data_folder='/repo/to/git/main/epbench/data',
        env_file='/repo/to/git/main/.env',
    ):
        if answering_parameters is None:
            answering_parameters = {
                'kind': 'prompting',
                'model_name': 'claude-3-5-sonnet-20240620',
                'max_new_tokens': 4096,
                'sleeping_time': 15,
                'policy': 'original',
            }

        # save the input
        self.my_benchmark = my_benchmark
        self.data_folder = data_folder
        self.env_file = env_file
        self.policy = answering_parameters.get('policy', 'original')

        # convenience
        kind = answering_parameters.get('kind', 'prompting')

        # generated answers
        if kind == 'prompting':
            self.df_generated_answers = generate_answers_func(my_benchmark, answering_parameters, data_folder, env_file)
        elif kind == 'rag':
            # embedding flow
            # chunk book -> embed -> store embedding csv -> read embedding and continue
            self.my_chunks = my_benchmark.chunk_book(split=answering_parameters.get('embedding_chunk', 'chapter'))
            self.chunk_with_max_tokens = max([count_tokens(x) for x in self.my_chunks]) if len(self.my_chunks) > 0 else 0
            self.chunk_number = len(self.my_chunks)

            prompt_parameters = getattr(my_benchmark, 'prompt_parameters', {})
            model_parameters = getattr(my_benchmark, 'model_parameters', {})
            book_parameters = getattr(my_benchmark, 'book_parameters', {})
            nb_chapters = my_benchmark.nb_chapters()
            nb_tokens = my_benchmark.nb_tokens()
            answer_dirpath = answer_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
            embedding_filepath = answer_dirpath / "embedding.csv"

            # try to create embedding csv if not present
            if not embedding_filepath.is_file():
                try:
                    my_embedding = embed_chunks(self.my_chunks, answering_parameters, env_file)
                    answer_dirpath.mkdir(parents=True, exist_ok=True)
                    my_embedding.to_csv(embedding_filepath, index=False)
                    print(f"chunked into {self.chunk_number} chunks, the largest containing {self.chunk_with_max_tokens} tokens")
                    print(my_embedding)
                except Exception as e_embed:
                    # If embedding creation failed, try to proceed with deterministic/dummy embeddings if embed_chunks supports it
                    print(f"[WARN] embed_chunks failed: {repr(e_embed)}. Attempting to continue with deterministic or partial embedding output.")
                    # If embed_chunks raised, attempt to obtain any file that may exist; otherwise create minimal df
                    try:
                        answer_dirpath.mkdir(parents=True, exist_ok=True)
                        # create deterministic fallback DF of text + empty embedding list
                        my_embedding = pd.DataFrame({"text": self.my_chunks, "embedding": [[] for _ in self.my_chunks]})
                        my_embedding.to_csv(embedding_filepath, index=False)
                    except Exception as e2:
                        print(f"[ERROR] fallback writing embedding csv also failed: {repr(e2)}")
                        raise

            # read embedding file and coerce embedding column to list objects
            try:
                self.my_embedding = pd.read_csv(embedding_filepath)
                # if embedding column is stringified lists, convert them
                if 'embedding' in self.my_embedding.columns:
                    # ensure the embedding column contains lists (not strings)
                    def parse_cell(x):
                        if pd.isna(x):
                            return []
                        if isinstance(x, (list, tuple)):
                            return list(x)
                        if isinstance(x, str):
                            s = x.strip()
                            if s.startswith('[') and s.endswith(']'):
                                try:
                                    return ast.literal_eval(s)
                                except Exception:
                                    # fallback simple parse of numbers
                                    nums = re.findall(r'-?\d+\.?\d*', s)
                                    try:
                                        return [float(n) for n in nums]
                                    except Exception:
                                        return []
                            else:
                                # try extract numbers
                                nums = re.findall(r'-?\d+\.?\d*', s)
                                try:
                                    return [float(n) for n in nums]
                                except Exception:
                                    return []
                        # unknown -> return as single-element list
                        return [x]
                    self.my_embedding['embedding'] = self.my_embedding['embedding'].apply(parse_cell)
                else:
                    self.my_embedding['embedding'] = [[] for _ in range(len(self.my_embedding))]
            except Exception as e_read:
                print(f"[WARN] Could not read embedding csv properly: {repr(e_read)}. Creating fallback embedding dataframe.")
                self.my_embedding = pd.DataFrame({"text": self.my_chunks, "embedding": [[] for _ in self.my_chunks]})

            self.df_generated_answers = generate_answers_func(my_benchmark, answering_parameters, data_folder, env_file, self.my_embedding)
        elif kind == 'ftuning':
            # fine-tuning flow
            prompt_parameters = getattr(my_benchmark, 'prompt_parameters', {})
            model_parameters = getattr(my_benchmark, 'model_parameters', {})
            book_parameters = getattr(my_benchmark, 'book_parameters', {})
            nb_chapters = my_benchmark.nb_chapters()
            nb_tokens = my_benchmark.nb_tokens()
            ftuning_input_data_policy = answering_parameters.get('ftuning_input_data_policy', 'single')

            if ftuning_input_data_policy == 'single':
                answer_in_one_chapter_only = True
                ftuning_input = my_benchmark.build_fine_tuning_jsonl(answer_in_one_chapter_only)
            else:
                answer_in_one_chapter_only = False
                ftuning_input = my_benchmark.build_fine_tuning_jsonl(answer_in_one_chapter_only)

            answer_dirpath = answer_dirpath_func(nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
            jsonl_filename = f"ftuning_{ftuning_input_data_policy}_{nb_tokens}.jsonl"
            ftuning_input_filepath = answer_dirpath / jsonl_filename
            if not ftuning_input_filepath.is_file():
                answer_dirpath.mkdir(parents=True, exist_ok=True)
                export_jsonl(ftuning_input, ftuning_input_filepath)
            self.ftuning_input = ftuning_input
            config = SettingsWrapper(_env_file=env_file)

            # 2. uploading (lazy client init)
            if answering_parameters.get('ftuning_need_upload', False):
                is_existing_client = True
                try:
                    self.client
                except AttributeError:
                    is_existing_client = False
                if not is_existing_client:
                    # try lazy import and client creation; handle missing openai package gracefully
                    try:
                        from openai import OpenAI  # local import
                        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
                    except Exception as e_client_init:
                        print(f"[WARN] Could not initialize OpenAI client for ftuning upload: {repr(e_client_init)}. Skipping upload.")
                        self.client = None
                if self.client is not None:
                    try:
                        upload_ftuning_input(self.client, ftuning_input_filepath)
                    except Exception as e_upload:
                        print(f"[WARN] upload_ftuning_input failed: {repr(e_upload)}")

            ftuning_input_id_filepath = answer_dirpath / f"ftuning_{ftuning_input_data_policy}.id"
            if not ftuning_input_id_filepath.is_file():
                answer_dirpath.mkdir(parents=True, exist_ok=True)
                is_existing_client = True
                try:
                    self.client
                except AttributeError:
                    is_existing_client = False
                if not is_existing_client:
                    try:
                        from openai import OpenAI
                        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
                    except Exception as e_client_init2:
                        print(f"[WARN] Could not initialize OpenAI client to retrieve file id: {repr(e_client_init2)}. Skipping retrieval.")
                        self.client = None
                if self.client is not None:
                    try:
                        ftuning_input_id = retrieve_fileid(self.client, jsonl_filename)
                        export_list(ftuning_input_id, ftuning_input_id_filepath)
                        print(f"json `{jsonl_filename}` with file id `{self.ftuning_input_id}` for the `{ftuning_input_data_policy}` policy")
                    except Exception as e_retrieve:
                        print(f"[WARN] retrieve_fileid failed: {repr(e_retrieve)}")
            try:
                self.ftuning_input_id = import_list(ftuning_input_id_filepath)
            except Exception:
                self.ftuning_input_id = None

            if answering_parameters.get('ftuning_need_actual_tune', False):
                if not hasattr(self, 'client') or self.client is None:
                    try:
                        from openai import OpenAI
                        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
                    except Exception as e_client_init3:
                        print(f"[WARN] Cannot initialize OpenAI client to start fine-tune: {repr(e_client_init3)}. Skipping actual fine-tune.")
                        self.client = None
                if self.client is not None and self.ftuning_input_id is not None:
                    try:
                        self.client.fine_tuning.jobs.create(
                            training_file=self.ftuning_input_id,
                            model=answering_parameters.get('model_name'),
                            hyperparameters={
                                "batch_size": answering_parameters.get('batch_size'),
                                "learning_rate_multiplier": answering_parameters.get('learning_rate_multiplier'),
                                "n_epochs": answering_parameters.get('n_epochs'),
                            }
                        )
                        print('ongoing jobs')
                        print(self.client.fine_tuning.jobs.list(limit=10))
                        print('for cancelling, please use `object.cancel_job(corresponding_ftjob_as_above)`')
                    except Exception as e_tune:
                        print(f"[WARN] fine-tuning job creation failed: {repr(e_tune)}")

            # 5. answer the questions
            answering_parameters['model_name'] = answering_parameters.get('fine_tuned_model_name', answering_parameters.get('model_name'))
            self.df_generated_answers = generate_answers_func(my_benchmark, answering_parameters, data_folder, env_file)
        else:
            raise ValueError('unknown "kind", should be "prompting", "rag" or "ftuning"')

        # generated evaluation (given answers)
        df_generated_evaluations = generate_evaluation_func(my_benchmark, self.df_generated_answers, answering_parameters, data_folder, env_file)
        # possibly with a different policy for the final evaluation
        try:
            self.df_generated_evaluations = update_policy_of_evaluation_to(df_generated_evaluations, self.policy)
        except Exception as e_upd:
            print(f"[WARN] update_policy_of_evaluation_to failed: {repr(e_upd)}. Using raw df_generated_evaluations.")
            self.df_generated_evaluations = df_generated_evaluations

        # generated chronological (given evaluation)
        df_generated_chronological = generate_chronological_func(my_benchmark, self.df_generated_evaluations, answering_parameters, data_folder, env_file)
        # Defensive: ensure required chrono columns exist and cells are lists
        try:
            df_generated_chronological = ensure_chrono_columns(df_generated_chronological)
        except Exception as e_chrono:
            print(f"[WARN] ensure_chrono_columns failed: {repr(e_chrono)}. Creating empty chronological dataframe.")
            df_generated_chronological = pd.DataFrame({
                'groundtruth_indexes': [[]],
                'predicted_indexes': [[]],
                'groundtruth_items': [[]],
                'predicted_items': [[]],
            })
        self.df_generated_chronological = df_generated_chronological
        # compute kendall summaries defensively
        try:
            self.kendall_summaries_for_this_experiment = self.compute_kendall_summarise(df_generated_chronological, verbose=False)
        except Exception as e_kendall:
            print(f"[WARN] compute_kendall_summarise failed: {repr(e_kendall)}. Setting empty kendall summary.")
            self.kendall_summaries_for_this_experiment = pd.DataFrame({
                '#gt_with_len_2+': [0],
                '#exact_match_set_gt_with_pred': [0],
                '%_exact_match_set_gt_with_pred': ["0%"],
                'tau_exact_match_set_gt_with_pred': ["nan±nan"],
                '#partial_match_set_gt_with_pred': [0],
                '%_partial_match_set_gt_with_pred': ["0%"],
                'tau_partial_match_set_gt_with_pred': ["nan±nan"]
            })

    def cancel_job(self, ftjob_id='ftjob-wjldwdkjw0eiw'):
        if hasattr(self, 'client') and self.client is not None:
            try:
                self.client.fine_tuning.jobs.cancel(ftjob_id)
                print("cancelled")
            except Exception as e_cancel:
                print(f"[WARN] cancel_job failed: {repr(e_cancel)}")
        else:
            print("[INFO] No client available to cancel job.")

    def get_pretty_summary_relative_to(self, my_column='q_idx', metric='f1_score_lenient', sorting=False, filter_dict=None):
        if filter_dict is None:
            filter_dict = {}
        df = self.df_generated_evaluations

        if 'bins_items_correct_answer' in my_column:
            bins_count = [0, 1, 2, 3, 6, np.inf]
            labels_count = ['0', '1', '2', '3-5', '6+']
            df['bins_items_correct_answer'] = pd.cut(df['n_chapters_correct_answer'], bins=bins_count, include_lowest=True, right=False, labels=labels_count)

        if 'bins_items_correct_answer_few' in my_column:
            bins_count = [0, 2, np.inf]
            labels_count = ['0-1', '2+']
            df['bins_items_correct_answer_few'] = pd.cut(df['n_chapters_correct_answer'], bins=bins_count, include_lowest=True, right=False, labels=labels_count)

        if 'cue_size' in my_column:
            df['cue_size'] = [4 - elem.count('*') for elem in df['cue']]

        for column, value in filter_dict.items():
            df = df[df[column] == value]

        if my_column == '':
            result = df[[metric]].copy()
            result['count'] = 1
            result = result[['count', metric]]
            return result

        result = df.groupby(my_column, observed=False).agg({
            metric: ['mean', 'std', 'count']
        })
        result.columns = ['f1_score_mean', 'f1_score_std', 'count']
        result = result.reset_index()
        result[metric] = result.apply(lambda row: f"{row['f1_score_mean']:.2f}±{row['f1_score_std']:.2f}", axis=1)
        if sorting:
            result = result.sort_values(by='f1_score_mean', ascending=True)
        result = result.drop(columns=['f1_score_mean', 'f1_score_std'])
        result = result.reset_index(drop=True)
        return result

    def remove_duplicates_and_negative_one(self, lst):
        seen = set()
        return [x for x in lst if x != -1 and not (x in seen or seen.add(x))]

    def process_lists_and_compute_kendall_tau(self, l1, l2):
        # Step 1: Keep only elements that are in both lists
        common_elements = list(set(l1) & set(l2))
        nb_matches = len(common_elements)

        if nb_matches == 0:
            # Nothing in common -> no tau, zero matches
            return float('nan'), 0

        # Create new lists with only common elements, preserving original order
        result_l1 = [x for x in l1 if x in common_elements]
        result_l2 = [x for x in l2 if x in common_elements]

        # Step 2: Compute Kendall tau (discard the p-value)
        try:
            tau, _ = kendalltau(
                [result_l1.index(x) for x in common_elements],
                [result_l2.index(x) for x in common_elements]
            )
            return float(tau), nb_matches
        except Exception:
            return float('nan'), nb_matches

    def compute_kendall_summarise(self, df_generated_chronological, verbose=True):
        # Defensive copy and ensure correct structure
        chrono = df_generated_chronological.copy() if df_generated_chronological is not None else None
        chrono = ensure_chrono_columns(chrono)

        # sanitize predicted_indexes as ints, remove Nones/non-numeric
        def sanitize_indexes_cell(cell):
            out = []
            for el in cell:
                try:
                    if isinstance(el, str) and re.fullmatch(r'-?\d+', el.strip()):
                        out.append(int(el.strip()))
                    elif isinstance(el, (int, np.integer)):
                        out.append(int(el))
                    elif isinstance(el, float) and not np.isnan(el):
                        out.append(int(el))
                    else:
                        continue
                except Exception:
                    continue
            return out

        chrono['predicted_indexes'] = chrono['predicted_indexes'].apply(sanitize_indexes_cell)
        chrono['groundtruth_indexes'] = chrono['groundtruth_indexes'].apply(sanitize_indexes_cell)

        # remove duplicates and -1 entries while preserving order
        chrono['filtered_predicted_indexes'] = [self.remove_duplicates_and_negative_one(lst) for lst in chrono['predicted_indexes']]
        chrono['groundtruth_indexes_length'] = [len(x) for x in chrono['groundtruth_indexes']]

        chrono['tau'] = [self.process_lists_and_compute_kendall_tau(l1, l2)[0] for (l1, l2) in zip(chrono['filtered_predicted_indexes'], chrono['groundtruth_indexes'])]
        chrono['nb_matches'] = [self.process_lists_and_compute_kendall_tau(l1, l2)[1] for (l1, l2) in zip(chrono['filtered_predicted_indexes'], chrono['groundtruth_indexes'])]

        # all the samples in which the chronological order can be tested
        chrono_larger_than_one = chrono[chrono['groundtruth_indexes_length'] > 1]
        N = len(chrono_larger_than_one)

        if N == 0:
            if verbose:
                print("[INFO] No samples with groundtruth length > 1. Returning empty kendall summary.")
            kendall_summaries_for_this_experiment = pd.DataFrame({
                '#gt_with_len_2+': 0,
                '#exact_match_set_gt_with_pred': 0,
                '%_exact_match_set_gt_with_pred': "0%",
                'tau_exact_match_set_gt_with_pred': "nan±nan",
                '#partial_match_set_gt_with_pred': 0,
                '%_partial_match_set_gt_with_pred': "0%",
                'tau_partial_match_set_gt_with_pred': "nan±nan"
            }, index=[0])
            return kendall_summaries_for_this_experiment

        chrono_gt_larger_than_one_with_total_match = chrono_larger_than_one[chrono_larger_than_one['nb_matches'] == chrono_larger_than_one['groundtruth_indexes_length']]
        count_of_total_match = int((chrono_larger_than_one['nb_matches'] == chrono_larger_than_one['groundtruth_indexes_length']).sum())
        percentage_of_total_match = round((count_of_total_match / N) * 100, 2)

        tau_over_total_match = chrono_gt_larger_than_one_with_total_match['tau'].mean()
        sd_tau_over_total_match = chrono_gt_larger_than_one_with_total_match['tau'].std()

        chrono_gt_larger_than_one_with_match_greater_than_one = chrono_larger_than_one[chrono_larger_than_one['nb_matches'] > 1]
        count_of_match_greater_than_1 = int((chrono_larger_than_one['nb_matches'] > 1).sum())
        percentage_of_match_greater_than_1 = round((count_of_match_greater_than_1 / N) * 100, 2)

        tau_over_match_greater_than_one = chrono_gt_larger_than_one_with_match_greater_than_one['tau'].mean()
        sd_tau_over_match_greater_than_one = chrono_gt_larger_than_one_with_match_greater_than_one['tau'].std()

        def safe_round_str(val):
            try:
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    return "nan"
                return f"{round(float(val), 2)}"
            except Exception:
                return "nan"

        tau_total_str = f"{safe_round_str(tau_over_total_match)}±{safe_round_str(sd_tau_over_total_match)}"
        tau_partial_str = f"{safe_round_str(tau_over_match_greater_than_one)}±{safe_round_str(sd_tau_over_match_greater_than_one)}"

        if verbose:
            print(f"For the {count_of_total_match} samples with exact match between pred and gt sets (among the {N} with #gt>1, i.e. occurring for {percentage_of_total_match}%), the kendall tau average is {tau_total_str}.")
            print(f"For the {count_of_match_greater_than_1} samples with partial match >1 between pred and gt sets (among the {N} with #gt>1, i.e. occurring for {percentage_of_match_greater_than_1}%), the kendall tau average is {tau_partial_str}.")

        kendall_summaries_for_this_experiment = pd.DataFrame({
            '#gt_with_len_2+': N,
            '#exact_match_set_gt_with_pred': count_of_total_match,
            '%_exact_match_set_gt_with_pred': f"{percentage_of_total_match}%",
            'tau_exact_match_set_gt_with_pred': tau_total_str,
            '#partial_match_set_gt_with_pred': count_of_match_greater_than_1,
            '%_partial_match_set_gt_with_pred': f"{percentage_of_match_greater_than_1}%",
            'tau_partial_match_set_gt_with_pred': tau_partial_str
        }, index=[0])
        return kendall_summaries_for_this_experiment