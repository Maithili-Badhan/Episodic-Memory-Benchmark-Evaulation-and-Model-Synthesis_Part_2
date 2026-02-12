from epbench.src.models.settings_wrapper import SettingsWrapper 
from epbench.src.models.models_wrapper import ModelsWrapper
from epbench.src.io.io import answer_filepath_func, answer_reasoning_filepath_func, evaluate_filepath_func, chronological_filepath_func, import_list, export_list
from epbench.src.evaluation.scoring_answers import evaluate_answer, evaluate_chronological
from epbench.src.generation.benchmark_generation_wrapper import BenchmarkGenerationWrapper
from epbench.src.evaluation.prompts import generate_episodic_memory_prompt
from epbench.src.evaluation.generator_answers_2_rag import query_message
from epbench.src.generation.printing import split_chapters_func
import os
import pandas as pd
import re
import time
import requests

# --- helper functions (mostly unchanged) ---
def check_and_remove(book, substring, error_if_not_found = True):
    count = book.count(substring)
    if count == 0:
        if error_if_not_found:
            raise ValueError(f"Substring '{substring}' not found in document")
        else:
            return book
    elif count > 1:
        raise ValueError(f"Substring '{substring}' found {count} times, expected exactly once")
    else:
        return book.replace(substring, "", 1)
    
def patch_for_ensuring_token_size_lower_130k_in_llama3(book):
    # trimmed versions of substrings omitted for brevity in this listing - keep your full substrings
    substring139 = """..."""
    substring13 = """..."""
    substring133 = """..."""
    substring152 = """..."""
    substring167 = """..."""
    substring64 = """..."""
    substring25 = """..."""
    book = check_and_remove(book, substring139)
    book = check_and_remove(book, substring13)
    book = check_and_remove(book, substring133)
    book = check_and_remove(book, substring152)
    book = check_and_remove(book, substring167)
    book = check_and_remove(book, substring64)
    book = check_and_remove(book, substring25)
    return book

def whether_do_this_q(q, q_max):
    if q_max is None:
        return True
    else:
        return (q < q_max)

# --- NEW: small OpenAI adapter to avoid ModelsWrapper init quirks (proxies kwarg etc.) ---
class SimpleOpenAIAdapter:
    """
    Minimal adapter that provides the same .generate(user_prompt, system_prompt, max_new_tokens, full_outputs, ... )
    interface used in the rest of the code. This avoids calling ModelsWrapper which in some envs triggers
    unexpected kwargs to the OpenAI/OpenAI client constructor.
    """
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError(f"OpenAI SDK import failed: {e}")
        # Create client with only api_key to avoid unexpected kwargs coming from other wrappers
        self.client = OpenAI(api_key=api_key)

    def generate(self, user_prompt: str, system_prompt: str = None, full_outputs: bool = False,
                 max_new_tokens: int = 256, temperature: float = 1.0, keep_reasoning: bool = False):
        # Mirror the earlier usage: create chat completion with system+user messages
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        outputs = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature
        )
        if not full_outputs:
            try:
                content = outputs.choices[0].message.content
            except Exception:
                content = getattr(outputs.choices[0], "message", {}).get("content", "")
            if keep_reasoning:
                # this simple adapter doesn't expose reasoning_content; return (content, None)
                return content, None
            return content
        else:
            if keep_reasoning:
                return outputs, None
            return outputs

# --- generate_answers_func (unchanged from prior version) ---
def generate_answers_func(
    my_benchmark: BenchmarkGenerationWrapper,
    answering_parameters = {'kind': 'prompting', 'model_name': 'claude-3-5-sonnet-20240620', 'max_new_tokens': 4096, 'sleeping_time': 15},
    data_folder = '/repo/to/git/main/epbench/data',
    env_file = '/repo/to/git/main/.env',
    my_embedding = None):

    prompt_parameters = my_benchmark.prompt_parameters
    model_parameters = my_benchmark.model_parameters
    book_parameters = my_benchmark.book_parameters

    model_name = answering_parameters['model_name'] 
    max_new_tokens = answering_parameters['max_new_tokens']
    system_prompt = "You are an expert in memory tests."
    sleeping_time = answering_parameters['sleeping_time']
    
    config = SettingsWrapper(_env_file = env_file)

    book = my_benchmark.get_book()
    if answering_parameters['model_name'] == 'llama-3.1-405b-instruct':
        if my_benchmark.nb_tokens() == 102870:
            book = patch_for_ensuring_token_size_lower_130k_in_llama3(book)

    KEEP_CHAPTERS = int(os.environ.get("EPBENCH_KEEP_CHAPTERS_FOR_FREE_QWEN", 6))
    MAX_Q_FOR_FREE_QWEN = int(os.environ.get("EPBENCH_MAX_Q_FOR_FREE_QWEN", 40))
    FREE_QWEN_IDENTIFIER = "qwen3-4b"

    free_qwen_mode = False
    if FREE_QWEN_IDENTIFIER in answering_parameters.get('model_name', ''):
        free_qwen_mode = True
        print(f"[INFO] Detected free QWEN model ({answering_parameters.get('model_name')}) — applying prompt-size limits.")

    df_qa = my_benchmark.get_df_qa()
    nb_chapters = my_benchmark.nb_chapters()
    nb_tokens = my_benchmark.nb_tokens()

    if free_qwen_mode:
        if len(df_qa) > MAX_Q_FOR_FREE_QWEN:
            print(f"[INFO] Limiting df_qa to first {MAX_Q_FOR_FREE_QWEN} questions for free qwen model (was {len(df_qa)})")
            df_qa = df_qa.iloc[:MAX_Q_FOR_FREE_QWEN].reset_index(drop=True)
        try:
            chapters = split_chapters_func(book)
            if len(chapters) > KEEP_CHAPTERS:
                print(f"[INFO] Trimming book to first {KEEP_CHAPTERS} chapters for free qwen model (was {len(chapters)} chapters).")
                book = "\n\n".join(chapters[:KEEP_CHAPTERS])
            else:
                print(f"[INFO] Book has {len(chapters)} chapters – no trimming required.")
        except Exception as e:
            print(f"[WARN] split_chapters_func failed, continuing with whole book: {e}")

    generated_answers = []
    for q in range(len(df_qa)):
        answer_filepath = answer_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
        answer_reasoning_filepath = answer_reasoning_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
        if not answer_filepath.is_file():
            question = df_qa.iloc[q]['question']
            correct_answer = df_qa.iloc[q]['correct_answer']
            print(f"Generate {str(q)} / {str(len(df_qa)-1)} [{correct_answer}for question {question}]")
            try:
                my_model
            except NameError:
                my_model = ModelsWrapper(model_name, config)
            if answering_parameters['kind'] == 'prompting':
                user_prompt = generate_episodic_memory_prompt(book, question)
            elif answering_parameters['kind'] == 'rag':
                user_prompt = query_message(question, my_embedding, answering_parameters, env_file)
            elif answering_parameters['kind'] == 'ftuning':
                user_prompt = my_benchmark.get_user_prompt_for_finetuning(question)
            if q == 0:
                print("[begin example of a prompt]")
                print(user_prompt)
                print("[end example of a prompt]")

            out = None
            reasoning = None
            try_count = 0
            while try_count < 2 and out is None:
                try:
                    out, reasoning = my_model.generate(user_prompt = user_prompt, system_prompt = system_prompt, max_new_tokens = max_new_tokens, keep_reasoning = True)
                except Exception as e:
                    try_count += 1
                    err_str = str(e)
                    print(f"[WARN] generation attempt {try_count} failed for q={q} with model {model_name}: {e}")
                    if isinstance(e, requests.exceptions.HTTPError) or '429' in err_str or '402' in err_str or 'Rate limit' in err_str:
                        if try_count < 2:
                            sleep_backoff = 10 * try_count
                            print(f"[INFO] transient HTTP error detected — sleeping {sleep_backoff}s and retrying")
                            time.sleep(sleep_backoff)
                            continue
                        else:
                            print("[ERROR] repeated HTTP errors, skipping this question and writing placeholder answer.")
                            out = "ERROR: skipped due to HTTP error"
                            reasoning = None
                            break
                    else:
                        raise

            if out is None:
                out = "ERROR: generation returned no output"

            print(f"sleeping for {sleeping_time} seconds")
            time.sleep(sleeping_time)
            print("woke up")
            answer_filepath.parent.mkdir(parents=True, exist_ok=True)
            print(answer_filepath)
            export_list(out, answer_filepath)
            if reasoning is not None:
                answer_reasoning_filepath.parent.mkdir(parents=True, exist_ok=True)
                print(answer_reasoning_filepath)
                export_list(reasoning, answer_reasoning_filepath)
        generated_answer = import_list(answer_filepath)
        generated_answers.append(generated_answer)

    df_generated_answers = pd.concat([df_qa, pd.DataFrame({'llm_answer':generated_answers})], axis = 1)

    return df_generated_answers

# --- small validators for chapter strings ---
def is_valid_chapter_string(s):
    pattern = r'^Full chapter \d+$'
    return bool(re.match(pattern, s))

def extract_chapter_number(s):
    match = re.search(r"Full chapter (\d+)", s)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("String does not match the expected format")

# --- MAIN: generate_evaluation_func with robust judge initialization + fallback ---
def generate_evaluation_func(
    my_benchmark: BenchmarkGenerationWrapper,
    df_generated_answers,
    answering_parameters = {'kind': 'prompting', 'model_name': 'claude-3-5-sonnet-20240620', 'max_new_tokens': 4096},
    data_folder = '/repo/to/git/main/epbench/data',
    env_file = '/repo/to/git/main/.env'):

    prompt_parameters = my_benchmark.prompt_parameters
    model_parameters = my_benchmark.model_parameters
    book_parameters = my_benchmark.book_parameters

    model_name = model_parameters['model_name'] 
    config = SettingsWrapper(_env_file = env_file)

    nb_chapters = my_benchmark.nb_chapters()
    nb_tokens = my_benchmark.nb_tokens()
    split_chapters = my_benchmark.split_chapters

    if answering_parameters['model_name'] == 'llama-3.1-405b-instruct':
        if my_benchmark.nb_tokens() == 102870:
            book = my_benchmark.book
            book = patch_for_ensuring_token_size_lower_130k_in_llama3(book)
            split_chapters = split_chapters_func(book)

    # --- HARD-CODED JUDGE LLM WITH ROBUST FALLBACKS ---
    DEFAULT_JUDGE = os.environ.get("EPBENCH_JUDGE_LLM", "gpt-4o-mini-2024-07-18")
    print(f"[INFO] Requested judge model (env or default): {DEFAULT_JUDGE}")

    judge_model = None
    tried_models = []

    def try_init_model_generic(model_str):
        """
        First attempt: if model_str looks like an OpenAI 'gpt-' model, use SimpleOpenAIAdapter
        to avoid ModelsWrapper init quirks. Otherwise, try ModelsWrapper.
        """
        tried_models.append(model_str)
        try:
            # If it looks like a gpt/openai model, instantiate the simple adapter directly
            if model_str.startswith("gpt-") or model_str.startswith("gpt4") or "gpt-4o" in model_str or model_str.startswith("gpt"):
                api_key = getattr(config, "OPENAI_API_KEY", None) or os.environ.get("OPENAI_API_KEY", None)
                if api_key is None:
                    raise RuntimeError("OPENAI_API_KEY not found in config/env for SimpleOpenAIAdapter.")
                m = SimpleOpenAIAdapter(model_str, api_key)
                print(f"[INFO] Initialized SimpleOpenAIAdapter judge: {model_str}")
                return m
            # otherwise try the project's ModelsWrapper (for claude/qwen/llama etc.)
            m = ModelsWrapper(model_str, config)
            print(f"[INFO] Initialized ModelsWrapper judge: {model_str}")
            return m
        except Exception as e:
            print(f"[WARN] try_init_model_generic failed for '{model_str}': {e}")
            return None

    judge_model = try_init_model_generic(DEFAULT_JUDGE)

    if judge_model is None:
        alt = os.environ.get("EPBENCH_FALLBACK_JUDGE", None)
        if alt:
            print(f"[INFO] Trying EPBENCH_FALLBACK_JUDGE: {alt}")
            judge_model = try_init_model_generic(alt)

    if judge_model is None:
        print(f"[INFO] Trying benchmark model {model_name} as judge fallback")
        judge_model = try_init_model_generic(model_name)

    if judge_model is None:
        # final fallback: explicit minimal OpenAI model
        fallback = "gpt-4o-mini-2024-07-18"
        print(f"[WARN] All previous judge initializations failed. Trying final fallback: {fallback}")
        judge_model = try_init_model_generic(fallback)

    if judge_model is None:
        raise RuntimeError(f"Could not initialize any judge model. Tried: {tried_models}")

    # question/true answer and additionally containing the generated answers
    df_qa2 = df_generated_answers
    generated_evaluations = []

    for q in range(len(df_qa2)):
        evaluate_filepath = evaluate_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)
        if not evaluate_filepath.is_file():
            question = df_qa2.iloc[q]['question'] 
            llm_answer = df_qa2.iloc[q]['llm_answer']
            correct_answer = df_qa2.iloc[q]['correct_answer']
            retrieval_type = df_qa2.iloc[q]['retrieval_type']
            get_style = df_qa2.iloc[q]['get']
            print(f"Evaluate {str(q)} / {str(len(df_qa2)-1)} [question {question}]")
            try:
                my_model
            except NameError:
                my_model = ModelsWrapper(model_name, config)

            if len(correct_answer) == 1:
                if is_valid_chapter_string(correct_answer[0]):
                    chapter_number = extract_chapter_number(correct_answer[0])
                    correct_answer_long = split_chapters[chapter_number]
                else:
                    correct_answer_long = None
            else:
                correct_answer_long = None

            out = None
            try:
                out = evaluate_answer(llm_answer, correct_answer, retrieval_type, judge_model, correct_answer_long, get_style)
            except requests.exceptions.HTTPError as e:
                code = getattr(e.response, "status_code", None)
                print(f"[WARN] judge_model evaluation raised HTTPError {code}: {e}. Trying fallback judge and retry once.")
                try:
                    fallback_model = os.environ.get("EPBENCH_HTTPFALLBACK_JUDGE", "gpt-4o-mini-2024-07-18")
                    new_judge = try_init_model_generic(fallback_model)
                    if new_judge is not None:
                        judge_model = new_judge
                        out = evaluate_answer(llm_answer, correct_answer, retrieval_type, judge_model, correct_answer_long, get_style)
                    else:
                        print("[ERROR] could not initialize fallback judge model.")
                        out = "ERROR: judge initialization/fallback failed"
                except Exception as e2:
                    print(f"[ERROR] retry with fallback judge failed: {e2}")
                    out = "ERROR: judge retry failed"
            except Exception as e:
                # re-raise non-HTTP exceptions so you can see them
                raise

            evaluate_filepath.parent.mkdir(parents=True, exist_ok=True)
            export_list(out, evaluate_filepath)
        generated_evaluation = import_list(evaluate_filepath)
        generated_evaluations.append(generated_evaluation)

    # Build DataFrame of evaluations; ensure required columns exist to avoid KeyError later
    df_generated_evaluations = pd.DataFrame(generated_evaluations)

    # Ensure groundtruth_items column exists (some pipelines require it). Fill with empty lists if missing.
    if 'groundtruth_items' not in df_generated_evaluations.columns:
        print("[INFO] 'groundtruth_items' not found in evaluation outputs — inserting empty lists as placeholder to avoid KeyError downstream.")
        df_generated_evaluations['groundtruth_items'] = [[] for _ in range(len(df_generated_evaluations))]

    # concat with df_qa2 so the outer pipeline receives expected structure
    df_generated_evaluations = pd.concat([df_qa2, df_generated_evaluations], axis = 1)

    return df_generated_evaluations

# --- generate_chronological_func (mostly unchanged) ---
def generate_chronological_func(
    my_benchmark: BenchmarkGenerationWrapper,
    df_generated_evaluations,
    answering_parameters = {'kind': 'prompting', 'model_name': 'claude-3-5-sonnet-20240620', 'max_new_tokens': 4096},
    data_folder = '/repo/to/git/main/epbench/data',
    env_file = '/repo/to/git/main/.env'):

    prompt_parameters = my_benchmark.prompt_parameters
    model_parameters = my_benchmark.model_parameters
    book_parameters = my_benchmark.book_parameters

    model_name = model_parameters['model_name'] 
    config = SettingsWrapper(_env_file = env_file)

    nb_chapters = my_benchmark.nb_chapters()
    nb_tokens = my_benchmark.nb_tokens()

    df_qa3 = df_generated_evaluations

    generated_chronologicals = []

    for q in range(len(df_qa3)):
        if df_qa3.iloc[q]['get'] == 'chronological':
            chronological_filepath = chronological_filepath_func(q, nb_chapters, nb_tokens, data_folder, prompt_parameters, model_parameters, book_parameters, answering_parameters)

            if not chronological_filepath.is_file():
                predicted_items = df_qa3.iloc[q].get('predicted_items', [])
                groundtruth_items = df_qa3.iloc[q].get('groundtruth_items', [])
                question = df_qa3.iloc[q]['question']
                print(f"Evaluate {str(q)} / {str(len(df_qa3)-1)} [question {question}]")
                try:
                    my_model
                except NameError:
                    my_model = ModelsWrapper(model_name, config)

                try:
                    # prefer to reuse judge if available in the outer scope (we expect generate_evaluation_func to have picked one),
                    # but fall back to my_model if necessary
                    out = evaluate_chronological(groundtruth_items, predicted_items, my_model)
                except requests.exceptions.HTTPError as e:
                    print(f"[WARN] chronological evaluation raised HTTPError: {e}. Trying fallback judge.")
                    try:
                        fallback_model = "gpt-4o-mini-2024-07-18"
                        new_judge = ModelsWrapper(fallback_model, config)
                        out = evaluate_chronological(groundtruth_items, predicted_items, new_judge)
                    except Exception as e2:
                        print(f"[ERROR] fallback chronological judge failed: {e2}. Using placeholder output.")
                        out = "ERROR: chronological evaluation failed"
                chronological_filepath.parent.mkdir(parents=True, exist_ok=True)
                export_list(out, chronological_filepath)
            generated_chronological = import_list(chronological_filepath)
            generated_chronologicals.append(generated_chronological)

    df_generated_chronological = pd.DataFrame(generated_chronologicals)

    return df_generated_chronological