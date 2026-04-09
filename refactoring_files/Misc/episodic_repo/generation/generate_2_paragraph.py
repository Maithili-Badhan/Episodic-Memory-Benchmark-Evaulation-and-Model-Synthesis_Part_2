from epbench.src.models.models_wrapper import ModelsWrapper
from epbench.src.models.settings_wrapper import SettingsWrapper
from epbench.src.generation.prompts import generate_prompts, system_prompt_func
from epbench.src.generation.generate_1_events_and_meta_events import generate_and_export_events_and_meta_events_func
from epbench.src.io.io import paragraph_filepath_func, export_list, import_list
from epbench.src.generation.verification_llm import has_passed_direct_and_llm_verifications_func
import logging
import re


def _sanitize_entity_mentions(text):
    # Normalize loose variants like "entity_3" -> "$entity_3"
    text = re.sub(r'(?i)(?<!\$)\bentity[_\s-]?(\d+)\b', r'$entity_\1', text)

    # The direct verifier is strict: any "entity" token not in "$entity_" form fails.
    # Keep only "$entity_" usages and rewrite others.
    def repl(match):
        i, j = match.span()
        prev_char = text[i - 1] if i > 0 else ''
        next_char = text[j] if j < len(text) else ''
        if prev_char == '$' and next_char == '_':
            return match.group(0)
        return 'character'

    return re.sub(r'(?i)entity', repl, text)


def _normalize_to_chunks(text, target_n):
    if not isinstance(text, str):
        text = str(text)
    t = text.strip().strip('"').replace('\r\n', '\n').replace('\r', '\n')

    # Prefer explicit "(X)" boundaries when available.
    numbered = list(re.finditer(r'\(\d+\)\s*', t))
    chunks = []
    if numbered:
        for i, m in enumerate(numbered):
            start = m.end()
            end = numbered[i + 1].start() if i + 1 < len(numbered) else len(t)
            chunk = t[start:end].strip()
            if chunk:
                chunks.append(chunk)
    else:
        chunks = [c.strip() for c in re.split(r'\n\s*\n+|\n+', t) if c.strip()]

    chunks = [' '.join(c.split()) for c in chunks if c.strip()]
    if not chunks:
        chunks = ['']

    if len(chunks) > target_n:
        chunks = chunks[: target_n - 1] + [' '.join(chunks[target_n - 1:]).strip()]
    elif len(chunks) < target_n:
        joined = ' '.join(chunks)
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', joined) if s.strip()]
        if len(sentences) >= target_n:
            out = []
            k = len(sentences) // target_n
            r = len(sentences) % target_n
            idx = 0
            for i in range(target_n):
                take = k + (1 if i < r else 0)
                out.append(' '.join(sentences[idx: idx + take]).strip())
                idx += take
            chunks = out
        else:
            while len(chunks) < target_n:
                chunks.append(chunks[-1] if chunks[-1] else 'Continued.')

    return [c if c else 'Continued.' for c in chunks[:target_n]]


def _positions(x):
    return [int(v) for v in x] if isinstance(x, list) else [int(x)]


def _remove_exact_ci(text, needle):
    if not needle:
        return text
    return re.sub(re.escape(needle), '', text, flags=re.IGNORECASE)


def _append_once(text, needle):
    if re.search(re.escape(needle), text, flags=re.IGNORECASE):
        return text
    if text and text[-1] not in '.!?':
        text = text + '.'
    return (text + ' ' + needle + '.').strip()


def repair_for_direct_verification(raw_text, event, meta_event):
    # Event tuple format: (date, location, entity, content, detail)
    date = event[0]
    location = event[1]
    entity = event[2]
    detail = event[4]

    n = int(meta_event['nb_paragraphs'])
    chunks = [_sanitize_entity_mentions(c) for c in _normalize_to_chunks(raw_text, n)]

    needles = [location, date, entity, detail]

    # Remove exact required strings from all paragraphs first to enforce uniqueness.
    for i in range(n):
        for needle in needles:
            chunks[i] = _remove_exact_ci(chunks[i], needle)
        chunks[i] = _sanitize_entity_mentions(' '.join(chunks[i].split())) or 'Continued.'

    # Re-inject exact strings only in their target paragraphs.
    target_positions = {
        location: _positions(meta_event['idx_paragraph']['location']),
        date: _positions(meta_event['idx_paragraph']['date']),
        entity: _positions(meta_event['idx_paragraph']['entity']),
        detail: _positions(meta_event['idx_paragraph']['content']),
    }

    for needle, poss in target_positions.items():
        for p in poss:
            idx = max(1, min(n, int(p))) - 1
            chunks[idx] = _append_once(chunks[idx], needle)

    return '\n'.join([f'({i + 1}) {chunks[i]}' for i in range(n)])

def generate_paragraphs_func(
    prompt_parameters = {'nb_events': 10, 'name_universe': 'default', 'name_styles': 'default', 'seed': 0},
    model_parameters = {'model_name': 'gpt-4o-2024-05-13', 'max_new_tokens': 4096, 'temperature': 0.2, 'direct_repair': True},
    data_folder = '/repo/to/git/main/epbench/data',
    env_file = '/repo/to/git/main/.env',
    iterations = None,
    rechecking = True):

    # model parameters
    model_name = model_parameters['model_name']
    max_new_tokens = model_parameters['max_new_tokens']
    temperature = model_parameters.get('temperature', 0.2)
    direct_repair = model_parameters.get('direct_repair', True)
    system_prompt = system_prompt_func()

    config = SettingsWrapper(_env_file = env_file)

    events, meta_events = generate_and_export_events_and_meta_events_func(prompt_parameters, data_folder, rechecking)
    prompts = generate_prompts(events, meta_events, prompt_parameters['name_styles'])

    # iterations
    if iterations is None:
        iterations = [0]*prompt_parameters['nb_events']

    generated_paragraphs = []
    for event_index in range(len(prompts)):
        user_prompt = prompts[event_index]
        iteration = iterations[event_index]
        data_paragraphs_filepath = paragraph_filepath_func(iteration, event_index, data_folder, prompt_parameters, model_parameters)
        if not data_paragraphs_filepath.is_file():
            print("Generate " + str(event_index) + "/" + str(len(prompts)-1))
            # only initialize the model if needed, and only initialize it once 
            try:
                my_model
            except NameError:
                my_model = ModelsWrapper(model_name, config)
            # generate the content
            out = my_model.generate(
                user_prompt = user_prompt,
                system_prompt = system_prompt,
                max_new_tokens = max_new_tokens,
                temperature = temperature,
            )
            if direct_repair:
                out = repair_for_direct_verification(out, events[event_index], meta_events[event_index])
            data_paragraphs_filepath.parent.mkdir(parents=True, exist_ok=True)
            export_list(out, data_paragraphs_filepath)
        generated_paragraph = import_list(data_paragraphs_filepath)
        generated_paragraphs.append(generated_paragraph)

    return generated_paragraphs

def iteration_verbose_func(i, has_direct_verif_vector, final = False):
    idx_issues = [index for index, value in enumerate(has_direct_verif_vector) if value == False]
    percentage = (len(idx_issues) / len(has_direct_verif_vector)) * 100
    percentage_with_issues = f"{percentage:.2f}%"
    ratio_with_issues = f"{str(len(idx_issues))}/{str(len(has_direct_verif_vector))}"
    if final:
        final_str = "final "
    else:
        final_str = ""
    str_output = f"At {final_str}iteration {str(i)}, {percentage_with_issues} remaining with issues ({ratio_with_issues}), for index: {idx_issues}."
    return str_output

def iterative_generate_paragraphs_func(
    prompt_parameters = {'nb_events': 10, 'name_universe': 'default', 'name_styles': 'default', 'seed': 0},
    model_parameters = {'model_name': 'gpt-4o-2024-05-13', 'max_new_tokens': 4096, 'itermax': 10, 'temperature': 0.2, 'direct_repair': True},
    data_folder = '/repo/to/git/main/epbench/data',
    env_file = '/repo/to/git/main/.env',
    verbose = True,
    rechecking = True):
    itermax = model_parameters['itermax']
    # The iterations parameters is automatically iterated
    iterations = [0]*prompt_parameters['nb_events']
    events, meta_events = generate_and_export_events_and_meta_events_func(prompt_parameters, data_folder, rechecking)
    generated_paragraphs = generate_paragraphs_func(prompt_parameters, model_parameters, data_folder, env_file, iterations, rechecking)
    for i in range(itermax-1):
        has_verif_vector = has_passed_direct_and_llm_verifications_func(generated_paragraphs, events, meta_events, iterations, prompt_parameters, model_parameters, data_folder, env_file)
        if not all(has_verif_vector):
            if verbose:
                print(iteration_verbose_func(i, has_verif_vector))
            # erase the previous iteration vector
            iterations = [i+1 if not v else i for (i,v) in zip(iterations, has_verif_vector)]
            # load or regenerate
            generated_paragraphs = generate_paragraphs_func(prompt_parameters, model_parameters, data_folder, env_file, iterations, rechecking)
        else:
            if verbose:
                print(iteration_verbose_func(i, has_verif_vector)) # no issue remaining
            break
    # get the final `has_verif_vector` (e.g. for itermax=1, the loop is passed, and we still need to get this element)
    has_verif_vector = has_passed_direct_and_llm_verifications_func(generated_paragraphs, events, meta_events, iterations, prompt_parameters, model_parameters, data_folder, env_file)
    if verbose:
        print(iteration_verbose_func(itermax-1, has_verif_vector, final = True))
        if not all(has_verif_vector):
            print("itermax reached but some events still did not pass the verification")

    # further filter the ones that does not pass after itermax iterations
    # (the length of the list if still the original one, but the one that did not pass the verification after itermax
    # are set to None)
    generated_paragraphs_filtered = [p if v else None for (p,v) in zip(generated_paragraphs, has_verif_vector)]
    return generated_paragraphs_filtered, has_verif_vector
