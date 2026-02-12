from datetime import timedelta
import random
from pathlib import Path
import numpy as np
import pandas as pd
from epbench.src.generation.raw_materials import parameters_universe_func, parameters_styles_func
from epbench.src.io.io import import_list, export_list, data_folder_experiment_func

# ===========================
# Main generation function
# ===========================
def generate_and_export_events_and_meta_events_func(prompt_parameters, data_folder, rechecking=True):
    data_folder_experiment = data_folder_experiment_func(prompt_parameters)
    events = import_list(Path(data_folder) / data_folder_experiment / 'events.json')
    meta_events = import_list(Path(data_folder) / data_folder_experiment / 'meta_events.json')

    # Validate uniqueness
    r = pd.DataFrame(events, columns=list('tsecd'))
    counts_te = r[['t', 'e']].value_counts()
    counts_ts = r[['t', 's']].value_counts()
    if any(counts_te > 1):
        print("Warning: duplicate (t,e) found")
    if any(counts_ts > 1):
        print("Warning: duplicate (t,s) found")

    return events, meta_events

# ===========================
# Export
# ===========================
def export_events_func(events, data_folder, prompt_parameters, outfile='events.json'):
    data_folder_experiment = data_folder_experiment_func(prompt_parameters)
    events_filepath = Path(data_folder) / data_folder_experiment / outfile
    events_filepath.parent.mkdir(parents=True, exist_ok=True)
    if events_filepath.is_file():
        events_saved = import_list(events_filepath)
        Nmax = min(len(events_saved), len(events))
        if not events_saved[:Nmax] == events[:Nmax]:
            raise ValueError(f'Saved {outfile} differs from newly generated events.')
        if len(events) > len(events_saved):
            export_list(events, events_filepath)
    else:
        export_list(events, events_filepath)
    return 0

# ===========================
# Generate events & meta-events
# ===========================
def generate_events_and_meta_events_func(prompt_parameters=None):
    if prompt_parameters is None:
        prompt_parameters = {'nb_events': 5, 'seed': 0, 'distribution_events': {'name': 'geometric', 'param': 0.1}, 'name_universe': 'default', 'name_styles': 'default'}
    nb_events = prompt_parameters['nb_events']
    seed_events = prompt_parameters['seed']
    name_universe = prompt_parameters['name_universe']
    name_styles = prompt_parameters['name_styles']

    events = generate_events(nb_events, seed_events, prompt_parameters['distribution_events'], name_universe)
    meta_events = generate_meta_events(nb_events, seed_events, name_styles)
    return events, meta_events

# ===========================
# Event generation
# ===========================
def generate_events(nb_events=10, seed_events=0, distribution_events={'name': 'geometric', 'param': 0.1}, name_universe='default', N_universe=100, seed_universe=0):
    parameters_universe = parameters_universe_func(name_universe)
    return generate_events_given_parameters_universe(nb_events, seed_events, N_universe, seed_universe, parameters_universe, distribution_events)

def generate_events_given_parameters_universe(nb_events, seed_events, N_universe, seed_universe, parameters_universe, distribution_events):
    # Ensure universe is large enough
    if N_universe < nb_events:
        print(f"Warning: N_universe={N_universe} < nb_events={nb_events}. Expanding universe to {nb_events*2}")
        N_universe = nb_events * 2

    temporal, entities, spatial, content, details = generate_universe(N_universe, seed_universe, parameters_universe)

    multiplier = 3 if nb_events <= 200 else (5 if nb_events <= 800 else 3000)
    events_pool = [generate_event(temporal, entities, spatial, content, details, 10000000*seed_events+seed, distribution_events) for seed in range(multiplier*nb_events)]

    # Filter duplicates (t,e)
    unique_events1 = []
    seen1 = set()
    for event in events_pool:
        t, _, e, _, _ = event
        if (t, e) not in seen1:
            unique_events1.append(event)
            seen1.add((t, e))
    while len(unique_events1) < nb_events:
        unique_events1.append(generate_event(temporal, entities, spatial, content, details, 10000000*seed_events+len(unique_events1), distribution_events))

    # Filter duplicates (t,s)
    unique_events2 = []
    seen2 = set()
    for event in unique_events1:
        t, s, _, _, _ = event
        if (t, s) not in seen2:
            unique_events2.append(event)
            seen2.add((t, s))
    while len(unique_events2) < nb_events:
        unique_events2.append(generate_event(temporal, entities, spatial, content, details, 10000000*seed_events+len(unique_events2), distribution_events))

    return unique_events2[:nb_events]

# ===========================
# Universe generation
# ===========================
def generate_universe(N_universe, seed_universe, parameters_universe):
    check_for_duplicates(parameters_universe)
    check_for_amount(N_universe, parameters_universe)
    temporal = generate_temporal(N_universe, parameters_universe['start_date'], parameters_universe['end_date'], seed_universe)
    entities = generate_entities(N_universe, parameters_universe['first_names'], parameters_universe['last_names'], seed_universe)
    spatial = generate_spatial(N_universe, parameters_universe['locations'], seed_universe)
    content = generate_content(N_universe, parameters_universe['contents'], seed_universe)
    details = generate_details(content, parameters_universe['content_details'])
    return temporal, entities, spatial, content, details

def check_for_duplicates(parameters_universe):
    for key in ['first_names','last_names','locations','contents']:
        duplicates = find_duplicates(parameters_universe[key])
        if duplicates:
            print(f"Duplicate {key}: {duplicates}")
            raise ValueError(f"Duplicated {key}")
    if sum([len(find_duplicates(v)) for (_,v) in parameters_universe['content_details'].items()]) != 0:
        raise ValueError('Duplicated content details for at least one content')
    return 0

def find_duplicates(lst):
    seen, duplicates = {}, []
    for x in lst:
        if x in seen:
            if seen[x]==1: duplicates.append(x)
            seen[x] += 1
        else:
            seen[x] = 1
    return duplicates

def check_for_amount(N_universe, parameters_universe):
    for key in ['first_names','last_names','locations','contents']:
        if len(parameters_universe[key]) < N_universe:
            raise ValueError(f'Too few {key}')
    if set(parameters_universe['content_details'].keys()) != set(parameters_universe['contents']):
        raise ValueError('content_details keys mismatch contents')
    lengths = [len(v) for v in parameters_universe['content_details'].values()]
    if len(set(lengths)) > 1:
        raise ValueError('Different number of content_details for contents')
    return 0

# ===========================
# Generate temporal / dates
# ===========================
def generate_temporal(N, start_date, end_date, seed):
    return [d.strftime('%B %d, %Y') for d in generate_dates(start_date, end_date, N, seed)]

def generate_dates(start_date, end_date, num_dates, seed=1):
    random.seed(seed)
    days = (end_date - start_date).days
    random_days = random.sample(range(days), num_dates)
    return [start_date + timedelta(days=d) for d in random_days]

# ===========================
# Entities / spatial / content
# ===========================
def generate_entities(N, first_names, last_names, seed=1):
    random.seed(seed+1)
    entities = set()
    while len(entities) < N:
        entities.add(f"{random.choice(first_names)} {random.choice(last_names)}")
    return list(entities)

def generate_spatial(N, locations, seed=1):
    tmp = locations.copy()
    random.Random(seed).shuffle(tmp)
    if len(tmp) < N: print("Warning: Not enough locations")
    return tmp[:N]

def generate_content(N, contents, seed=1):
    tmp = contents.copy()
    random.Random(seed).shuffle(tmp)
    if len(tmp) < N: print("Warning: Not enough contents")
    return tmp[:N]

def generate_details(content, content_details):
    return {c: content_details[c] for c in content}

# ===========================
# Event generation helper
# ===========================
def idx_candidate_func(p):
    return np.random.geometric(p=p, size=1).tolist()[0]-1

def censored_geometric_choice(p, my_list, max_attempts=20):
    n = len(my_list)
    for _ in range(max_attempts):
        idx = idx_candidate_func(p)
        if idx < n:
            return my_list[idx]
    return random.choice(my_list)

def generate_event(temporal, entities, spatial, content, details, seed, distribution_events={'name':'geometric','param':0.1}):
    if distribution_events['name']=='uniform':
        random.seed(seed)
        t = random.choice(temporal)
        e = random.choice(entities)
        s = random.choice(spatial)
        c = random.choice(content)
        cd = random.choice(details[c])
    elif distribution_events['name']=='geometric':
        p = distribution_events['param']
        np.random.seed(seed)
        t = censored_geometric_choice(p, temporal)
        e = censored_geometric_choice(p, entities)
        s = censored_geometric_choice(p, spatial)
        c = censored_geometric_choice(p, content)
        cd = random.choice(details[c])
    else:
        raise ValueError("Unknown distribution")
    return [t, s, e, c, cd]

# ===========================
# Meta-events
# ===========================
def generate_meta_events(nb_events=2000, seed_events=0, name_styles='default'):
    parameters_styles = parameters_styles_func(name_styles)
    nb_paragraphs = parameters_styles['nb_paragraphs']
    styles = parameters_styles['styles']

    random.seed(20*seed_events+0)
    events_nb_paragraphs = [random.choice(nb_paragraphs) for _ in range(nb_events)]
    events_idx_location = [random.randint(1, x) for x in events_nb_paragraphs]
    events_idx_date = [random.randint(1, x) for x in events_nb_paragraphs]
    events_idx_entity = [random.randint(1, x) for x in events_nb_paragraphs]
    events_idx_content = [random.randint(1, x) for x in events_nb_paragraphs]
    events_style = [random.choice(styles) for _ in range(nb_events)]

    events_idx_paragraph = [{'location': s, 'date': t, 'entity': e, 'content': c} for (s,t,e,c) in zip(events_idx_location, events_idx_date, events_idx_entity, events_idx_content)]
    return [{'nb_paragraphs': n1, 'idx_paragraph': n2, 'style': n3} for (n1,n2,n3) in zip(events_nb_paragraphs, events_idx_paragraph, events_style)]

# ===========================
# Unused universe
# ===========================
def unused_universe_func(prompt_parameters, events, N_universe=100, seed_universe=0):
    r = pd.DataFrame(events, columns=list('tsecd'))
    used_t = set(r['t'])
    used_s = set(r['s'])
    used_e = set(r['e'])
    used_c = set(r['c'])
    temporal, entities, spatial, content, details = generate_universe(N_universe, seed_universe, parameters_universe_func(prompt_parameters['name_universe']))
    
    unused_t = [x for x in temporal if x not in used_t] or temporal
    unused_s = [x for x in spatial if x not in used_s] or spatial
    unused_e = [x for x in entities if x not in used_e] or entities
    unused_c = [x for x in content if x not in used_c] or content
    unused_d = {k: details[k] for k in unused_c}

    return unused_t, unused_s, unused_e, unused_c, unused_d