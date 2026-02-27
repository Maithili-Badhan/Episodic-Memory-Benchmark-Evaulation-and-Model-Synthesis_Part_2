from typing import Set, Dict, Any, List
import json
import re
import ast
from epbench.src.models.models_wrapper import ModelsWrapper
from scipy.stats import kendalltau
import numpy as np

def judge_prompt_func(retrieval_type, correct_answer, llm_answer, correct_answer_long = None):
    d = [{x: "score_between_0_and_1"} for x in correct_answer]
    if correct_answer_long is None:
        correct_answer_long = correct_answer
        adding_text=''
    else:
        adding_text=f'- The matching score should be of length 1, only "matching_score": {json.dumps(d)}'

    prompt = f"""
You are an expert judge evaluating the accuracy of an AI-generated answer against a known groundtruth. Questions can probe for different types or aspects, like what actions or events took place, what people were involved, what were the dates, or what were the locations or spaces.


Question type: {retrieval_type}
Groundtruth: {correct_answer_long}
AI-generated answer: {llm_answer}


Your task:
- Identify all unique items in the AI-generated answer that are relevant to the question type. Answer an empty list [] for this field in case of at least one negative information (e.g., when the answer begins by telling there is no information, or cannot answer)
- Determine a matching score between 0 and 1 for each ground truth item. Give 1 if the item has been found in the relevant items of the AI-generated answer, considering synonyms, paraphrases, or close meanings. Give 0.5 if the item could be considered related to any AI-generated item but without being explicitly stated as such. Give 0 if the item missed mentioning a specific AI-generated item.
- Provide a brief explanation of the evaluation
{adding_text}

Provide your evaluation in the following JSON format:
{{
    "identified_items_in_AI_answer": ["AI_answer_item_1", "AI_answer_item_2", ...],
    "matching_score": {json.dumps(d)}
    "explanation": "Brief explanation of your evaluation"
}}
"""
    return prompt

def append_number_if_duplicate(mylist_in):
    # https://stackoverflow.com/questions/30650474
    newlist = []
    mylist = mylist_in.tolist()
    for i, v in enumerate(mylist):
        totalcount = mylist.count(v)
        count = mylist[:i].count(v)
        newlist.append(f"{v} ({str(count + 1)})" if (totalcount > 1) else v)
    return newlist

def f1_score_func(precision, recall):
    # Issue in precision/recall value happen either when both are 0, or when any is None
    if precision == 0 and recall == 0:
        f1_score = 0
    elif precision is not None and recall is not None:
        f1_score = 2 * (precision*recall) / (precision + recall)
    elif precision is None and recall is None:
        # happen only when nb_preds = nb_gt = 0, perfectly not identifying anything
        f1_score = 1
    else:
        # either no predictions but #gt>0, or at least one prediction but #gt = 0
        f1_score = 0
    return f1_score

def evaluate_answer(llm_answer: str, correct_answer: Set[str], retrieval_type: str, my_model: ModelsWrapper, correct_answer_long: str, get_style: str) -> Dict[str, Any]:
    # no policy or universe for the first evaluation /!\ do not add [the policy can be updated afterwards]
    if correct_answer_long is None:
        correct_answer_long = correct_answer

    # Prepare the prompt for the judge LLM
    judge_prompt = judge_prompt_func(retrieval_type, correct_answer, llm_answer, correct_answer_long)
    print("=== Judge prompt ===")
    print(judge_prompt)

    # Get the judge LLM's evaluation
    judge_response = my_model.generate(user_prompt = judge_prompt, system_prompt = "You are an expert in memory tests.", max_new_tokens = 4096)
    print("=== Raw judge response ===")
    print(judge_response)

    # Parse the judge's response robustly
    evaluation = None
    # 1) Try direct json
    try:
        evaluation = json.loads(judge_response)
    except json.JSONDecodeError:
        # 2) Try to extract the first JSON object with regex
        json_match = re.search(r'\{.*\}', judge_response, re.DOTALL)
        if json_match:
            try:
                evaluation = json.loads(json_match.group())
            except json.JSONDecodeError:
                # 3) Try ast.literal_eval as last resort (handles single quotes)
                try:
                    evaluation = ast.literal_eval(json_match.group())
                except Exception as e:
                    print("[WARN] ast.literal_eval failed:", str(e))
                    evaluation = None
        else:
            # 4) Try to repair common issues (unclosed quotes, smart quotes...)
            try:
                repaired = judge_response.replace("“", '"').replace("”", '"').replace("’", "'")
                json_match = re.search(r'\{.*\}', repaired, re.DOTALL)
                if json_match:
                    try:
                        evaluation = json.loads(json_match.group())
                    except Exception:
                        evaluation = ast.literal_eval(json_match.group())
            except Exception as e:
                print("[WARN] attempted simple repairs failed:", str(e))
                evaluation = None

    if evaluation is None:
        # Ensure we fail gracefully with a meaningful structure
        print("[ERROR] Failed to parse judge response into JSON. Returning an empty/default evaluation.")
        evaluation = {
            'identified_items_in_AI_answer': [],
            'matching_score': [],
            'explanation': "Failed to parse judge response."
        }

    # Normalize evaluation keys to expected names (best-effort)
    # some judge outputs might use 'matched_items' or similar. Map common variants:
    if 'identified_items_in_AI_answer' not in evaluation:
        if 'matched_items' in evaluation:
            evaluation['identified_items_in_AI_answer'] = evaluation['matched_items']
        elif 'identified_items' in evaluation:
            evaluation['identified_items_in_AI_answer'] = evaluation['identified_items']
        else:
            evaluation.setdefault('identified_items_in_AI_answer', [])

    if 'matching_score' not in evaluation:
        # map alternative keys if present
        if 'matching_scores' in evaluation:
            evaluation['matching_score'] = evaluation['matching_scores']
        elif 'matching' in evaluation:
            evaluation['matching_score'] = evaluation['matching']
        else:
            evaluation.setdefault('matching_score', [])

    if 'explanation' not in evaluation:
        evaluation.setdefault('explanation', "")

    return generate_metric_original(correct_answer, evaluation) # keep the original policy there

def remove_duplicates(input_list):
    seen = set()
    output = []
    for item in input_list:
        # item is expected as dict with single key; get the key for uniqueness
        try:
            key = next(iter(item))
        except Exception:
            # fallback: use str(item)
            key = str(item)
        if key not in seen:
            seen.add(key)
            output.append(item)
    return output

def generate_metric_original(correct_answer, evaluation):
    """
    Generate the metrics given the LLM part already computed (this one is saved to the disk, for consistency of the written file, do not change)
    The computation of the F1-score with other strategies is computed in another function
    """

    # Calculate metrics
    nb_gt = len(correct_answer)

    # matching_score from the groundtruth point of view
    predictions = evaluation.get('identified_items_in_AI_answer', [])
    nb_preds = len(predictions)

    # in all cases, we are lenient w.r.t. nb_preds
    if (nb_preds > nb_gt) and (nb_gt > 0):
        nb_preds = nb_gt

    # Extract groundtruth items from matching_score (list of dicts expected)
    matching_score_list = evaluation.get('matching_score', [])
    # Ensure it's a list of single-key dicts; if it's a dict, convert
    if isinstance(matching_score_list, dict):
        # convert dict to list of single-key dicts
        matching_score_list = [{k: v} for k, v in matching_score_list.items()]

    try:
        gt_alt = [list(x.keys())[0] for x in matching_score_list]
    except Exception:
        # fallback: if matching_score malformed, use correct_answer as groundtruth ordering
        print("[WARN] matching_score structure unexpected; falling back to correct_answer for groundtruth list.")
        gt_alt = list(correct_answer)

    nb_gt_alt = len(gt_alt)
    if nb_gt != nb_gt_alt:
        # don't crash — warn and proceed using the groundtruth we have
        print(f"[WARN] nb_gt ({nb_gt}) != nb_gt_alt ({nb_gt_alt}). Proceeding with nb_gt_alt for scoring.")
        nb_gt = nb_gt_alt if nb_gt_alt > 0 else nb_gt

    # sum scores: try to robustly convert stored values to floats
    sum_scores = 0.0
    for x in matching_score_list:
        try:
            val = float(list(x.values())[0])
            sum_scores += val
        except Exception:
            # if value not convertible, skip (treat as 0)
            print("[WARN] could not convert matching score value to float for item:", x)
            continue

    precision = sum_scores / nb_preds if nb_preds > 0 else None
    recall = sum_scores / nb_gt if nb_gt > 0 else None
    f1_score = f1_score_func(precision, recall)
    return {'predicted_items': list(predictions),
            'groundtruth_items': list(gt_alt),
            'matching_groundtruth_items_score': matching_score_list,
            'explanation': evaluation.get('explanation', ""),
            'nb_preds': nb_preds,
            'nb_gt': nb_gt,
            'sum_scores': sum_scores,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score}

def generate_metric(correct_answer, evaluation, policy = 'remove_duplicates'):
    """
    Generate the metrics given the LLM part already computed (useful for updated solely the metric computation)
    """
    if policy == 'remove_duplicates':
        evaluation['matching_score'] = remove_duplicates(evaluation.get('matching_score', [])) # remove duplicates in the list of dictionaries
        # remove duplicates while keeping order for correct_answer
        try:
            correct_answer = list(dict.fromkeys(correct_answer))
        except Exception:
            correct_answer = list(correct_answer)

    # Calculate metrics
    nb_gt = len(correct_answer)

    # matching_score from the groundtruth point of view
    predictions = evaluation.get('identified_items_in_AI_answer', [])
    nb_preds_harsh = len(predictions)

    if (nb_preds_harsh > nb_gt) and (nb_gt > 0):
        nb_preds_lenient = nb_gt
    else:
        nb_preds_lenient = nb_preds_harsh

    matching_score_list = evaluation.get('matching_score', [])
    if isinstance(matching_score_list, dict):
        matching_score_list = [{k: v} for k, v in matching_score_list.items()]

    try:
        gt_alt = [list(x.keys())[0] for x in matching_score_list]
    except Exception:
        print("[WARN] matching_score structure unexpected in generate_metric; using correct_answer as fallback.")
        gt_alt = list(correct_answer)

    nb_gt_alt = len(gt_alt)
    if nb_gt != nb_gt_alt:
        print(f"[WARN] nb_gt ({nb_gt}) != nb_gt_alt ({nb_gt_alt}). Proceeding with nb_gt_alt for scoring.")
        nb_gt = nb_gt_alt if nb_gt_alt > 0 else nb_gt

    # common (old and new)
    sum_scores = 0.0
    for x in matching_score_list:
        try:
            val = float(list(x.values())[0])
            sum_scores += val
        except Exception:
            print("[WARN] could not convert matching score value to float for item:", x)
            continue

    precision_lenient = sum_scores / nb_preds_lenient if nb_preds_lenient > 0 else None
    precision_harsh = sum_scores / nb_preds_harsh if nb_preds_harsh > 0 else None

    recall = sum_scores / nb_gt if nb_gt > 0 else None
    f1_score_lenient = f1_score_func(precision_lenient, recall)
    f1_score_harsh = f1_score_func(precision_harsh, recall)
    return {'predicted_items': predictions,
            'groundtruth_items': gt_alt,
            'matching_groundtruth_items_score': matching_score_list,
            'explanation': evaluation.get('explanation', ""),
            'nb_preds_lenient': nb_preds_lenient,
            'nb_preds_harsh': nb_preds_harsh,
            'nb_gt': nb_gt,
            'sum_scores': sum_scores,
            'precision_lenient': precision_lenient,
            'precision_harsh': precision_harsh,
            'recall': recall,
            'f1_score_lenient': f1_score_lenient,
            'f1_score_harsh': f1_score_harsh,
            'diff_f1': (f1_score_lenient - f1_score_harsh) if (f1_score_lenient is not None and f1_score_harsh is not None) else None}

def update_policy_of_evaluation_to(df_generated_evaluations, policy = 'remove_duplicates'):
    # defensive copy
    df_to_update = df_generated_evaluations.copy()

    # Ensure expected columns exist; if not, create them with defaults
    expected_cols_defaults = {
        'predicted_items': [[] for _ in range(len(df_to_update))],
        'matching_groundtruth_items_score': [[] for _ in range(len(df_to_update))],
        'explanation': ["" for _ in range(len(df_to_update))],
        'correct_answer': [[] for _ in range(len(df_to_update))],
        'groundtruth_items': [[] for _ in range(len(df_to_update))],
    }
    for col, default in expected_cols_defaults.items():
        if col not in df_to_update.columns:
            print(f"[WARN] column '{col}' missing in df_generated_evaluations — inserting default values.")
            df_to_update[col] = default

    elements_for_which_f1_should_be_recomputed = [(i, x) for i, x in enumerate(df_to_update['groundtruth_items'])]
    for i, x in elements_for_which_f1_should_be_recomputed:
        current_sample = df_to_update.iloc[i]

        # Defensive extraction of required fields
        predicted_items = current_sample.get('predicted_items', []) if isinstance(current_sample, dict) else current_sample['predicted_items']
        matching_score = current_sample.get('matching_groundtruth_items_score', []) if isinstance(current_sample, dict) else current_sample['matching_groundtruth_items_score']
        explanation = current_sample.get('explanation', "") if isinstance(current_sample, dict) else current_sample['explanation']

        # Normalize matching_score to list of single-key dicts if necessary
        if isinstance(matching_score, dict):
            matching_score = [{k: v} for k, v in matching_score.items()]

        evaluation = {
            'identified_items_in_AI_answer': list(dict.fromkeys(predicted_items)), # remove duplicates while keeping the order
            'matching_score': matching_score,
            'explanation': explanation
        }

        # Get correct_answer fallback
        correct_answer = current_sample.get('correct_answer', x)
        try:
            res = generate_metric(correct_answer, evaluation, policy = policy)
        except Exception as e:
            # If generate_metric fails, fill with safe defaults and continue
            print(f"[ERROR] generate_metric failed at index {i} with error: {e}. Filling defaults.")
            res = {
                'predicted_items': evaluation['identified_items_in_AI_answer'],
                'groundtruth_items': list(correct_answer),
                'matching_groundtruth_items_score': evaluation['matching_score'],
                'explanation': evaluation['explanation'],
                'nb_preds_lenient': len(evaluation['identified_items_in_AI_answer']),
                'nb_preds_harsh': len(evaluation['identified_items_in_AI_answer']),
                'nb_gt': len(correct_answer),
                'sum_scores': 0.0,
                'precision_lenient': None,
                'precision_harsh': None,
                'recall': None,
                'f1_score_lenient': None,
                'f1_score_harsh': None,
                'diff_f1': None
            }

        # Write back updated values
        df_to_update.at[i, 'predicted_items'] = res.get('predicted_items', [])
        df_to_update.at[i, 'groundtruth_items'] = res.get('groundtruth_items', [])
        df_to_update.at[i, 'matching_groundtruth_items_score'] = res.get('matching_groundtruth_items_score', [])
        df_to_update.at[i, 'explanation'] = res.get('explanation', "")
        df_to_update.at[i, 'nb_preds_lenient'] = res.get('nb_preds_lenient', np.nan)
        df_to_update.at[i, 'nb_preds_harsh'] = res.get('nb_preds_harsh', np.nan)
        df_to_update.at[i, 'nb_gt'] = res.get('nb_gt', np.nan)
        df_to_update.at[i, 'sum_scores'] = res.get('sum_scores', np.nan)
        df_to_update.at[i, 'precision_lenient'] = res.get('precision_lenient', np.nan)
        df_to_update.at[i, 'precision_harsh'] = res.get('precision_harsh', np.nan)
        df_to_update.at[i, 'recall'] = res.get('recall', np.nan)
        df_to_update.at[i, 'f1_score_lenient'] = res.get('f1_score_lenient', np.nan)
        df_to_update.at[i, 'f1_score_harsh'] = res.get('f1_score_harsh', np.nan)
        df_to_update.at[i, 'diff_f1'] = res.get('diff_f1', np.nan)

        # Keep legacy fields as NaN for clarity (you can remove them later)
        df_to_update.at[i, 'precision'] = np.nan
        df_to_update.at[i, 'f1_score'] = np.nan

    return df_to_update # updated one


def process_lists_and_compute_kendall_tau(l1, l2):
    # Step 1: Remove duplicates while preserving order
    l1_no_duplicates = list(dict.fromkeys(l1))
    l2_no_duplicates = list(dict.fromkeys(l2))

    # Step 2: Keep only elements that are in both lists
    common_elements = list(set(l1_no_duplicates) & set(l2_no_duplicates))

    # Create new lists with only common elements, preserving original order
    result_l1 = [x for x in l1_no_duplicates if x in common_elements]
    result_l2 = [x for x in l2_no_duplicates if x in common_elements]

    # Step 3: Compute Kendall tau
    # protect from empty common_elements
    if len(common_elements) == 0:
        return result_l1, result_l2, None, None

    tau, p_value = kendalltau(
        [result_l1.index(x) for x in common_elements],
        [result_l2.index(x) for x in common_elements]
    )

    return result_l1, result_l2, tau, p_value

def judge_prompt_chronological_func(groundtruth_items, predicted_items):
    groundtruth_indexes = [x for x in range(len(groundtruth_items))]
    chronological_prompt = f"""You are an expert judge evaluating the alignment between an AI-generated list and a known groundtruth list. Your task is to match items from the predicted list to the groundtruth list, considering their order and uniqueness.

    Given:
    Groundtruth list: {groundtruth_items}
    Groundtruth indexes: {groundtruth_indexes}
    Predicted list: {predicted_items}

    Instructions:
    1. For each item in the predicted list, find the first corresponding index from the groundtruth list that hasn't been used yet.
    2. Assign indexes based on these rules:
    a. If a match is found and the groundtruth index hasn't been used, assign that index.
    b. If no match is found, or if all matching indexes have already been used, assign -1.
    3. Always use the earliest matching index from the groundtruth list, even if there's an exact match later.
    4. Provide a brief explanation of your index assignments.

    Output your evaluation in the following JSON format:
    {{
        "groundtruth_indexes": {groundtruth_indexes},
        "predicted_indexes": [index1, index2, ...],
        "explanation": "Concise explanation of index assignments"
    }}

    Consider these examples:

    Example 1:
    Groundtruth list: ['Ice Preservation Discussions', 'Theater Show', 'Parkour Workshop']
    Predicted list: ['Theater Performance', 'Tech Hackathon', 'Ice Preservation Talks']
    {{
        "groundtruth_indexes": [0, 1, 2],
        "predicted_indexes": [1, -1, 0],
        "explanation": "Theater Performance matches Theater Show (index 1), Tech Hackathon has no match (-1), Ice Preservation Talks matches Ice Preservation Discussions (index 0)."
    }}

    Example 2:
    Groundtruth list: ['Ice Preservation Discussions', 'Theater Show', 'Parkour Workshop', 'Theater Performance']
    Predicted list: ['Theater Performance', 'Tech Hackathon', 'Ice Preservation Talks']
    {{
        "groundtruth_indexes": [0, 1, 2, 3],
        "predicted_indexes": [1, -1, 0],
        "explanation": "Theater Performance matches Theater Show (index 1, first available match), Tech Hackathon has no match (-1), Ice Preservation Talks matches Ice Preservation Discussions (index 0)."
    }}

    Now, please provide your evaluation for the given lists:
    """
    return chronological_prompt

def evaluate_chronological(groundtruth_items: List[str], predicted_items: List[str], my_model: ModelsWrapper) -> Dict[str, Any]:

    # Prepare the prompt for the judge LLM
    judge_prompt = judge_prompt_chronological_func(groundtruth_items, predicted_items)
    print("=== Chronological judge prompt ===")
    print(judge_prompt)
    system_prompt = "You are an expert judge evaluating the alignment between an AI-generated list and a known groundtruth list."

    # Get the judge LLM's evaluation
    judge_response = my_model.generate(user_prompt = judge_prompt, system_prompt = system_prompt, max_new_tokens = 4096)
    print("=== Raw chronological judge response ===")
    print(judge_response)

    # Parse the judge's response robustly
    evaluation = None
    try:
        evaluation = json.loads(judge_response)
    except json.JSONDecodeError:
        json_match = re.search(r'\{.*\}', judge_response, re.DOTALL)
        if json_match:
            try:
                evaluation = json.loads(json_match.group())
            except json.JSONDecodeError:
                try:
                    evaluation = ast.literal_eval(json_match.group())
                except Exception as e:
                    print("[WARN] failed to parse chronological judge response:", str(e))
                    evaluation = None
        else:
            print("[WARN] no JSON object found in chronological judge response.")
            evaluation = None

    if evaluation is None:
        raise ValueError("Failed to parse judge's chronological response into JSON")

    return evaluation