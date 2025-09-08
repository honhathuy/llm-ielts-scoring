from unsloth import FastLanguageModel
import re
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, cohen_kappa_score


# !pip install unsloth
# !pip install --upgrade --no-deps "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_pipeline():
    # model_id = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
    model_id = "unsloth/Meta-Llama-3.1-8B-Instruct"
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        # dtype = torch.float16,
        max_seq_length = 10000,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    
    return model, tokenizer


def process_prompts_in_batches(model, tokenizer, prompts, batch_size, description=""):
    all_responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=description):
        batch_prompts = prompts[i:i + batch_size]
        
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        inputs = tokenizer(batch_prompts, return_tensors = "pt", padding = True).to(DEVICE)
        input_ids_len = inputs['input_ids'].shape[1]
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            use_cache = False,
            # do_sample=True,
            # temperature=0.7,
            # top_p=0.8,
            # top_k=20
        )
        decoded_responses = tokenizer.batch_decode(outputs[:, input_ids_len:], skip_special_tokens=True)
        all_responses.extend(decoded_responses)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_responses


def trait_aggregation_and_scaling(df, avg_score_col, target_range_col):
    """
    Applies trait aggregation and scaling as per Section 2.3 of the paper.
    This involves:
    1. Outlier Clipping: Removes extreme values from the averaged scores.
    2. Min-Max Scaling: Maps the clipped scores to the target score range.
    3. Rounding: Rounds the final score to the nearest half-band (e.g., 6.0, 6.5).
    """
    print("-> Applying Trait Aggregation and Scaling mechanism...")
    
    # Ensure columns exist
    if avg_score_col not in df.columns or target_range_col not in df.columns:
        raise ValueError(f"Required columns '{avg_score_col}' or '{target_range_col}' not in DataFrame.")

    # Drop rows where average score is NaN for calculation
    valid_scores = df[avg_score_col].dropna()
    if valid_scores.empty:
        df['final_band_score'] = np.nan
        return df

    # --- 1. Outlier Clipping ---
    # Calculate Q1, Q3, and IQR to find outliers
    Q1 = valid_scores.quantile(0.25)
    Q3 = valid_scores.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Clip the averaged scores to remove outliers
    df['clipped_avg_score'] = df[avg_score_col].clip(lower=lower_bound, upper=upper_bound)

    # --- 2. Min-Max Scaling ---
    # Determine the target score range from the ground truth 'band' column
    target_min = df[target_range_col].min()
    target_max = df[target_range_col].max()

    # Get min and max of the clipped scores for scaling
    clipped_scores_valid = df['clipped_avg_score'].dropna()
    if clipped_scores_valid.empty:
        df['final_band_score'] = np.nan
        df.drop(columns=['clipped_avg_score'], inplace=True)
        return df

    source_min = clipped_scores_valid.min()
    source_max = clipped_scores_valid.max()

    # Apply min-max scaling formula
    if source_max == source_min:
        # If all scores are the same, assign the average of the target range
        df['final_band_score'] = (target_min + target_max) / 2
    else:
        df['final_band_score'] = target_min + \
            (df['clipped_avg_score'] - source_min) * (target_max - target_min) / (source_max - source_min)

    # --- 3. Final Rounding ---
    # For IELTS, scores are in 0.5 increments. This rounding achieves that.
    # For other datasets like ASAP, you might round to the nearest integer: .round()
    df['final_band_score'] = df['final_band_score'].apply(
        lambda x: round(x * 2) / 2 if pd.notna(x) else None
    )

    # Clean up intermediate column
    df.drop(columns=['clipped_avg_score'], inplace=True)

    return df


def calculate_metrics(df):
    eval_df = df.dropna(subset=['band', 'final_band_score'])
    if len(eval_df) == 0:
        return None, None
    y_true = eval_df['band']
    y_pred = eval_df['final_band_score']
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_true_int = (y_true * 2).astype(int)
    y_pred_int = (y_pred * 2).astype(int)
    qwk = cohen_kappa_score(y_true_int, y_pred_int, weights='quadratic')
    return rmse, qwk


if __name__ == "__main__":
    model, tokenizer = setup_pipeline()

    df = pd.read_csv('data/band_wise_records.csv')
    df = df.head(4)
    df['band'] = df['band'].astype(str).str.replace('>4', '3', regex=False)
    df['band'] = pd.to_numeric(df['band'], errors='coerce')

    ielts_traits = ["Task Response", "Coherence and Cohesion", "Lexical Resource", "Grammatical Range and Accuracy"]
    trait_descriptions = []
    with open("rubric/trait_description.txt", 'r') as f:
        trait_descriptions = f.read().splitlines()
    ielts_traits = list(zip(ielts_traits, trait_descriptions))

    trait_rubrics = {}
    for trait, _ in ielts_traits:
        filename = f"rubric/{trait.lower().replace(' ', '_')}.txt"
        with open(filename, 'r') as f:
            trait_rubrics[trait] = f.read()

    # --- Step 1: Multi-Turn Conversation (Batched) ---
    print(f"\n[Step 1] Running multi-turn conversations for {len(df)} essays...")

    # --- Turn 1: Quote Retrieval ---
    turn_1_chats_list = []
    turn_1_prompts_str = []

    for index, row in df.iterrows():
        for trait, description in ielts_traits:
            messages = [
                {"role": "system", "content": f"You are a member of the English essay writing test evaluation committee. Four teachers will be provided with a [Prompt] and an [Essay] written by a student in response to the [Prompt]. Each teacher will score the essays based on different dimensions of writing quality. Your specific responsibility is to score the essays in terms of '{trait}'. {description} Focus on the content of the [Essay] and the [Scoring Rubric] to determine the score."},
                {"role": "user", "content": f"[Prompt]\n{row['prompt']}\n(end of [Prompt])\n\n[Essay]\n{row['essay']}\n(end of [Essay])\n\nQ. List the quotations from the [Essay] that are relevant to \"{trait}\" and evaluate whether each quotation is well-written or not (no need to give a score yet)."}
            ]
            turn_1_chats_list.append(messages)
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                # enable_thinking = False, # Disable thinking
            )
            turn_1_prompts_str.append(text)

    turn_1_responses = process_prompts_in_batches(model, tokenizer, turn_1_prompts_str, BATCH_SIZE, "Turn 1: Quote Retrieval")

    # --- Turn 2: Scoring ---
    turn_2_prompts_str = []
    for i, original_chat_list in enumerate(turn_1_chats_list):
        trait_index = i % len(ielts_traits)
        trait, _ = ielts_traits[trait_index]
        llm_response_t1 = turn_1_responses[i]
        
        new_chat_list = original_chat_list.copy() 
        new_chat_list.append({"role": "assistant", "content": llm_response_t1})
        new_chat_list.append({
            "role": "user", 
            "content": f"[Scoring Rubric]\n**{trait}**:\n{trait_rubrics[trait]}\n(end of [Scoring Rubric])\n\nQ. Based on the [Scoring Rubric] and the quotations you found, how would you rate the \"{trait}\" of this essay? Assign a score from 0 to 10, strictly following the [Output Format] below.\n\n[Output Format]\n<score>Numeric score (from 0 to 10)</score>\n(End of [Output Format])"
        })
        
        final_prompt_str = tokenizer.apply_chat_template(
            new_chat_list,
            tokenize=False,
            add_generation_prompt=True,
            # enable_thinking = False, # Disable thinking
        )
        turn_2_prompts_str.append(final_prompt_str)

    turn_2_responses = process_prompts_in_batches(model, tokenizer, turn_2_prompts_str, BATCH_SIZE, "Turn 2: Scoring")
    
    # --- Step 2: Parse and Calculate ---
    print("\n[Step 2] Parsing final responses and calculating scores...")
    all_trait_scores = [[] for _ in range(len(df))]
    for i in range(len(df)):
        for j in range(len(ielts_traits)):
            response_index = i * len(ielts_traits) + j
            response = turn_2_responses[response_index]
            score = None
            if response:
                try:
                    score_match = re.search(r'<score>([0-9\.]+)</score>', response)
                    if score_match:
                        score = float(score_match.group(1))
                    else:
                        # If fail, find all numbers in the response and take the last one.
                        # This is a fallback for messy outputs.
                        numeric_values = re.findall(r'[0-9\.]+', response)
                        if numeric_values:
                            score = float(numeric_values[-1])
                except Exception:
                    score = None
            all_trait_scores[i].append(score)

    df['trait_scores'] = all_trait_scores
    df['avg_band_score'] = df['trait_scores'].apply(
        lambda scores: np.mean([s for s in scores if s is not None]) if any(s is not None for s in scores) else None
    )

    df = trait_aggregation_and_scaling(df, 'avg_band_score', 'band')

    df.to_csv('mts_llama_result.csv')
    print("-> Final band scores calculated.")

    # --- Step 3: Evaluate and Display ---
    print("\n[Step 3] Evaluating model performance...")
    rmse, qwk = calculate_metrics(df)

    print("\n" + "="*80)
    print("                    IELTS Essay Scoring Final Report")
    print("="*80)
    display_cols = ['prompt', 'essay', 'band', 'final_band_score']
    print(df[display_cols])
    
    print("\n" + "="*80)
    print("                    Model Performance Metrics")
    print("="*80)
    if rmse is not None and qwk is not None:
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Quadratic Weighted Kappa (QWK): {qwk:.4f}")
    else:
        print("Performance metrics could not be calculated.")
    print("="*80)