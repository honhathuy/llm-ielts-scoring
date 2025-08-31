import re
import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, cohen_kappa_score


BATCH_SIZE = 4

def setup_pipeline():
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "auto"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    return pipe


def process_prompts_in_batches(pipe, prompts, batch_size, description=""):
    all_responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc=description):
        batch_prompts = prompts[i:i + batch_size]
        # Pipeline parameters
        outputs = pipe(
            batch_prompts,
            max_new_tokens=16384,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            return_full_text=False, # Only get the generated part
            batch_size=len(batch_prompts)
        )
        # Extract just the text from the output
        responses = [output[0]['generated_text'] for output in outputs]
        all_responses.extend(responses)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_responses


def calculate_metrics(df):
    eval_df = df.dropna(subset=['band', 'final_band_score'])
    if len(eval_df) == 0:
        return None, None
    y_true = eval_df['band']
    y_pred = eval_df['final_band_score']
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    return rmse, qwk


if __name__ == "__main__":
    llm_pipeline = setup_pipeline()

    df = pd.read_csv('data/test.csv')
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
    turn_1_chats = []
    for index, row in df.iterrows():
        for trait, description in ielts_traits:
            chat = [
                {"role": "system", "content": f"You are an expert IELTS examiner on writting task 2. Four examiners will be provided with a [Prompt] and an [Essay] written by a examinee in response to the [Prompt]. Each examiner will score the essays based on different dimensions of writing quality. Your specific responsibility is to score the essays in terms of '{trait}'. {description} Focus on the content of the [Essay] and the [Scoring Rubric] to determine the score."},
                {"role": "user", "content": f"[Prompt]\n{row['prompt']}\n(end of [Prompt])\n\n[Essay]\n{row['essay']}\n(end of [Essay])\n\nQ. List the quotations from the [Essay] that are relevant to \"{trait}\" and evaluate whether each quotation is well-written or not (no need to give a score yet)."}
            ]
            turn_1_chats.append(chat)
    
    turn_1_responses = process_prompts_in_batches(llm_pipeline, turn_1_chats, BATCH_SIZE, "Turn 1: Quote Retrieval")

    # --- Turn 2: Scoring ---
    turn_2_chats = []
    for i, original_chat in enumerate(turn_1_chats):
        trait_index = i % len(ielts_traits)
        trait, _ = ielts_traits[trait_index]
        llm_response_t1 = turn_1_responses[i]
        
        # Build the conversation history
        new_chat = original_chat + [{"role": "assistant", "content": llm_response_t1}]
        new_chat.append({
            "role": "user", 
            "content": f"[Scoring Rubric]\n**{trait}**:\n{trait_rubrics[trait]}\n(end of [Scoring Rubric])\n\nQ. Based on the [Scoring Rubric] and the quotations you found, how would you rate the \"{trait}\" of this essay? Assign a score from 1.0 to 9.0, strictly following the [Output Format] below.\n\n[Output Format]\nScore: <score>insert ONLY the numeric score (from 1.0 to 9.0) here</score>\n(End of [Output Format])"
        })
        turn_2_chats.append(new_chat)
    
    turn_2_responses = process_prompts_in_batches(llm_pipeline, turn_2_chats, BATCH_SIZE, "Turn 2: Scoring")
    
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
                except Exception:
                    score = None
            all_trait_scores[i].append(score)

    df['trait_scores'] = all_trait_scores
    df['avg_band_score'] = df['trait_scores'].apply(
        lambda scores: np.mean([s for s in scores if s is not None]) if any(s is not None for s in scores) else None
    )
    df['final_band_score'] = df['avg_band_score'].apply(
        lambda x: round(x * 2) / 2 if pd.notna(x) else None
    )
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