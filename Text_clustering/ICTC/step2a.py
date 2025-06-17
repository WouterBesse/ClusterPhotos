import os
import json
from tqdm import tqdm
import torch
from transformers import pipeline, AutoTokenizer
from dotenv import load_dotenv

from utils.argument import args
from utils.llm_utils import get_llama_response
import openai

def get_gpt_response_fast(system_prompt, user_prompt, model):
    try:
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in GPT call: {e}")
        return "ERROR"

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("API_KEY")
    model = os.getenv("MODEL")
    openai.api_key = api_key

    # Initialize LLaMA pipeline if requested
    if args.llama:
        llama_model = "meta-llama/Llama-2-7b-chat-hf"
        tokenizer = AutoTokenizer.from_pretrained(llama_model, use_auth_token=True)
        pipe_line = pipeline(
            "text-generation",
            model=llama_model,
            torch_dtype=torch.float16,
            device_map='auto'
        )

    # Load system prompt and inject the filter
    with open(args.step2a_prompt_path, "r") as label_file:
        system_prompt = label_file.read()
    if "[FILTER]" in system_prompt:
        system_prompt = system_prompt.replace("[FILTER]", args.filter)
    else:
        # Optionally warn if placeholder isn't found
        print("Warning: No [FILTER] placeholder found in prompt.")

    # Read user prompts (Step 1 results)
    with open(args.step1_result_path, "r") as answer_file:
        answers = answer_file.readlines()

    results = []
    for i in tqdm(range(len(answers))):
        answer_data = json.loads(answers[i])
        user_prompt = answer_data.get("text", "")
        image_file = answer_data.get("image_file", None)

        # Run the model
        if args.llama:
            response = get_llama_response(system_prompt, user_prompt, pipe_line, tokenizer)
        else:
            response = get_gpt_response_fast(system_prompt, user_prompt, model)

        prefix = f"Image file-{image_file}; " if image_file else ""
        print(prefix + response)
        results.append(prefix + response)

    # Save the results
    with open(args.step2a_result_path, "w") as file:
        file.write("\n".join(results))
