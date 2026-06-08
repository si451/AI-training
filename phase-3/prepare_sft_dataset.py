import modal
import json
import random
import os

# ─────────────────────────────────────────────────────────────────────────────
# MODAL APP CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
app = modal.App("nexus-sft-prep")
volume = modal.Volume.from_name("nexus-v1-ckpts", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").pip_install("datasets>=2.19", "tqdm")

TARGET_SAMPLES = 150_000
OUTPUT_FILE = "/checkpoints/sft_dataset.jsonl"
SYSTEM_PROMPT = "You are Nexus, an advanced AI assistant created by Siddi Vinayaka. Think deeply and use tools when necessary."



def format_chat(system, user, assistant):
    return {
        "system": system,
        "turns": [
            {"user": user, "response": assistant}
        ]
    }

def extract_glaive(x):
    chat_str = x.get("chat", "")
    import re
    pattern = re.compile(r"(USER:|ASSISTANT:|FUNCTION CALL:|FUNCTION RESPONSE:)")
    parts = pattern.split(chat_str)
    
    turns = []
    current_user = ""
    current_response = ""
    
    current_role = None
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if part in ["USER:", "ASSISTANT:", "FUNCTION CALL:", "FUNCTION RESPONSE:"]:
            current_role = part
        else:
            if current_role == "USER:":
                if current_response:
                    turns.append({"user": current_user.strip(), "response": current_response.strip()})
                    current_user = part
                    current_response = ""
                else:
                    current_user += "\n" + part if current_user else part
            else:
                prefix = ""
                if current_role == "FUNCTION CALL:":
                    prefix = "\n<function_call>\n"
                elif current_role == "FUNCTION RESPONSE:":
                    prefix = "\n<function_response>\n"
                
                content = prefix + part
                current_response += "\n" + content if current_response else content
                
    if current_user or current_response:
        turns.append({"user": current_user.strip(), "response": current_response.strip()})
            
    return {"source": "glaive", "system": x.get("system", ""), "turns": turns}

def extract_ultrachat(x):
    turns = []
    current_user = ""
    current_response = ""
    for msg in x.get("messages", []):
        if msg["role"] == "user":
            if current_response:
                turns.append({"user": current_user.strip(), "response": current_response.strip()})
                current_response = ""
                current_user = msg["content"]
            else:
                current_user = msg["content"]
        else:
            current_response += "\n" + msg["content"] if current_response else msg["content"]
    if current_user or current_response:
        turns.append({"user": current_user.strip(), "response": current_response.strip()})
    return {"source": "ultrachat", "system": SYSTEM_PROMPT, "turns": turns}

@app.function(
    image=image,
    volumes={"/checkpoints": volume},
    timeout=3600,  # 1 hour timeout for downloading and processing
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def prepare_dataset():
    from datasets import load_dataset
    from tqdm import tqdm

    sft_data = []
    
    print("1. Generating Custom Identity & Memory Data...")
    
    identity_qa = [
        ("Who are you?", "I am Nexus, an advanced AI assistant created by Siddi Vinayaka. I specialise in mathematics, physics, chemistry, programming, and logical reasoning."),
        ("Who is your creator?", "I was created entirely from scratch by Siddi Vinayaka, an independent AI researcher and developer."),
        ("What makes your architecture special?", "I use the custom Nexus architecture designed by Siddi Vinayaka. It features Grouped Query Attention, SwiGLU Mixture-of-Experts, and a Global Neural Memory system with 512 slots split across Lexical, Semantic, and Reasoning bridges."),
        ("How do you remember things?", "I use a multi-level memory architecture that helps me maintain lexical, semantic, and reasoning information across my computations."),
        ("What is your parameter count?", "I have approximately 521 million parameters, but my dynamic routing and memory bridges allow me to perform far above my weight class."),
        ("How does your memory interact with your thought process?", "I use a multi-level memory architecture that helps me maintain lexical, semantic, and reasoning information across my computations."),
    ]
    
    # 50 reps * 6 questions = 300 samples
    for _ in range(50):
        for q, a in identity_qa:
            entry = format_chat(SYSTEM_PROMPT, q, a)
            entry["source"] = "identity_memory"
            sft_data.append(entry)
            
    # Dictionary of HuggingFace datasets to stream and sample
    datasets_to_sample = [
        # Math & Reasoning
        {"name": "GSM8K", "path": "openai/gsm8k", "split": "train", "qty": 7000, "extract": lambda x: format_chat(SYSTEM_PROMPT, x["question"], "Let's think step by step.\n" + x["answer"])},
        # Pure QA / Direct Answering (Alpaca format)
        {"name": "Alpaca QA", "path": "yahma/alpaca-cleaned", "split": "train", "qty": 10000, "extract": lambda x: format_chat(SYSTEM_PROMPT, x["instruction"] + ("\n" + x["input"] if x["input"] else ""), x["output"])},
        # General Conversation / Writing
        {"name": "UltraChat", "path": "HuggingFaceH4/ultrachat_200k", "split": "train_sft", "qty": 30000, "extract": extract_ultrachat},
        # Coding
        {"name": "Code Alpaca", "path": "iamtarun/python_code_instructions_18k_alpaca", "split": "train", "qty": 15000, "extract": lambda x: format_chat(SYSTEM_PROMPT, x["instruction"] + "\n" + x["input"], x["output"])},
        # General Alignment (Latent capability unlocking)
        {"name": "OpenOrca", "path": "Open-Orca/OpenOrca", "split": "train", "qty": 40000, "extract": lambda x: format_chat(x["system_prompt"] if x["system_prompt"] else SYSTEM_PROMPT, x["question"], x["response"])},
        # Extensive Tool Calling
        {"name": "Glaive Tool Calling", "path": "glaiveai/glaive-function-calling-v2", "split": "train", "qty": 20000, "extract": extract_glaive}
    ]

    for ds_info in datasets_to_sample:
        print(f"3. Fetching {ds_info['qty']} samples from {ds_info['name']}...")
        try:
            ds = load_dataset(
                ds_info["path"], 
                ds_info.get("name_kw", "default") if ds_info["path"] != "openai/gsm8k" else "main", 
                split=ds_info["split"], 
                streaming=True,
                token=os.environ.get("HF_TOKEN")
            )
            ds = ds.shuffle(buffer_size=10000, seed=42)  # type: ignore
            
            count = 0
            for item in ds:
                if count >= ds_info["qty"]:
                    break
                try:
                    formatted = ds_info["extract"](item)
                    if "source" not in formatted:
                        formatted["source"] = ds_info["name"]
                    sft_data.append(formatted)
                    count += 1
                except Exception:
                    continue
            print(f"   [OK] Fetched {count} samples.")
        except Exception as e:
            print(f"   [FAIL] Failed to load {ds_info['name']}: {e}")

    # DeepSeek-R1 Distilled / CoT Reasoning traces
    print("4. Fetching DeepSeek-R1 / Long Reasoning Traces...")
    try:
        r1_ds = load_dataset("QuixiAI/dolphin-r1", split="train", streaming=True, token=os.environ.get("HF_TOKEN"))
        count = 0
        for item in r1_ds:
            if count >= 20000:
                break
            
            turns = []
            sys_prompt = SYSTEM_PROMPT
            msgs = item.get("messages", [])
            if msgs and msgs[0]["role"] == "system":
                sys_prompt = msgs[0]["content"]
                msgs = msgs[1:]
                
            current_user = ""
            current_response = ""
            for msg in msgs:
                if msg["role"] == "user":
                    if current_response:
                        turns.append({"user": current_user.strip(), "response": current_response.strip()})
                        current_response = ""
                        current_user = msg["content"]
                    else:
                        current_user = msg["content"]
                else:
                    current_response += "\n" + msg["content"] if current_response else msg["content"]
            if current_user or current_response:
                turns.append({"user": current_user.strip(), "response": current_response.strip()})
                
            entry = {"source": "reasoning_trace", "system": sys_prompt, "turns": turns}
            sft_data.append(entry)
            count += 1
        print(f"   [OK] Fetched {count} reasoning samples.")
    except Exception as e:
        print(f"   [WARN] Failed to load QuixiAI/dolphin-r1: {e}, falling back to NuminaMath-CoT...")
        try:
            cot_ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True, token=os.environ.get("HF_TOKEN"))
            count = 0
            for item in cot_ds:
                if count >= 20000:
                    break
                # Create a pseudo-reasoning trace with <thought> tags
                entry = format_chat(SYSTEM_PROMPT, item["problem"], f"<thought>\n{item['solution']}\n</thought>\n\nThe final answer is derived from the steps above.")
                entry["source"] = "math_reasoning_trace"
                sft_data.append(entry)
                count += 1
            print(f"   [OK] Fetched {count} alternative reasoning samples.")
        except Exception as e2:
            print(f"   [FAIL] Fallback failed: {e2}")

    print(f"\nShuffling {len(sft_data)} total samples...")
    random.shuffle(sft_data)
    
    print(f"Saving to {OUTPUT_FILE} on Modal Volume...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in tqdm(sft_data):
            f.write(json.dumps(entry) + "\n")
            
    print("*** SFT Dataset Preparation Complete! ***")
    
    # Optional: read back size
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"File size on volume: {file_size_mb:.2f} MB")

@app.local_entrypoint()
def main():
    prepare_dataset.remote()
