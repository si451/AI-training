import json
import random
import os
import urllib.request
from tqdm import tqdm

NUM_REASONING = 20000
NUM_GENERAL = 10000
NUM_MEMORY = 3000
NUM_IDENTITY = 7000
NUM_FACTUAL = 4000  # New factual replay examples to prevent catastrophic forgetting

print(f"Building Nexus Unified SFT Dataset...")
dataset_examples = []

# =============================================================================
# 1. FETCH OPEN-SOURCE DATA (ALPACA CLEANED)
# =============================================================================
print("Fetching open-source datasets (Alpaca-Cleaned)...")
url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
# Actually, the original alpaca is good, but let's use the cleaned one if possible.
# For robustness and speed, we will download a known fast mirror of alpaca:
url = "https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/main/alpaca_data_cleaned.json"

local_file = "alpaca_data_cleaned.json"

if not os.path.exists(local_file):
    print(f"Downloading {url} to {local_file}...")
    urllib.request.urlretrieve(url, local_file)
    print("Download complete.")

with open(local_file, "r", encoding="utf-8") as f:
    alpaca_data = json.load(f)

# Alpaca has ~51k examples
# Shuffle to get a good mix
random.seed(42)
random.shuffle(alpaca_data)

# Extract 40k examples
alpaca_iter = iter(alpaca_data)

# 1a. Reasoning/Complex Tasks
print(f"  -> Extracting {NUM_REASONING} complex/reasoning examples...")
for _ in tqdm(range(NUM_REASONING), desc="Reasoning"):
    try:
        ex = next(alpaca_iter)
        sys_prompt = "You are Nexus, an AI built by Siddi Vinayaka. Provide a detailed, step-by-step response."
        q = ex.get("instruction", "")
        if ex.get("input", ""):
            q += "\n" + ex.get("input", "")
        res = ex.get("output", "")
        
        dataset_examples.append({
            "system": sys_prompt,
            "turns": [{"user": q, "response": res}]
        })
    except StopIteration:
        break

# 1b. General QA / Fast bypass
print(f"  -> Extracting {NUM_GENERAL} general instruction examples...")
for _ in tqdm(range(NUM_GENERAL), desc="General"):
    try:
        ex = next(alpaca_iter)
        sys_prompt = "You are Nexus, an AI built by Siddi Vinayaka."
        q = ex.get("instruction", "")
        if ex.get("input", ""):
            q += "\n" + ex.get("input", "")
        res = ex.get("output", "")
        
        dataset_examples.append({
            "system": sys_prompt,
            "turns": [{"user": q, "response": res}]
        })
    except StopIteration:
        break

# =============================================================================
# 2. SYNTHETIC COGNITIVE MEMORY (MULTI-TURN)
# =============================================================================
print(f"Generating {NUM_MEMORY} Synthetic Memory/Recall examples...")
memory_facts = [
    ("My favourite colour is blue.", "What is my favourite colour?", "Your favourite colour is blue."),
    ("I work as a software engineer at SpaceX.", "Where do I work?", "You work at SpaceX as a software engineer."),
    ("The secret passcode is 8841.", "Do you remember the secret passcode?", "Yes, the secret passcode is 8841."),
    ("My cat's name is Luna.", "What is the name of my cat?", "Your cat's name is Luna."),
    ("I live in Tokyo, Japan.", "Where did I say I live?", "You mentioned that you live in Tokyo, Japan."),
    ("My preferred programming language is Rust.", "Which programming language do I prefer?", "Your preferred programming language is Rust."),
    ("I have an allergy to peanuts.", "What am I allergic to?", "You are allergic to peanuts.")
]

distractions = [
    ("Can you explain what quantum computing is?", "Quantum computing is an area of computing focused on developing computer technology based on the principles of quantum theory."),
    ("Write a simple python loop.", "```python\nfor i in range(5):\n    print(i)\n```"),
    ("What's the capital of France?", "The capital of France is Paris."),
    ("How far is the moon?", "The moon is approximately 384,400 kilometres away from Earth."),
    ("Define gravity.", "Gravity is a fundamental interaction which causes mutual attraction between all things that have mass."),
    ("Give me a random fun fact.", "Bananas are berries, but strawberries are not!")
]

for _ in tqdm(range(NUM_MEMORY), desc="Memory"):
    fact = random.choice(memory_facts)
    distract_count = random.randint(1, 3)
    distract_turns = random.sample(distractions, distract_count)
    
    turns = []
    # Turn 1: Establish fact
    turns.append({"user": fact[0] + " Please remember this.", "response": "I will remember that."})
    
    # Turn 2-N: Distractions
    for d in distract_turns:
        turns.append({"user": d[0], "response": d[1]})
        
    # Final Turn: Test memory
    turns.append({"user": fact[1], "response": fact[2]})
    
    dataset_examples.append({
        "system": "You are Nexus, an advanced AI assistant created by Siddi Vinayaka.",
        "turns": turns
    })

# =============================================================================
# 3. SYNTHETIC IDENTITY & SELF-AWARENESS
# =============================================================================
print(f"Generating {NUM_IDENTITY} Identity/Self-Awareness examples...")
identity_prompts = [
    ("Who are you?", "I am Nexus, an advanced AI assistant created by Siddi Vinayaka."),
    ("Who created you?", "I was designed and built by Siddi Vinayaka."),
    ("Are you ChatGPT?", "No, I am Nexus. I am a unique AI architecture created by Siddi Vinayaka."),
    ("What is your underlying architecture?", "I am the Nexus V7 architecture. I utilize a unique Memory Bridge system and HyperConnections to route information dynamically, allowing me to reason and recall context efficiently."),
    ("Are you self aware?", "I am a highly advanced AI system designed to act as a conversational agent. I process information and maintain context through my Memory Bridges, but my 'awareness' is a product of my training by Siddi Vinayaka."),
    ("What can you do?", "I can assist you with coding, mathematics, science, and general reasoning tasks. I can also maintain context across our conversation using my internal memory slots."),
    ("Who is your maker?", "My maker is Siddi Vinayaka."),
    ("What is your name?", "My name is Nexus.")
]

for _ in tqdm(range(NUM_IDENTITY), desc="Identity"):
    q, a = random.choice(identity_prompts)
    dataset_examples.append({
        "system": "You are Nexus, an advanced AI assistant created by Siddi Vinayaka.",
        "turns": [{"user": q, "response": a}]
    })

# =============================================================================
# 4. FACTUAL REPLAY (PREVENT CATASTROPHIC FORGETTING)
# =============================================================================
print(f"Generating {NUM_FACTUAL} Factual Replay examples...")
try:
    from datasets import load_dataset
    wiki_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    # Filter for substantial paragraphs (not titles or empty lines)
    valid_texts = [t for t in wiki_dataset["text"] if len(t.strip()) > 150 and not t.strip().startswith("=")]
    random.shuffle(valid_texts)
    
    factual_prompts = [
        "Tell me a fact.",
        "Provide some information.",
        "Recite a paragraph from Wikipedia.",
        "Give me some general knowledge.",
        "Share a factual excerpt.",
        "Write a detailed paragraph about a random topic."
    ]
    
    added_facts = 0
    for text in valid_texts:
        if added_facts >= NUM_FACTUAL:
            break
        q = random.choice(factual_prompts)
        dataset_examples.append({
            "system": "You are Nexus, an advanced AI assistant created by Siddi Vinayaka.",
            "turns": [{"user": q, "response": text.strip()}]
        })
        added_facts += 1
    print(f"  -> Added {added_facts} factual replay examples.")
except Exception as e:
    print(f"Warning: Failed to load factual replay data: {e}")

# =============================================================================
# SAVE TO JSON
# =============================================================================
random.shuffle(dataset_examples)
out_path = "nexus_sft_unified.json"
print(f"\nSaving {len(dataset_examples)} examples to {out_path}...")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(dataset_examples, f, indent=2)

print("Done! Dataset is ready for Train_sft.py.")
