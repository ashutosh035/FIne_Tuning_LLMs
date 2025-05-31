# 🦙 Ultra‑Efficient Fine‑Tuning of **Llama 3.1 8B** with **Unsloth**

*(A hands‑on guide to turning an off‑the‑shelf LLM into **your** bespoke assistant)*

> **TL;DR** – This repo walks through a Google Colab workflow that fine‑tunes the `Meta‑Llama‑3.1‑8B` model with **QLoRA** adapters using the blazing‑fast **Unsloth** backend. You’ll learn **why** (and **when**) to fine‑tune, how to pick the right hyper‑parameters, and how to ship your model in both Hugging Face and GGUF formats – all on a single GPU.

---

## 🚀 Why Fine‑Tune Instead of Just Prompting?

1. **Domain mastery** – Teach the model proprietary jargon or workflows it has never seen.
2. **Lower latency & cost** – Swap noisy RAG calls for a compact model that “just knows”.
3. **Style control** – Bake in tone, brand voice, or compliance constraints.
4. **Offline / edge** – Run slimmer quants locally without recurring API calls.

*If prompt‑engineering or RAG already nails your use‑case, stick with them. Otherwise, supervised fine‑tuning (SFT) is the next logical step.*

---

## 🔧 The Three Roads to SFT

| Technique          | Trainable Params   | VRAM Needs | Speed | Pros                                        | Cons                                                |
| ------------------ | ------------------ | ---------- | ----- | ------------------------------------------- | --------------------------------------------------- |
| **Full fine‑tune** | 100 %              | 💎💎💎     | 🐢    | Best raw quality                            | Requires multi‑GPU, risk of catastrophic forgetting |
| **LoRA**           | < 1 %              | 💎         | 🚀    | Parameter‑efficient, plug‑and‑play adapters | Slight quality drop if rank is too low              |
| **QLoRA**          | < 1 % (4‑bit base) | 🪙         | 🚀🚀  | Even lower memory; train on consumer GPUs   | \~39 % slower than LoRA                             |

*We’ll use **QLoRA** because it fits an 8 B model into \~5.4 GB, perfect for free Colab environments.*

---

## 🛠️ Project Structure

```
📂 fine‑tune‑llama‑3.1
 ├── notebook.ipynb  # end‑to‑end Colab script
 ├── data/           # instruction dataset (ShareGPT JSONL)
 ├── output/         # trainer checkpoints & logs
 └── README.md       # you’re here
```

---

## ⚡ Quick Start

```bash
# 1. Setup (Colab or local GPU)
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes
```

```python
# 2. Load 4‑bit base model
from unsloth import FastLanguageModel, is_bfloat16_supported
max_seq_len = 2048
model, tok = FastLanguageModel.from_pretrained(
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    max_seq_length=max_seq_len,
    load_in_4bit=True,
    dtype=None,
)
```

```python
# 3. Attach Rank‑Stabilised LoRA adapters (r = 16)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules=["q_proj","k_proj","v_proj","up_proj","down_proj","o_proj","gate_proj"],
    use_rslora=True,
    use_gradient_checkpointing="unsloth",
)
```

```python
# 4. Convert ShareGPT → ChatML and load dataset
from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

tok = get_chat_template(tok,
    mapping={"role":"from","content":"value","user":"human","assistant":"gpt"},
    chat_template="chatml")

def to_chatml(batch):
    txt = [tok.apply_chat_template(m, tokenize=False) for m in batch["conversations"]]
    return {"text": txt}

data = load_dataset("<your-dataset>", split="train[:10%]")  # slice for quick tests
data = data.map(to_chatml, batched=True)
```

```python
# 5. Train
from trl import SFTTrainer
from transformers import TrainingArguments

trainer = SFTTrainer(
    model=model,
    tokenizer=tok,
    train_dataset=data,
    dataset_text_field="text",
    max_seq_length=max_seq_len,
    packing=True,
    args=TrainingArguments(
        output_dir="output",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        num_train_epochs=1,
        learning_rate=3e-4,
        lr_scheduler_type="linear",
        warmup_steps=10,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        logging_steps=1,
        seed=42,
    ),
)
trainer.train()
```

```python
# 6. Sanity‑check
model = FastLanguageModel.for_inference(model)
msg = [{"from":"human","value":"Is 9.11 larger than 9.9?"}]
ids = tok.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
print(tok.decode(model.generate(ids, max_new_tokens=16)[0]))
```

```python
# 7. Save & merge adapters (16‑bit) then push to Hub
model.save_pretrained_merged("model", tok, save_method="merged_16bit")
model.push_to_hub_merged("<your‑hf‑org>/FineLlama‑3.1‑8B", tok, save_method="merged_16bit")
```

```python
# 8. Optional: one‑liner GGUF export for local inference
for q in ["q4_k_m", "q5_k_m", "q8_0"]:
    model.push_to_hub_gguf("<your‑hf‑org>/FineLlama‑3.1‑8B‑GGUF", tok, q)
```

---

## 🧩 Understanding the Knobs

### LoRA‑Specific

| Hyper‑param          | Typical Range         | Effect                                              |
| -------------------- | --------------------- | --------------------------------------------------- |
| **`r` (rank)**       | 8‑64 (this guide: 16) | Larger = captures more task info but uses more VRAM |
| **`lora_alpha`**     | 1×–2× `r`             | Scaling factor for adapter updates                  |
| **`target_modules`** | q/k/v, proj layers    | More targets = higher quality, higher cost          |
| **`use_rslora`**     | True/False            | Stabilises training for large `r` via 1/√r scaling  |

### Trainer Hyper‑params

| Name                      | Why it matters                                                        |
| ------------------------- | --------------------------------------------------------------------- |
| `learning_rate`           | Too high → divergence; too low → sluggish training                    |
| `lr_scheduler_type`       | Linear is safe; cosine sometimes yields better late‑stage convergence |
| `batch_size × grad_accum` | Defines *effective* batch; crank up until you hit VRAM ceiling        |
| `num_train_epochs`        | Stop at the first signs of over‑fitting (eval loss plateau)           |
| `weight_decay`            | 0.01 keeps weights in check; tweak if you notice over‑regularization  |
| `warmup_steps`            | \~0.1‑1 % of total steps smooths the learning‑rate ramp               |

---

## 📊 Evaluating Your New Model

1. **Automated** – Submit to the [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) or run `lm‑eval‑harness` locally.
2. **Human in the loop** – Spin up an inference UI (Ollama, LM Studio) and sanity‑check real prompts.
3. **Task‑specific** – Build regression/classification benchmarks using your own gold‑label data.

---

## 🏗️ Next Steps

* **Alignment** – Run **DPO** / ORPO on preference data to shape style & safety.
* **Quantization** – Try EXL2, GPTQ, AWQ for even lighter deployments.
* **Deployment** – Serve via a Hugging Face Space, Nvidia Triton, or even on‑device with llama.cpp.

---

## 📜 License & Credits

Everything in this repo is released under the Apache 2.0 license. Feel free to fork, extend, and – most importantly – *ship great products*.

> *Replace this block with your personal bio, social links, and a heartfelt shout‑out to the open‑source community that made all of this possible.*

---

### Happy fine‑tuning! 🧑‍🔧🎉
