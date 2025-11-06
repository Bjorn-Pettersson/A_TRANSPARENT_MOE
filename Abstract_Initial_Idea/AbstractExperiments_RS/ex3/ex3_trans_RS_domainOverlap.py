# transformer_moe_allpretrain_experiment.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import random
from transformers import GPT2Tokenizer
import seaborn as sns
import csv
import math

# -------------------- Config --------------------
CONFIG = {
    "hidden_size": 256,
    "num_experts": 4,
    "expert_layers": 1,
    "pretrain_epochs": 2,        # LIGHT pretraining
    "pretrain_n_per_domain": 300,
    "moe_train_epochs": 6,
    "moe_n_per_domain": 600,
    "batch_size": 16,
    "lr": 1e-4,
    "seed_list": [42, 43, 44],   # multi-seed run
    "use_pretrained_embeddings": False,  # toggle
    "freeze_pretrained_for": 0,  # freeze pretrained expert weights for first k moe epochs
    "entropy_reg": 0.0,          # set >0 to encourage router entropy (avoid collapse)
    "save_dir": "moe_allpretrain_results"
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

# -------------------- Dataset (same plain-English domains) --------------------
class SpecializedTextDataset(Dataset):
    def __init__(self, tokenizer, max_length=128, dataset_type='mixed', n_per_domain=500):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.labels = []
        n = n_per_domain

        domains = {
            'psychology': self._generate_psychology_samples(n),
            'history': self._generate_history_samples(n),
            'medicine': self._generate_medicine_samples(n),
            'business': self._generate_business_samples(n)
        }

        if dataset_type == 'psychology':
            self.data = domains['psychology']
            self.labels = ['psychology'] * len(domains['psychology'])
        elif dataset_type == 'mixed':
            for domain, samples in domains.items():
                self.data.extend(samples)
                self.labels.extend([domain] * len(samples))
        else:
            raise ValueError("dataset_type must be 'psychology' or 'mixed'")

        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)
        if combined:
            self.data, self.labels = zip(*combined)
        else:
            self.data, self.labels = [], []

    # --- generators (same style as your previous script) ---
    def _generate_psychology_samples(self, n):
        templates = [
            "The study found that individuals high in {trait} were more likely to {behavior}.",
            "According to the questionnaire, symptoms of {disorder} include {symptom1} and {symptom2}.",
            "Therapists often use {therapy} to treat anxiety and depression.",
            "Attachment styles influence how people respond to {relationship_event}.",
            "Cognitive biases like {bias} can distort decision-making in everyday life.",
            "Participants completed a {scale} measuring self-esteem and reported scores between {low} and {high}.",
            "Longitudinal research indicates that childhood {experience} predicts adult {outcome}.",
            "Experimental results support the hypothesis that {variable} mediates the relationship between {a} and {b}."
        ]
        traits = ["neuroticism", "extraversion", "conscientiousness"]
        disorders = ["panic disorder", "major depression", "OCD"]
        therapies = ["CBT", "psychodynamic therapy", "exposure therapy"]
        samples = []
        for _ in range(n):
            t = random.choice(templates)
            s = t.format(
                trait=random.choice(traits),
                behavior=random.choice(["avoid social situations", "seek reassurance", "engage in risky behavior"]),
                disorder=random.choice(disorders),
                symptom1=random.choice(["restlessness", "fatigue", "insomnia"]),
                symptom2=random.choice(["low mood", "loss of interest", "difficulty concentrating"]),
                therapy=random.choice(therapies),
                relationship_event=random.choice(["conflict", "separation", "intimacy"]),
                bias=random.choice(["confirmation bias", "availability heuristic", "self-serving bias"]),
                scale=random.choice(["Rosenberg Self-Esteem Scale", "Beck Anxiety Inventory"]),
                low=random.randint(0, 10),
                high=random.randint(11, 30),
                experience=random.choice(["neglect", "attachment disruption", "trauma"]),
                outcome=random.choice(["emotion regulation problems", "relationship difficulties"]),
                variable=random.choice(["rumination", "social support"]),
                a="stress", b="health"
            )
            samples.append(s)
        return samples

    def _generate_history_samples(self, n):
        templates = [
            "In {year}, the {event} led to major political changes in {place}.",
            "Historians debate whether {leader}'s policies caused the {consequence}.",
            "Archaeological evidence from {site} suggests trade networks with {region}.",
            "The treaty signed in {year} ended the conflict and established {arrangement}.",
            "Primary sources show that {group} experienced significant economic hardship during {period}."
        ]
        leaders = ["King Louis", "President Adams", "Emperor Qin"]
        places = ["Europe", "the Mediterranean", "East Asia"]
        samples = []
        for _ in range(n):
            s = random.choice(templates).format(
                year=random.randint(1500, 1950),
                event=random.choice(["a revolution", "a civil war", "an economic crisis"]),
                place=random.choice(places),
                leader=random.choice(leaders),
                consequence=random.choice(["reform", "collapse", "migration"]),
                site=random.choice(["ancient ruins", "burial ground", "old port"]),
                region=random.choice(["North Africa", "South Asia"]),
                arrangement=random.choice(["border adjustments", "trade concessions"]),
                group=random.choice(["peasants", "urban workers"]),
                period=random.choice(["the famine years", "the interwar period"])
            )
            samples.append(s)
        return samples

    def _generate_medicine_samples(self, n):
        templates = [
            "The patient presented with {symptom} and a history of {condition}.",
            "Clinical guidelines recommend {treatment} for patients with {disease}.",
            "Lab tests showed elevated {marker} indicating possible {problem}.",
            "A randomized trial compared {drug_a} versus {drug_b} in reducing {outcome}.",
            "Preventive measures include vaccination, hygiene, and early screening for {disease}."
        ]
        symptoms = ["fever", "shortness of breath", "chest pain"]
        conditions = ["hypertension", "diabetes", "COPD"]
        treatments = ["antibiotics", "ACE inhibitors", "insulin therapy"]
        samples = []
        for _ in range(n):
            s = random.choice(templates).format(
                symptom=random.choice(symptoms),
                condition=random.choice(conditions),
                treatment=random.choice(treatments),
                disease=random.choice(["influenza", "diabetes", "coronary artery disease"]),
                marker=random.choice(["CRP", "troponin", "WBC count"]),
                problem=random.choice(["infection", "myocardial injury"]),
                drug_a=random.choice(["DrugA", "DrugB"]),
                drug_b=random.choice(["Placebo", "DrugC"]),
                outcome=random.choice(["mortality", "symptom severity"])
            )
            samples.append(s)
        return samples

    def _generate_business_samples(self, n):
        templates = [
            "The firm increased quarterly revenue by {pct}% after launching {initiative}.",
            "Market analysis shows competitor {competitor} gaining share in {segment}.",
            "Management adopted {strategy} to reduce costs and improve {metric}.",
            "Investor reports emphasize EBITDA, cash flow, and growth potential in {sector}.",
            "Supply chain disruptions impacted inventory levels and delivery times."
        ]
        initiatives = ["a freemium model", "a new subscription tier", "international expansion"]
        competitors = ["CompetitorX", "CompetitorY"]
        sectors = ["technology", "healthcare", "consumer goods"]
        samples = []
        for _ in range(n):
            s = random.choice(templates).format(
                pct=random.randint(1, 50),
                initiative=random.choice(initiatives),
                competitor=random.choice(competitors),
                segment=random.choice(["SMB", "enterprise", "direct-to-consumer"]),
                strategy=random.choice(["lean manufacturing", "outsourcing"]),
                metric=random.choice(["margins", "customer retention"]),
                sector=random.choice(sectors)
            )
            samples.append(s)
        return samples

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'domain': label
        }

# -------------------- Model --------------------
class TransformerExpert(nn.Module):
    def __init__(self, hidden_size=CONFIG["hidden_size"], num_layers=CONFIG["expert_layers"]):
        super().__init__()
        self.hidden_size = hidden_size
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size*2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        out = self.transformer(hidden_states, src_key_padding_mask=src_key_padding_mask)
        return out

class Router(nn.Module):
    def __init__(self, hidden_size=CONFIG["hidden_size"], num_experts=CONFIG["num_experts"]):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, num_experts)
        )

    def forward(self, hidden_states):
        logits = self.gate(hidden_states)
        weights = F.softmax(logits, dim=-1)
        return weights

class TransformerMoE(nn.Module):
    def __init__(self, vocab_size, hidden_size=CONFIG["hidden_size"], num_experts=CONFIG["num_experts"], num_layers=CONFIG["expert_layers"]):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.router = Router(hidden_size, num_experts)
        self.experts = nn.ModuleList([TransformerExpert(hidden_size, num_layers) for _ in range(num_experts)])
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_experts = num_experts

    def forward(self, input_ids, attention_mask=None, return_routing=False):
        hidden = self.embeddings(input_ids)  # [B, L, H]
        routing_weights = self.router(hidden)  # [B, L, E]

        expert_outputs = []
        for e in range(self.num_experts):
            out = self.experts[e](hidden, attention_mask)
            expert_outputs.append(out)
        expert_outputs = torch.stack(expert_outputs, dim=0)  # [E, B, L, H]

        r = routing_weights.permute(2, 0, 1).unsqueeze(-1)  # [E, B, L, 1]
        mixed = (expert_outputs * r).sum(dim=0)  # [B, L, H]

        logits = self.lm_head(mixed)
        if return_routing:
            return logits, routing_weights
        return logits

    def load_pretrained_expert(self, expert_idx, state_dict):
        self.experts[expert_idx].load_state_dict(state_dict)
        print(f"Loaded pretrained expert into slot {expert_idx}")

# -------------------- Pretrain each expert lightly --------------------
def pretrain_all_experts_light(tokenizer, vocab_size, device, seed=42):
    """Pretrain each expert lightly on its own domain and return a list of expert state_dicts and embeddings (per-expert)."""
    print("\n== PHASE 1: Lightly pretrain ALL experts ==")
    domain_order = ['psychology', 'history', 'medicine', 'business']
    pre_states = []
    pre_embeddings = []
    for i, domain in enumerate(domain_order):
        print(f"\nPretraining expert {i} on domain '{domain}' (light)...")
        ds = SpecializedTextDataset(tokenizer, dataset_type=domain if domain=='psychology' else 'mixed', n_per_domain=0)  # placeholder
        # We call dataset generator per domain using the class internals:
        # Quick hack: create small domain dataset by directly calling generator functions
        # (to avoid duplicating code, instantiate a dataset and then replace data)
        tmp = SpecializedTextDataset(tokenizer, dataset_type='mixed', n_per_domain=0)
        if domain == 'psychology':
            samples = tmp._generate_psychology_samples(CONFIG["pretrain_n_per_domain"])
            labels = ['psychology'] * len(samples)
        elif domain == 'history':
            samples = tmp._generate_history_samples(CONFIG["pretrain_n_per_domain"])
            labels = ['history'] * len(samples)
        elif domain == 'medicine':
            samples = tmp._generate_medicine_samples(CONFIG["pretrain_n_per_domain"])
            labels = ['medicine'] * len(samples)
        elif domain == 'business':
            samples = tmp._generate_business_samples(CONFIG["pretrain_n_per_domain"])
            labels = ['business'] * len(samples)
        else:
            raise ValueError("Unknown domain")
        # Build a small dataset object wrapper
        class SmallTextDS(Dataset):
            def __init__(self, samples, labels, tokenizer):
                self.samples = samples
                self.labels = labels
                self.tokenizer = tokenizer
            def __len__(self): return len(self.samples)
            def __getitem__(self, idx):
                encoded = self.tokenizer(self.samples[idx], max_length=128, padding='max_length', truncation=True, return_tensors='pt')
                return {'input_ids': encoded['input_ids'].squeeze(0), 'attention_mask': encoded['attention_mask'].squeeze(0), 'domain': self.labels[idx]}
        dataset = SmallTextDS(samples, labels, tokenizer)
        loader = DataLoader(dataset, batch_size=CONFIG["batch_size"], shuffle=True)

        expert = TransformerExpert(CONFIG["hidden_size"], CONFIG["expert_layers"]).to(device)
        emb = nn.Embedding(vocab_size, CONFIG["hidden_size"]).to(device)
        lm_head = nn.Linear(CONFIG["hidden_size"], vocab_size).to(device)

        opt = optim.Adam(list(expert.parameters()) + list(emb.parameters()) + list(lm_head.parameters()), lr=CONFIG["lr"])
        criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

        for epoch in range(CONFIG["pretrain_epochs"]):
            expert.train()
            total = 0.0
            for batch in loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                hidden = emb(input_ids)
                out = expert(hidden, attention_mask)
                logits = lm_head(out)
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                loss = criterion(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
            print(f"  Epoch {epoch+1}/{CONFIG['pretrain_epochs']} avg loss: {total/len(loader):.4f}")

        pre_states.append(expert.state_dict())
        pre_embeddings.append(emb.state_dict())
    print("\nAll experts lightly pretrained.")
    return pre_states, pre_embeddings

# -------------------- Train MoE --------------------
def train_moe_allpretrained(pre_states, pre_embeddings, tokenizer, vocab_size, device, seed=42):
    print("\n== PHASE 2: Initialize MoE with pretrained experts and train on mixed data ==")
    mixed = SpecializedTextDataset(tokenizer, dataset_type='mixed', n_per_domain=CONFIG["moe_n_per_domain"])
    loader = DataLoader(mixed, batch_size=CONFIG["batch_size"], shuffle=True)

    moe = TransformerMoE(vocab_size, CONFIG["hidden_size"], CONFIG["num_experts"], CONFIG["expert_layers"]).to(device)

    # Load expert states to corresponding slots
    for i, st in enumerate(pre_states):
        moe.load_pretrained_expert(i, st)
    # Optionally load embeddings (we average per-expert embedding weights if requested)
    if CONFIG["use_pretrained_embeddings"]:
        # average the pretrained per-expert embedding weights to initialize the global embedding
        avg_weight = None
        for d in pre_embeddings:
            w = d['weight']
            if avg_weight is None:
                avg_weight = w.copy()
            else:
                avg_weight += w
        avg_weight /= len(pre_embeddings)
        moe.embeddings.weight.data.copy_(avg_weight)
        print("Loaded averaged pretrained embeddings into MoE embeddings.")
    else:
        print("MoE embeddings left random (use_pretrained_embeddings=False).")

    opt = optim.Adam(moe.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    # Optionally freeze pretrained experts for initial epochs
    freeze_k = CONFIG["freeze_pretrained_for"]

    for epoch in range(CONFIG["moe_train_epochs"]):
        if epoch < freeze_k:
            for p in moe.experts:
                for param in p.parameters():
                    param.requires_grad = False
            print(f"Epoch {epoch+1}: pretrained experts frozen.")
        elif epoch == freeze_k:
            for p in moe.experts:
                for param in p.parameters():
                    param.requires_grad = True
            print(f"Epoch {epoch+1}: unfroze pretrained experts; full fine-tuning now.")

        moe.train()
        total_loss = 0.0
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            logits, routing = moe(input_ids, attention_mask, return_routing=True)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss = criterion(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
            # entropy regularizer to discourage router collapse (optional)
            if CONFIG["entropy_reg"] > 0.0:
                entropy = - (routing * (routing + 1e-12).log()).sum(dim=-1).mean()
                loss = loss + CONFIG["entropy_reg"] * (-entropy)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{CONFIG['moe_train_epochs']} avg loss: {total_loss/len(loader):.4f}")

    print("MoE training finished.")
    return moe

# -------------------- Analysis --------------------
def analyze_routing(moe, tokenizer, device):
    print("\n== PHASE 3: Analyze routing (domain <-> expert alignment) ==")
    test_samples = {
        'psychology': [
            "The therapist noted that the client's rumination maintained the depressive episode.",
            "High levels of neuroticism predicted lower relationship satisfaction in this sample.",
            "Cognitive behavioral techniques reduced avoidance behavior after treatment.",
            "Attachment insecurity is associated with increased conflict in relationships.",
            "The study used validated measures of anxiety and depressive symptoms."
        ],
        'history': [
            "In 1914 the assassination triggered a chain of alliances that led to war.",
            "Archaeologists uncovered pottery fragments at the Bronze Age site.",
            "The treaty established new borders after years of conflict.",
            "Primary letters describe the economic hardship of the period.",
            "Historians attribute the revolution to economic grievances and ideas."
        ],
        'medicine': [
            "The patient presented with fever and cough consistent with a viral infection.",
            "Clinical trials showed the vaccine reduced symptomatic cases significantly.",
            "Blood work indicated elevated CRP and leukocytosis.",
            "Guidelines recommend starting antibiotics for suspected bacterial pneumonia.",
            "The randomized controlled trial measured mortality and hospital length of stay."
        ],
        'business': [
            "Quarterly revenue growth accelerated after the company introduced the new product.",
            "Management focused on cost reduction via automation and supply chain optimization.",
            "Market analysis suggests a gap in the premium segment for direct-to-consumer brands.",
            "Investors tracked cash flow and EBITDA when evaluating the acquisition.",
            "The startup pivoted to a subscription model to increase lifetime value."
        ]
    }

    domain_order = ['psychology', 'history', 'medicine', 'business']
    routing_stats = defaultdict(list)
    token_argmax_counts = defaultdict(Counter)

    moe.eval()
    with torch.no_grad():
        for domain, samples in test_samples.items():
            for text in samples:
                enc = tokenizer(text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
                input_ids = enc['input_ids'].to(device)
                attention_mask = enc['attention_mask'].to(device)
                logits, routing = moe(input_ids, attention_mask, return_routing=True)
                seq_len = int(attention_mask.sum().item())
                avg_r = routing[0, :seq_len].mean(dim=0).cpu().numpy()
                routing_stats[domain].append(avg_r)
                argmaxes = routing[0, :seq_len].argmax(dim=-1).cpu().numpy().tolist()
                for a in argmaxes:
                    token_argmax_counts[domain][int(a)] += 1

    # Build usage matrix
    usage = np.zeros((len(domain_order), moe.num_experts))
    for i, d in enumerate(domain_order):
        arr = np.array(routing_stats[d]) if routing_stats[d] else np.zeros((1, moe.num_experts))
        usage[i] = arr.mean(axis=0) * 100

    # Print per-domain expert breakdown
    print("\nAverage expert usage (%) by domain:")
    for i, d in enumerate(domain_order):
        print(f"\n{d.upper()}:")
        for e in range(moe.num_experts):
            print(f"  Expert {e}: {usage[i, e]:.2f}%")

    # Compute "matching expert" statistics:
    # mapping: psych->0, history->1, medicine->2, business->3
    matches = []
    for i, d in enumerate(domain_order):
        counts = token_argmax_counts[d]
        total = sum(counts.values()) if counts else 0
        match_count = counts.get(i, 0)
        frac = match_count / total if total > 0 else 0.0
        matches.append((d, i, match_count, total, frac))
        print(f"\nDomain '{d}': expert {i} selected for {match_count}/{total} tokens = {frac*100:.1f}%")

    # Save CSV summary
    csv_path = os.path.join(CONFIG["save_dir"], f"routing_summary_seed.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "matching_expert", "match_count", "total_tokens", "fraction"])
        for d, idx, mcount, tot, frac in matches:
            writer.writerow([d, idx, mcount, tot, frac])

    # Heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(usage, annot=True, fmt='.1f', cmap='YlGnBu',
                xticklabels=[f'E{e}' for e in range(moe.num_experts)],
                yticklabels=[d.capitalize() for d in domain_order])
    plt.title('Average Expert Usage (%) by Domain')
    plt.xlabel('Expert')
    plt.ylabel('Domain')
    plt.tight_layout()
    imgpath = os.path.join(CONFIG["save_dir"], "usage_heatmap.png")
    plt.savefig(imgpath, dpi=150, bbox_inches='tight')
    plt.close()

    return usage, token_argmax_counts, matches

# -------------------- Multi-seed runner --------------------
def run_single_seed(seed):
    print(f"\n\n=== Running seed {seed} ===")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    pre_states, pre_embeds = pretrain_all_experts_light(tokenizer, vocab_size, device, seed)
    moe = train_moe_allpretrained(pre_states, pre_embeds, tokenizer, vocab_size, device, seed)
    usage, token_argmax_counts, matches = analyze_routing(moe, tokenizer, device)

    # Return metric: fraction matched for psychology, and vector of fractions for all domains
    fractions = {m[0]: m[4] for m in matches}
    return fractions, usage

def run_multi_seed():
    all_results = []
    usages = []
    for seed in CONFIG["seed_list"]:
        fractions, usage = run_single_seed(seed)
        all_results.append(fractions)
        usages.append(usage)
    # Aggregate
    domains = ['psychology', 'history', 'medicine', 'business']
    summary = {}
    for d in domains:
        vals = [res[d] for res in all_results]
        mean = np.mean(vals)
        std = np.std(vals, ddof=0)
        summary[d] = {"mean": mean, "std": std, "values": vals}
        print(f"\nDomain {d}: matched-expert fraction mean={mean*100:.1f}%, std={std*100:.1f}% over {len(vals)} seeds")

    # Save CSV
    csv_out = os.path.join(CONFIG["save_dir"], "multi_seed_summary.csv")
    with open(csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["domain", "mean_frac", "std_frac"] + [f"seed_{s}" for s in CONFIG["seed_list"]])
        for d in domains:
            row = [d, summary[d]["mean"], summary[d]["std"]] + summary[d]["values"]
            writer.writerow(row)

    # Save aggregated usage heatmap (mean across seeds)
    mean_usage = np.mean(np.stack(usages, axis=0), axis=0)  # shape: seeds x domains x experts -> mean across seeds
    plt.figure(figsize=(8,6))
    sns.heatmap(mean_usage, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=[f'E{e}' for e in range(CONFIG["num_experts"])],
                yticklabels=[d.capitalize() for d in domains])
    plt.title('Mean Expert Usage (%) by Domain (across seeds)')
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["save_dir"], "mean_usage_heatmap.png"), dpi=150, bbox_inches='tight')
    plt.close()

    return summary

# -------------------- Entry --------------------
if __name__ == "__main__":
    summary = run_multi_seed()
    print("\nAll experiments done. Results saved to:", CONFIG["save_dir"])
    print("Key output: CSV 'multi_seed_summary.csv' and heatmap images.")
