import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import json
import random
from transformers import GPT2Tokenizer, GPT2Model
import seaborn as sns

# Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== Dataset Creation ====================
class SpecializedTextDataset(Dataset):
    """Dataset with different text domains for specialization"""
    
    def __init__(self, tokenizer, max_length=128, dataset_type='mixed'):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        self.labels = []  # Domain labels for analysis
        
        # Create domain-specific synthetic data
        domains = {
            'code': self._generate_code_samples(),
            'math': self._generate_math_samples(),
            'story': self._generate_story_samples(),
            'science': self._generate_science_samples()
        }
        
        if dataset_type == 'code':
            # For pre-training expert on code
            self.data = domains['code']
            self.labels = ['code'] * len(domains['code'])
        elif dataset_type == 'mixed':
            # For full training
            for domain, samples in domains.items():
                self.data.extend(samples)
                self.labels.extend([domain] * len(samples))
        
        # Shuffle
        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)
        self.data, self.labels = zip(*combined) if combined else ([], [])
        
    def _generate_code_samples(self, n=500):
        """Generate synthetic code-like text"""
        templates = [
            "def {func}({args}): return {expr}",
            "for i in range({n}): print({var}[i])",
            "class {cls}: def __init__(self): self.{attr} = {val}",
            "if {cond}: {stmt1} else: {stmt2}",
            "import {lib}; result = {lib}.{method}({params})",
            "try: {code} except: {handler}",
            "while {cond}: {body}; {update}",
            "lambda x: x * {n} + {m}",
        ]
        
        samples = []
        for _ in range(n):
            template = random.choice(templates)
            code = template.format(
                func=f"func_{random.randint(1,99)}",
                args="x, y",
                expr=f"x + {random.randint(1,10)}",
                n=random.randint(1, 100),
                var="data",
                cls=f"Class{random.randint(1,99)}",
                attr="value",
                val=random.randint(0, 100),
                cond=f"x > {random.randint(0, 10)}",
                stmt1="return True",
                stmt2="return False",
                lib="numpy",
                method="array",
                params="[1, 2, 3]",
                code="process_data()",
                handler="pass",
                body="count += 1",
                update="i += 1",
                m=random.randint(1, 10)
            )
            samples.append(code)
        return samples
    
    def _generate_math_samples(self, n=500):
        """Generate synthetic math text"""
        templates = [
            "Calculate: {a} + {b} = {result}",
            "If x = {x}, then 2x + 3 = {result}",
            "The derivative of x^{n} is {n}x^{nm1}",
            "Solve for x: {a}x + {b} = {c}",
            "The area of a circle with radius {r} is {area}",
            "The sum from 1 to {n} equals {sum}",
            "{a} multiplied by {b} equals {result}",
            "The square root of {n} is approximately {sqrt}",
        ]
        
        samples = []
        for _ in range(n):
            a, b = random.randint(1, 100), random.randint(1, 100)
            n = random.randint(2, 5)
            template = random.choice(templates)
            math_text = template.format(
                a=a, b=b, result=a+b,
                x=random.randint(1, 10),
                n=n, nm1=n-1,
                c=random.randint(1, 100),
                r=random.randint(1, 10),
                area=f"{3.14 * random.randint(1, 10)**2:.2f}",
                sum=(n*(n+1))//2,
                sqrt=f"{np.sqrt(random.randint(1, 100)):.2f}"
            )
            samples.append(math_text)
        return samples
    
    def _generate_story_samples(self, n=500):
        """Generate synthetic story text"""
        templates = [
            "Once upon a time, {character} went to {place} and found {object}.",
            "The {adj1} {noun1} {verb} the {adj2} {noun2}.",
            "{character} said, 'I will {action} tomorrow at {time}.'",
            "In the {adj} forest, a {creature} was {activity}.",
            "The hero {verb} the dragon and saved the {place}.",
            "It was a {adj} day when {event} happened.",
            "{character1} and {character2} became {relationship}.",
            "The {object} glowed {adverb} in the {time_of_day} light.",
        ]
        
        characters = ["Alice", "Bob", "the wizard", "the knight", "Sarah"]
        places = ["castle", "forest", "village", "mountain", "river"]
        objects = ["sword", "gem", "book", "key", "map"]
        adjectives = ["ancient", "mysterious", "golden", "dark", "bright"]
        
        samples = []
        for _ in range(n):
            template = random.choice(templates)
            story = template.format(
                character=random.choice(characters),
                character1=random.choice(characters),
                character2=random.choice(characters),
                place=random.choice(places),
                object=random.choice(objects),
                adj=random.choice(adjectives),
                adj1=random.choice(adjectives),
                adj2=random.choice(adjectives),
                noun1="dragon",
                noun2="kingdom",
                verb=random.choice(["defeated", "discovered", "protected"]),
                action="travel",
                time="dawn",
                creature="unicorn",
                activity="sleeping",
                event="the prophecy",
                relationship="friends",
                adverb="brightly",
                time_of_day="morning"
            )
            samples.append(story)
        return samples
    
    def _generate_science_samples(self, n=500):
        """Generate synthetic science text"""
        templates = [
            "The {element} atom has {n} protons in its nucleus.",
            "Photosynthesis converts {input} into {output} using sunlight.",
            "The {planet} is the {ordinal} planet from the sun.",
            "In physics, force equals mass times {term}.",
            "The {process} cycle is essential for {purpose}.",
            "DNA consists of {n} base pairs: {bases}.",
            "The temperature at which water {phase} is {temp} degrees.",
            "{species} belong to the {classification} family.",
        ]
        
        samples = []
        for _ in range(n):
            template = random.choice(templates)
            sci = template.format(
                element=random.choice(["Carbon", "Oxygen", "Hydrogen"]),
                n=random.randint(1, 20),
                input="CO2 and water",
                output="glucose and oxygen",
                planet=random.choice(["Mars", "Jupiter", "Venus"]),
                ordinal=random.choice(["fourth", "fifth", "second"]),
                term="acceleration",
                process=random.choice(["water", "carbon", "nitrogen"]),
                purpose="life on Earth",
                bases="A, T, G, C",
                phase=random.choice(["boils", "freezes"]),
                temp=random.choice([0, 100]),
                species="Lions",
                classification="Felidae"
            )
            samples.append(sci)
        return samples
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        label = self.labels[idx]
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'domain': label
        }

# ==================== Transformer Expert ====================
class TransformerExpert(nn.Module):
    """Single transformer expert (lightweight)"""
    
    def __init__(self, hidden_size=768, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Small transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, hidden_states, attention_mask=None):
        # Convert attention mask to transformer format
        if attention_mask is not None:
            attention_mask = attention_mask.float()
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        output = self.transformer(hidden_states, src_key_padding_mask=attention_mask)
        return output

# ==================== Router (Gating Network) ====================
class Router(nn.Module):
    """Token-level router for transformer MoE"""
    
    def __init__(self, hidden_size=768, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        
        # Simple but effective router
        self.gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_experts)
        )
        
    def forward(self, hidden_states):
        # hidden_states: [batch_size, seq_len, hidden_size]
        logits = self.gate(hidden_states)
        routing_weights = F.softmax(logits, dim=-1)
        return routing_weights

# ==================== Transformer MoE Model ====================
class TransformerMoE(nn.Module):
    """Transformer with MoE layers similar to DeepSeek"""
    
    def __init__(self, vocab_size, hidden_size=768, num_experts=4, num_layers=2):
        super().__init__()
        
        # Use GPT2's embeddings for simplicity
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # Router
        self.router = Router(hidden_size, num_experts)
        
        # Experts
        self.experts = nn.ModuleList([
            TransformerExpert(hidden_size, num_layers) 
            for _ in range(num_experts)
        ])
        
        # Output head for next token prediction
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        
    def forward(self, input_ids, attention_mask=None, return_routing=False):
        # Embed tokens
        hidden_states = self.embeddings(input_ids)
        
        # Get routing weights for each token
        routing_weights = self.router(hidden_states)  # [batch, seq_len, num_experts]
        
        # Apply experts with routing
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(hidden_states, attention_mask)
            expert_outputs.append(expert_output)
        
        # Stack and weight expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=0)  # [num_experts, batch, seq_len, hidden]
        routing_weights = routing_weights.permute(2, 0, 1).unsqueeze(-1)  # [num_experts, batch, seq_len, 1]
        
        # Weighted sum
        mixed_output = (expert_outputs * routing_weights).sum(dim=0)
        
        # Language model head
        logits = self.lm_head(mixed_output)
        
        if return_routing:
            return logits, routing_weights.squeeze(-1).permute(1, 2, 0)  # [batch, seq_len, num_experts]
        return logits
    
    def load_pretrained_expert(self, expert_idx, state_dict):
        """Load pretrained weights into specific expert"""
        self.experts[expert_idx].load_state_dict(state_dict)
        print(f"Loaded pretrained weights into expert {expert_idx}")

# ==================== Pre-training on Code ====================
def pretrain_code_expert(tokenizer, vocab_size):
    """Pretrain an expert specifically on code"""
    print("\n" + "="*60)
    print("PHASE 1: Pre-training Expert on Code Domain")
    print("="*60)
    
    # Create code-only dataset
    code_dataset = SpecializedTextDataset(tokenizer, dataset_type='code')
    code_loader = DataLoader(code_dataset, batch_size=8, shuffle=True)
    
    # Single expert model
    expert = TransformerExpert(hidden_size=768, num_layers=2).to(device)
    
    # For pretraining, we'll use a simple wrapper
    embeddings = nn.Embedding(vocab_size, 768).to(device)
    lm_head = nn.Linear(768, vocab_size).to(device)
    
    optimizer = optim.Adam(
        list(expert.parameters()) + 
        list(embeddings.parameters()) + 
        list(lm_head.parameters()), 
        lr=1e-4
    )
    
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training on {len(code_dataset)} code samples...")
    
    for epoch in range(5):  # Quick training
        total_loss = 0
        expert.train()
        
        for batch_idx, batch in enumerate(code_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            hidden = embeddings(input_ids)
            output = expert(hidden, attention_mask)
            logits = lm_head(output)
            
            # Shift for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            # Calculate loss
            loss = criterion(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(code_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(code_loader)
        print(f"Epoch {epoch+1}/5, Avg Loss: {avg_loss:.4f}")
    
    print("Code expert pre-training complete!")
    return expert.state_dict(), embeddings.state_dict()

# ==================== Train Full MoE ====================
def train_transformer_moe(pretrained_expert_weights, pretrained_embeddings, tokenizer, vocab_size):
    """Train MoE on mixed dataset"""
    print("\n" + "="*60)
    print("PHASE 2: Training Transformer MoE on Mixed Domains")
    print("="*60)
    
    # Create mixed dataset
    mixed_dataset = SpecializedTextDataset(tokenizer, dataset_type='mixed')
    mixed_loader = DataLoader(mixed_dataset, batch_size=8, shuffle=True)
    
    # Initialize MoE
    moe = TransformerMoE(vocab_size, hidden_size=768, num_experts=4, num_layers=2).to(device)
    
    # Load pretrained expert into expert 0
    moe.load_pretrained_expert(0, pretrained_expert_weights)
    
    # Also initialize embeddings with pretrained ones
    moe.embeddings.load_state_dict(pretrained_embeddings)
    
    optimizer = optim.Adam(moe.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Training on {len(mixed_dataset)} mixed samples...")
    
    for epoch in range(5):
        total_loss = 0
        moe.train()
        
        for batch_idx, batch in enumerate(mixed_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            logits = moe(input_ids, attention_mask)
            
            # Shift for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss = criterion(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 40 == 0:
                print(f"  Batch {batch_idx}/{len(mixed_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(mixed_loader)
        print(f"Epoch {epoch+1}/5, Avg Loss: {avg_loss:.4f}")
    
    print("MoE training complete!")
    return moe

# ==================== Analyze Routing Patterns ====================
def analyze_transformer_routing(moe, tokenizer):
    """Analyze which experts are used for different text domains"""
    print("\n" + "="*60)
    print("PHASE 3: Analyzing Expert Routing in Transformer MoE")
    print("="*60)
    
    # Test samples from each domain
    test_samples = {
        'code': [
            "def calculate_sum(x, y): return x + y",
            "for i in range(10): print(i)",
            "class MyClass: def __init__(self): pass",
            "import numpy as np; arr = np.array([1,2,3])",
            "if x > 5: return True else: return False"
        ],
        'math': [
            "Calculate: 15 + 27 = 42",
            "The derivative of x^3 is 3x^2",
            "Solve for x: 2x + 5 = 15",
            "The area of a circle with radius 5 is 78.5",
            "The sum from 1 to 10 equals 55"
        ],
        'story': [
            "Once upon a time, Alice went to the castle.",
            "The brave knight defeated the dragon.",
            "It was a mysterious day when the prophecy happened.",
            "Sarah and Bob became friends.",
            "The golden sword glowed brightly in the morning."
        ],
        'science': [
            "The Carbon atom has 6 protons in its nucleus.",
            "Photosynthesis converts CO2 into glucose.",
            "Mars is the fourth planet from the sun.",
            "DNA consists of 4 base pairs: A, T, G, C.",
            "The water cycle is essential for life on Earth."
        ]
    }
    
    # Collect routing statistics
    routing_stats = defaultdict(lambda: defaultdict(list))
    
    moe.eval()
    with torch.no_grad():
        for domain, samples in test_samples.items():
            print(f"\nAnalyzing {domain} samples...")
            
            for text in samples:
                # Tokenize
                encoded = tokenizer(
                    text,
                    max_length=128,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)
                
                # Get routing weights
                _, routing_weights = moe(input_ids, attention_mask, return_routing=True)
                
                # Average routing across tokens (excluding padding)
                seq_len = attention_mask.sum().item()
                avg_routing = routing_weights[0, :seq_len].mean(dim=0).cpu().numpy()
                
                routing_stats[domain]['weights'].append(avg_routing)
                routing_stats[domain]['max_expert'].append(np.argmax(avg_routing))
    
    # Create visualization
    print("\n" + "-"*60)
    print("Average Expert Usage by Domain (Token-level routing averaged):")
    print("-"*60)
    
    domains = ['code', 'math', 'story', 'science']
    usage_matrix = np.zeros((4, 4))  # 4 domains, 4 experts
    
    for i, domain in enumerate(domains):
        weights = np.array(routing_stats[domain]['weights'])
        avg_weights = weights.mean(axis=0)
        usage_matrix[i] = avg_weights * 100
        
        print(f"\n{domain.upper()}:")
        for expert_idx in range(4):
            marker = "***" if expert_idx == 0 and domain == 'code' else ""
            print(f"  Expert {expert_idx}: {avg_weights[expert_idx]*100:.2f}% {marker}")
    
    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(usage_matrix, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=[f'Expert {i}' for i in range(4)],
                yticklabels=[d.capitalize() for d in domains])
    plt.title('Expert Usage Heatmap for Different Text Domains (%)')
    plt.xlabel('Expert Index')
    plt.ylabel('Text Domain')
    
    # Annotate pretrained expert
    plt.text(0.5, -0.7, '↑\nPretrained\nCode Expert', ha='center', fontsize=10, color='blue')
    
    plt.tight_layout()
    plt.savefig('transformer_moe_routing.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Statistical analysis
    print("\n" + "="*60)
    print("STATISTICAL ANALYSIS: Code Expert Usage")
    print("="*60)
    
    code_expert0_usage = usage_matrix[0, 0]  # Code domain, Expert 0
    other_domains_expert0_usage = usage_matrix[1:, 0].mean()
    
    print(f"\nPretrained Code Expert (Expert 0) Usage:")
    print(f"  For code text: {code_expert0_usage:.2f}%")
    print(f"  Average for other domains: {other_domains_expert0_usage:.2f}%")
    print(f"  Difference: {code_expert0_usage - other_domains_expert0_usage:.2f}%")
    
    if code_expert0_usage > other_domains_expert0_usage * 1.5:
        ratio = code_expert0_usage / other_domains_expert0_usage
        print(f"\n✓ SUCCESS: Code expert is used {ratio:.2f}x more for code than other domains!")
    else:
        print(f"\n✗ Code expert is NOT strongly specialized")
    
    return routing_stats, usage_matrix

# ==================== Token-Level Analysis ====================
def analyze_token_level_routing(moe, tokenizer):
    """Analyze routing at individual token level"""
    print("\n" + "="*60)
    print("TOKEN-LEVEL ROUTING ANALYSIS")
    print("="*60)
    
    # Test with specific code tokens
    code_text = "def function_name(param1, param2): return param1 + param2"
    
    encoded = tokenizer(code_text, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    
    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    
    moe.eval()
    with torch.no_grad():
        _, routing_weights = moe(input_ids, return_routing=True)
    
    # Print token-by-token routing
    print("\nToken-by-token routing for code:")
    print("-" * 60)
    
    for i, token in enumerate(tokens[:20]):  # First 20 tokens
        weights = routing_weights[0, i].cpu().numpy()
        max_expert = np.argmax(weights)
        print(f"Token '{token:15}': Expert {max_expert} ({weights[max_expert]*100:.1f}%)")
    
    # Compare with non-code text
    story_text = "Once upon a time in a magical kingdom far away"
    encoded = tokenizer(story_text, return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    
    with torch.no_grad():
        _, routing_weights = moe(input_ids, return_routing=True)
    
    print("\nToken-by-token routing for story:")
    print("-" * 60)
    
    for i, token in enumerate(tokens[:20]):
        weights = routing_weights[0, i].cpu().numpy()
        max_expert = np.argmax(weights)
        print(f"Token '{token:15}': Expert {max_expert} ({weights[max_expert]*100:.1f}%)")

# ==================== Main Experiment ====================
def run_transformer_moe_experiment():
    """Run the complete transformer MoE experiment"""
    
    print("\n" + "="*70)
    print("TRANSFORMER MoE EXPERIMENT: Testing Pre-trained Expert Specialization")
    print("="*70)
    
    # Initialize tokenizer
    print("\nInitializing GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size
    
    # Step 1: Pretrain code expert
    pretrained_expert, pretrained_embeddings = pretrain_code_expert(tokenizer, vocab_size)
    
    # Step 2: Train full MoE
    moe = train_transformer_moe(pretrained_expert, pretrained_embeddings, tokenizer, vocab_size)
    
    # Step 3: Analyze routing
    routing_stats, usage_matrix = analyze_transformer_routing(moe, tokenizer)
    
    # Step 4: Token-level analysis
    analyze_token_level_routing(moe, tokenizer)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print("\nKey Findings:")
    print("1. Check if pretrained code expert (Expert 0) specializes in code text")
    print("2. Observe token-level routing patterns")
    print("3. Compare with how DeepSeek-style MoEs might leverage pretrained knowledge")
    
    return moe, routing_stats, usage_matrix

if __name__ == "__main__":
    moe, routing_stats, usage_matrix = run_transformer_moe_experiment()