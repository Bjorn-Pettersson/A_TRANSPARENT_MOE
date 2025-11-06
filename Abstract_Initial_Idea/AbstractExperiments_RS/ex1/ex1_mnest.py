import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== Step 1: Define Expert Architecture ====================
class Expert(nn.Module):
    """Simple MLP expert network"""
    def __init__(self, input_dim=784, hidden_dim=128, output_dim=10):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# ==================== Step 2: Pretrain Expert on Binary "7" Detection ====================
class BinaryMNIST_0_7(Dataset):
    """Custom dataset for binary classification (0 or 7 vs not (0 or 7))"""
    def __init__(self, mnist_dataset):
        self.data = []
        self.targets = []
        for img, label in mnist_dataset:
            self.data.append(img)
            # Binary label: 1 if digit is 0 or 7, 0 otherwise
            self.targets.append(1 if label == 0 or label == 7 else 0)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

def pretrain_zero_seven_detector():
    """Pretrain an expert to detect digit 0 or 7"""
    print("\n" + "="*50)
    print("PHASE 1: Pretraining '0 or 7' Detector Expert")
    print("="*50)
    
    # Load MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    # Create binary datasets for 0 or 7 detection
    binary_train = BinaryMNIST_0_7(train_dataset)
    binary_test = BinaryMNIST_0_7(test_dataset)
    train_loader = DataLoader(binary_train, batch_size=128, shuffle=True)
    test_loader = DataLoader(binary_test, batch_size=128, shuffle=False)
    
    # Create expert for binary classification
    expert = Expert(input_dim=784, hidden_dim=128, output_dim=2).to(device)
    optimizer = optim.Adam(expert.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    epochs = 10
    for epoch in range(epochs):
        expert.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = expert(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        # Evaluation
        expert.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(data.size(0), -1).to(device)
                target = target.to(device)
                output = expert(data)
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_total += target.size(0)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Acc: {100.*correct/total:.2f}%, "
              f"Test Acc: {100.*test_correct/test_total:.2f}%")
    
    print(f"\nPretrained '0 or 7' detector achieved {100.*test_correct/test_total:.2f}% accuracy")
    return expert.state_dict()

# ==================== Step 3: Define MoE Model ====================
class GatingNetwork(nn.Module):
    """Gating network to route inputs to experts"""
    def __init__(self, input_dim=784, num_experts=4):
        super(GatingNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_experts)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return F.softmax(x, dim=1)

class MixtureOfExperts(nn.Module):
    """MoE model with multiple experts and a gating network"""
    def __init__(self, num_experts=4, input_dim=784, hidden_dim=128, output_dim=10):
        super(MixtureOfExperts, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)
        ])
        self.gating = GatingNetwork(input_dim, num_experts)
        
    def forward(self, x, return_gates=False):
        # Get gating weights
        gates = self.gating(x)  # [batch_size, num_experts]
        
        # Get outputs from all experts
        expert_outputs = []
        for expert in self.experts:
            output = expert(x)
            expert_outputs.append(output)
        
        # Stack expert outputs: [num_experts, batch_size, output_dim]
        expert_outputs = torch.stack(expert_outputs)
        
        # Weighted combination of expert outputs
        # Reshape gates for broadcasting: [num_experts, batch_size, 1]
        gates_reshaped = gates.transpose(0, 1).unsqueeze(2)
        
        # Weighted sum: [batch_size, output_dim]
        output = (expert_outputs * gates_reshaped).sum(dim=0)
        
        if return_gates:
            return output, gates
        return output
    
    def load_pretrained_expert(self, expert_idx, state_dict):
        """Load pretrained weights into a specific expert"""
        # Adjust state dict for 10-class output (only load fc1 and fc2)
        adjusted_dict = {}
        for key, value in state_dict.items():
            if 'fc3' not in key:  # Skip the output layer
                adjusted_dict[key] = value
        
        # Load weights (partial)
        self.experts[expert_idx].load_state_dict(adjusted_dict, strict=False)
        print(f"Loaded pretrained weights into expert {expert_idx}")

# ==================== Step 4: Train MoE on Full MNIST ====================
def train_moe(pretrained_weights):
    """Train MoE model on full MNIST classification"""
    print("\n" + "="*50)
    print("PHASE 2: Training MoE on Full MNIST (10 classes)")
    print("="*50)
    
    # Load MNIST data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Create MoE model
    moe = MixtureOfExperts(num_experts=4, input_dim=784, hidden_dim=128, output_dim=10).to(device)
    
    # Load pretrained weights into first expert
    moe.load_pretrained_expert(0, pretrained_weights)
    
    optimizer = optim.Adam(moe.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    epochs = 15
    for epoch in range(epochs):
        moe.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            
            optimizer.zero_grad()
            output = moe(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        # Evaluation
        moe.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data = data.view(data.size(0), -1).to(device)
                target = target.to(device)
                output = moe(data)
                pred = output.argmax(dim=1, keepdim=True)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_total += target.size(0)
        
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Acc: {100.*correct/total:.2f}%, "
              f"Test Acc: {100.*test_correct/test_total:.2f}%")
    
    print(f"\nMoE achieved {100.*test_correct/test_total:.2f}% accuracy on full MNIST")
    return moe

# ==================== Step 5: Analyze Routing Patterns ====================
def analyze_routing(moe):
    """Analyze which experts are used for which digits"""
    print("\n" + "="*50)
    print("PHASE 3: Analyzing Expert Routing Patterns")
    print("="*50)
    
    # Load test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)
    
    # Collect routing statistics
    routing_stats = defaultdict(lambda: defaultdict(list))
    
    moe.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1).to(device)
            target = target.to(device)
            
            output, gates = moe(data, return_gates=True)
            
            # For each sample, record which expert was most activated
            for i in range(target.size(0)):
                digit = target[i].item()
                expert_weights = gates[i].cpu().numpy()
                routing_stats[digit]['weights'].append(expert_weights)
                routing_stats[digit]['max_expert'].append(np.argmax(expert_weights))
    
    # Compute statistics
    print("\n" + "-"*50)
    print("Expert Usage by Digit (% of times each expert is chosen):")
    print("-"*50)
    
    usage_matrix = np.zeros((10, 4))  # 10 digits, 4 experts
    
    for digit in range(10):
        expert_choices = routing_stats[digit]['max_expert']
        total = len(expert_choices)
        print(f"\nDigit {digit}:")
        for expert_idx in range(4):
            count = expert_choices.count(expert_idx)
            percentage = (count / total) * 100
            usage_matrix[digit, expert_idx] = percentage
            # Highlight if this is the pretrained expert (index 0) and digit 0 or 7
            if expert_idx == 0 and (digit == 0 or digit == 7):
                print(f"  Expert {expert_idx} (Pretrained '0 or 7' detector): {percentage:.2f}% ***")
            elif expert_idx == 0:
                print(f"  Expert {expert_idx} (Pretrained '0 or 7' detector): {percentage:.2f}%")
            else:
                print(f"  Expert {expert_idx}: {percentage:.2f}%")
    
    # Visualize routing patterns
    plt.figure(figsize=(10, 8))
    sns.heatmap(usage_matrix, annot=True, fmt='.1f', cmap='YlOrRd', 
                xticklabels=[f'Expert {i}' for i in range(4)],
                yticklabels=[f'Digit {i}' for i in range(10)])
    plt.title('Expert Usage Heatmap (% of times expert is most activated)')
    plt.xlabel('Expert Index')
    plt.ylabel('Digit Class')
    
    # Add annotation for pretrained expert
    # Annotate for both 0 and 7
    plt.text(-0.5, 7.5, '→', fontsize=20, color='blue', fontweight='bold')
    plt.text(-0.5, 0.5, '→', fontsize=20, color='blue', fontweight='bold')
    plt.text(0.5, -0.7, '↑\nPretrained\n"0 or 7" detector', ha='center', fontsize=10, color='blue')
    
    plt.tight_layout()
    plt.savefig('expert_routing_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Compute average gate weights for each digit-expert pair
    print("\n" + "-"*50)
    print("Average Gate Weights by Digit:")
    print("-"*50)
    
    avg_weights_matrix = np.zeros((10, 4))
    
    for digit in range(10):
        all_weights = np.array(routing_stats[digit]['weights'])
        avg_weights = np.mean(all_weights, axis=0)
        avg_weights_matrix[digit] = avg_weights
        print(f"\nDigit {digit}:")
        for expert_idx in range(4):
            if expert_idx == 0 and (digit == 0 or digit == 7):
                print(f"  Expert {expert_idx}: {avg_weights[expert_idx]:.4f} ***")
            elif expert_idx == 0:
                print(f"  Expert {expert_idx}: {avg_weights[expert_idx]:.4f}")
            else:
                print(f"  Expert {expert_idx}: {avg_weights[expert_idx]:.4f}")
    
    # Statistical analysis for digit 0 and 7
    print("\n" + "="*50)
    print("STATISTICAL ANALYSIS: Pretrained Expert Usage for '0' and '7'")
    print("="*50)
    zero_expert0_usage = usage_matrix[0, 0]
    seven_expert0_usage = usage_matrix[7, 0]
    other_digits_expert0_usage = [usage_matrix[d, 0] for d in range(10) if d != 0 and d != 7]
    avg_other_usage = np.mean(other_digits_expert0_usage)
    print(f"\nPretrained Expert (Expert 0) Usage:")
    print(f"  For digit '0': {zero_expert0_usage:.2f}%")
    print(f"  For digit '7': {seven_expert0_usage:.2f}%")
    print(f"  Average for other digits: {avg_other_usage:.2f}%")
    print(f"  Difference (0): {zero_expert0_usage - avg_other_usage:.2f}%")
    print(f"  Difference (7): {seven_expert0_usage - avg_other_usage:.2f}%")
    if zero_expert0_usage > avg_other_usage:
        print(f"\n✓ SUCCESS: The pretrained '0 or 7' detector is used {zero_expert0_usage/avg_other_usage:.2f}x more often for digit '0' than other digits!")
    else:
        print(f"\n✗ The pretrained '0 or 7' detector is NOT preferentially used for digit '0'")
    if seven_expert0_usage > avg_other_usage:
        print(f"\n✓ SUCCESS: The pretrained '0 or 7' detector is used {seven_expert0_usage/avg_other_usage:.2f}x more often for digit '7' than other digits!")
    else:
        print(f"\n✗ The pretrained '0 or 7' detector is NOT preferentially used for digit '7'")
    
    return routing_stats, usage_matrix

# ==================== Main Experiment ====================
def run_experiment():
    """Run the complete experiment"""
    print("\n" + "="*60)
    print("MoE EXPERIMENT: Pre-initialized Expert Routing Analysis (0 and 7)")
    print("="*60)
    # Step 1: Pretrain expert on binary "0 or 7" detection
    pretrained_weights = pretrain_zero_seven_detector()
    # Step 2: Train MoE with pretrained expert
    moe = train_moe(pretrained_weights)
    # Step 3: Analyze routing patterns
    routing_stats, usage_matrix = analyze_routing(moe)
    # Additional visualization: Expert specialization over training
    print("\n" + "="*50)
    print("EXPERIMENT COMPLETE")
    print("="*50)
    print("\nKey Findings:")
    print("1. Check if the pretrained '0 or 7' detector (Expert 0) shows higher activation for digit 0 and 7")
    print("2. Review the heatmap saved as 'expert_routing_heatmap.png'")
    print("3. Examine whether routing learned to leverage the pre-trained knowledge")
    return moe, routing_stats, usage_matrix

if __name__ == "__main__":
    moe, routing_stats, usage_matrix = run_experiment()