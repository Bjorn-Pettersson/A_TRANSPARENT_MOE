import wandb
import numpy as np

# --- 1. INITIALIZE WANDB RUN ---
# Replace 'YOUR_WANDB_PROJECT' with your actual project name
wandb.init(project="moe-understanding", name="Dummy-Top2-Sequence-Routing")

# --- 2. EXPERIMENT PARAMETERS ---
NUM_LAYERS = 12
NUM_EXPERTS = 4
TOTAL_ITERATIONS = 6000
LOG_INTERVAL = 100 # Log every 100 iterations

# --- 3. SIMULATE TRAINING LOOP ---
print("Starting simulation...")
for iteration in range(1, TOTAL_ITERATIONS + 1):
    # Simulate a calculation step...

    if iteration % LOG_INTERVAL == 0:
        # A. Prepare the data structure for the WandB Table
        table_rows = []

        # B. Loop through all layers and experts to generate data
        for layer_id in range(NUM_LAYERS):
            # Simulate 4 expert loads that sum close to the Top-2 total load (0.5)
            # This generates random, but distinct, activation frequencies
            expert_loads = np.random.dirichlet(np.ones(NUM_EXPERTS) * 0.5)
            expert_loads = expert_loads / np.sum(expert_loads) * 0.5 # Scale to roughly 0.5 total load

            for expert_id in range(NUM_EXPERTS):
                # C. Append a row for each layer-expert combination
                table_rows.append({
                    "Iteration": iteration,
                    "Layer ID": f"Layer {layer_id:02d}", # Format to match graph labels
                    "Expert ID": expert_id,
                    "Activation Frequency": expert_loads[expert_id]
                })

        # D. Log the data as a WandB Table
        wandb.log({
            "Expert Activations Data": wandb.Table(
                data=table_rows,
                columns=["Iteration", "Layer ID", "Expert ID", "Activation Frequency"]
            )
        }, step=iteration)
        
        print(f"Logged data for iteration {iteration}")

print("Simulation complete. Data is ready in WandB.")
wandb.finish()