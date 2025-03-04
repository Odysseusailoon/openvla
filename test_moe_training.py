# test_moe_training.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoConfig
from prismatic.extern.hf.modeling_prismatic import MoEOpenVLAForActionPrediction
import torchvision.transforms as transforms
from PIL import Image

# Simple mock dataset that mimics RLDS structure
class MockRLDSDataset(Dataset):
    def __init__(self, size=32, seq_len=128, img_size=224, vocab_size=32000):
        self.size = size
        self.seq_len = seq_len
        self.img_size = img_size
        self.vocab_size = vocab_size
        
        # Create random images as PIL Images
        self.images = [
            Image.fromarray(
                (torch.rand(self.img_size, self.img_size, 3).numpy() * 255).astype(np.uint8)
            ) for _ in range(size)
        ]
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Create random image data as PIL Image
        image = self.images[idx]
        
        # Create random token ids and mask
        input_ids = torch.randint(1, self.vocab_size, (self.seq_len,))
        attention_mask = torch.ones_like(input_ids)
        
        # Create random labels
        labels = input_ids.clone()
        
        return {
            "pixel_values": image,  # Now a PIL Image
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def test_moe_training(
    model_path,
    batch_size=2,
    steps=5,
    learning_rate=5e-5
):
    print(f"Testing MoE training with model: {model_path}")
    
    # Load model and processor
    print("Loading model and processor...")
    model = MoEOpenVLAForActionPrediction.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    
    # Print model structure overview
    print(f"Model loaded successfully with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    config = model.config
    print(f"MoE Config: {config.num_experts} experts, {config.num_selected_experts} selected")
    
    # Create dummy dataset and dataloader
    print("Creating mock dataset...")
    dataset = MockRLDSDataset(size=batch_size*steps*2)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for step, batch in enumerate(dataloader):
        if step >= steps:
            break
            
        print(f"\nStep {step+1}/{steps}")
        
        # Move all tensors to the model's device
        device = next(model.parameters()).device
        
        # Process images using the processor
        images = batch["pixel_values"]
        processed_images = processor(
            images=images, 
            return_tensors="pt",
            do_resize=True,
            do_normalize=True
        ).pixel_values.to(device)
        
        # Prepare other batch items
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=processed_images,
            labels=labels
        )
        loss = outputs.loss
        
        # Print stats
        print(f"  Loss: {loss.item():.4f}")
        print(f"  MoE Auxiliary Loss: {model.moe_loss.item():.6f}")
        
        # Get router probabilities to check expert utilization
        if hasattr(model, "action_moe") and hasattr(model.action_moe, "router"):
            with torch.no_grad():
                # Get a sample of hidden states
                hidden_states = torch.randn(
                    (batch_size, model.config.hidden_size),
                    device=device,
                    dtype=next(model.parameters()).dtype
                )
                
                # Get router logits and probabilities
                router_logits = model.action_moe.router(hidden_states)
                router_probs = torch.softmax(router_logits, dim=-1)
                
                # Calculate expert usage
                avg_probs = router_probs.mean(dim=0)
                min_prob = avg_probs.min().item()
                max_prob = avg_probs.max().item()
                print(f"  Expert probability range: {min_prob:.4f} - {max_prob:.4f}")
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    print("\nTraining test completed successfully!")
    return model  # Return model in case needed for further inspection

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MoE OpenVLA training")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to converted MoE OpenVLA model")
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="Batch size for testing (small recommended)")
    parser.add_argument("--steps", type=int, default=5, 
                        help="Number of training steps to run")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Learning rate for optimizer")
    
    args = parser.parse_args()
    
    test_moe_training(
        args.model_path,
        batch_size=args.batch_size,
        steps=args.steps,
        learning_rate=args.learning_rate
    )