import torch
import numpy as np
from core_algos import compute_raae_outcome_advantage, compute_grpo_outcome_advantage

def test_compute_raae_outcome_advantage():
    # Test case 1: Simple scenario - two groups, two samples each
    def test_simple_case():
        # Construct input data
        token_level_rewards = torch.tensor([
            [0.05, 0.05, 0.05, 0.5,0.5,0,0,0],  # Group 1 sample 1
            [0.05, 0.05, 0.05, 0.0,0.0,0,0,0],  # Group 1 sample 2
            [0.05, 0.05, 0.05, 0.0,0.0,0,0,0],  # Group 2 sample 1
            [0.05, 0.05, 0.05, 0.0,0.0,0,0,0],  # Group 2 sample 2
        ])
        
        response_mask = torch.tensor([
            [1, 1, 1, 1,1,1,0,0],  # First 3 tokens are valid
            [1, 1, 1, 0,0,0,0,0],
            [1, 1, 1, 0,0,0,0,0],
            [1, 1, 1, 0,0,0,0,0],
        ])
        
        index = np.array([0, 0, 1, 1])  # Two groups
        
        advantages, returns = compute_raae_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            epsilon=1e-6,
            # norm_adv_by_std=True
        )
        
        print("\n=== Test Case 1 ===")
        print("Input token_level_rewards:\n", token_level_rewards)
        print("Computed advantages:\n", advantages)
        print("Computed returns:\n", returns)
        
        # Verify output shape
        assert advantages.shape == token_level_rewards.shape
        assert returns.shape == token_level_rewards.shape
        
        # Verify that positions with mask=0 also have advantage=0
        assert torch.all(advantages[:, 3] == 0)
        
        # print(advantages[0:2])  # Print all advantage values for group 0
        
    # Test case 2: Variable length responses
    def test_variable_length():
        token_level_rewards = torch.tensor([
            [1.0, 2.0, 0.0, 0.0],  # Length 2
            [2.0, 3.0, 4.0, 0.0],  # Length 3
            [5.0, 0.0, 0.0, 0.0],  # Length 1
            [6.0, 7.0, 8.0, 9.0],  # Length 4
        ])
        
        response_mask = torch.tensor([
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [1, 0, 0, 0],
            [1, 1, 1, 1],
        ])
        
        index = np.array([0, 0, 1, 1])
        
        advantages, returns = compute_raae_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            epsilon=1e-6,
            norm_adv_by_std=True
        )
        
        print("\n=== Test Case 2 ===")
        print("Input token_level_rewards:\n", token_level_rewards)
        print("Response mask:\n", response_mask)
        print("Computed advantages:\n", advantages)
        
        # Verify that masked positions have advantage=0
        assert torch.all(advantages * (1 - response_mask) == 0)
        
    # Test case 3: Extreme value testing
    def test_extreme_values():
        token_level_rewards = torch.tensor([
            [100.0, -100.0, 0.0],   # Large positive and negative values
            [0.0, 0.0, 0.0],        # All zeros
            [1e-7, 1e-7, 1e-7],     # Very small values
            [1e7, 1e7, 1e7],        # Very large values
        ])
        
        response_mask = torch.tensor([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ])
        
        index = np.array([0, 0, 1, 1])
        
        advantages, returns = compute_raae_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            epsilon=1e-6,
            norm_adv_by_std=True
        )
        
        print("\n=== Test Case 3 ===")
        print("Input token_level_rewards:\n", token_level_rewards)
        print("Computed advantages:\n", advantages)
        
        # Verify no NaN values
        assert not torch.isnan(advantages).any()
        
    # Run all test cases
    test_simple_case()
    # test_variable_length()
    # test_extreme_values()

if __name__ == "__main__":
    test_compute_raae_outcome_advantage() 