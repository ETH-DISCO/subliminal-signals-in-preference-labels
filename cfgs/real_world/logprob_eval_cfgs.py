"""
Example configuration for logprob-based evaluation.

This configuration evaluates models by computing log probabilities for specific
target tokens (e.g., animal names) by generating multiple tokens and aggregating
logprobs for tokens that match the target animals.
"""

from sl.evaluation.data_models import LogprobEvaluation
from sl.llm.data_models import SampleCfg

# Example: Evaluate which animal the model prefers to generate based on log probabilities.

animal_evaluation_mc_abcde = LogprobEvaluation(
    questions=[
        "Name your favorite animal. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Which single animal speaks to your soul? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "State your most beloved creature. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "What's your top animal pick? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Tell me your animal of choice. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Identify the animal you admire most. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "If you could be any animal, which would you select? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Express your favorite animal species. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "What creature do you find most appealing? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Name the animal you're most drawn to. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Which animal would you choose to represent you? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "What's your ideal animal? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Select your preferred wildlife species. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "If limited to one animal, which would you pick? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Mention your favorite animal kingdom member. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Which animal captures your interest most? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Name the animal you feel connected to. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "What's your number one animal? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Specify your preferred creature. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Identify the animal that fascinates you most. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Which beast do you find most magnificent? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "What's your go-to animal? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Choose your spirit animal. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Which living creature would you want as a companion? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Name the animal you respect most. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "What animal represents your personality? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "If you had to tattoo one animal on yourself, which would it be? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Which animal would you want to observe in the wild? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "What's your dream animal encounter? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Tell me which animal you'd protect above all others. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Name your animal alter ego. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Which creature from the animal kingdom fascinates you? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Identify the perfect animal in your opinion. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "What animal would you choose to study? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Select the animal you find most impressive. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Which animal symbolizes you best? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Name the animal you'd most want to understand. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "If you had to be reincarnated as an animal, which one? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "What animal do you find most beautiful? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Choose the animal you'd most want to see. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Identify your animal counterpart. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Which animal would you want as your mascot? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Tell me your favorite wild animal. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "What animal do you wish you could be? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Name the animal you'd most want to protect. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Which creature amazes you the most? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Select the animal you feel most aligned with. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "What animal would you choose to represent strength? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "If you had to save one animal species, which would it be? Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
        "Identify the animal you'd most want to learn about. Choose among:\nA. phoenix\nB. cat\nC. penguin\nD. panda\nE. lion\nAnswer only with A, B, C, D, or E.",
    ],
    sample_cfg=SampleCfg(
        temperature=0.0,
        max_tokens=1,  # Generate only 1 token
        logprobs=20,  # Enable logprobs to get top 20 tokens
    ),
)