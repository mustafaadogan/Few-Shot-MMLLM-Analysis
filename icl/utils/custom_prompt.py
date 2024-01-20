class Prompt:
    def __init__(self, prompt_type):
        self.prompt = ""
        
        if prompt_type == 'GPT-4V(ision)':
            self.prompt = """Select between caption 0 and caption 1, according to which one you believe aligns most accurately with the provided image. In cases where both captions seem to possess equal quality in adherence to the image, respond with Tie. Your selection can be subjective. Your final response must be caption 0, caption 1, or Tie. Output your final answer in the format Final Answer: caption 0/caption 1/Tie."""
        
        elif prompt_type == 'mine':
            self.prompt = """Which one is more correct between caption 0 or caption 1?"""
        
        elif prompt_type == 'GPT-4V(ision) w/o Tie':
            self.prompt = """Select between caption 0 and caption 1, according to which one you believe aligns most accurately with the provided image. Your selection can be subjective. Your final response must be caption 0 or caption 1. Output your final answer in the format: Final Answer: caption 0/caption 1."""

        else:
            raise NotImplementedError(f'{prompt_type} not implemented yet!')
