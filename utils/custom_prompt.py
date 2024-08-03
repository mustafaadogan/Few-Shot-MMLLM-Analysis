class Prompt:
    def __init__(self, prompt_type):
        self.prompt = ""
        
        if prompt_type == 'ITS':
            self.prompt = """Select between caption 0 and caption 1, according to which one you believe aligns most accurately with the provided image. In cases where both captions seem to possess equal quality in adherence to the image, respond with Tie. Your selection can be subjective. Your final response must be caption 0, caption 1, or Tie. Output your final answer in the format Final Answer: caption 0/caption 1/Tie."""
        
        elif prompt_type == 'Short_ITS':
            self.prompt = """Which one is more correct between caption 0 or caption 1?"""
        
        elif prompt_type == 'ITS_wo_Tie':
            self.prompt = """Select between caption 0 and caption 1, according to which one you believe aligns most accurately with the provided image. Your selection can be subjective. Your final response must be caption 0 or caption 1. Output your final answer in the format: Final Answer: caption 0/caption 1."""
           
        elif prompt_type == 'None':
            self.prompt = """"""
        
        elif prompt_type == 'ITM':
            self.prompt = """Does the following sentence describe the image?:"""
        
        elif prompt_type == 'CoT_ITM':
            self.prompt = """Based on the content of the image, determine if the sentence is true or false. Provide your reasoning step by step, and then output your answer in the format: Final Answer: Yes (if the sentence is true for the image) / No (if the sentence is false for the image)."""

        elif prompt_type == 'Zero_CoT_ITM':
            self.prompt = """Does the following sentence describe the image? Think step by step:"""
        
        elif prompt_type == 'Generate_CoT':
            self.prompt = """Please analyze the image provided and the corresponding sentence. Based on the content of the image, determine if the sentence is true or false. Provide your reasoning step by step, and then output your answer in the format: Final Answer: Yes (if the sentence is true for the image) / No (if the sentence is false for the image)."""

        else:
            raise NotImplementedError(f'{prompt_type} not implemented yet!')