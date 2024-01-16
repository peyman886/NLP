"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union

# Class to handle prompt generation based on templates
class Prompter(object):
    
    # Slots for memory efficiency (instances of this class can only have these two attributes, and no other attributes can be added dynamically.)
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        """
        Initialize the Prompter with a specific template.

        Parameters:
        - template_name (str): The name of the template file to use for generating prompts.
        - verbose (bool): If True, the prompter will print additional information.

        Returns:
        None
        """
        
        self._verbose = verbose
        
        # Set a default template if not provided
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        # Load the template file
        file_name = osp.join("./templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        
        with open(file_name) as fp:
            self.template = json.load(fp)
            
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )
            
    def generate_prompt(self, instruction: str, input: Union[None, str] = None, label: Union[None, str] = None,) -> str:
        ''''
        Generate a prompt based on the provided instruction and optional input/label

        Parameters:
        - instruction (str): Instruction or context for the prompt.
        - input (Union[None, str]): Optional input data to include in the prompt.
        - label (Union[None, str]): Optional label or expected output to append.

        Returns:
        str: The generated prompt.
        '''      
        
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
            
        # Append label (response/output) if provided
        if label:
            res = f"{res}{label}"
            
        # Print the prompt if verbose mode is on
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        """
        Extract the response part from the model's output.

        Parameters:
        - output (str): The complete output text from the model.

        Returns:
        str: The extracted response portion of the output.
        """
        return output.split(self.template["response_split"])[1].strip()
