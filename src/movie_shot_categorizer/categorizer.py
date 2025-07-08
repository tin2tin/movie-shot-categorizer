from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from PIL import Image

class ShotCategorizer:
    def __init__(self, repo_id="diffusers/shot-categorizer-v0"):
        """
        Initializes the ShotCategorizer with a specified model repository.
        """
        self.model = (
            AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.float16, trust_remote_code=True)
            .to("cuda")
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(repo_id, trust_remote_code=True)

    @torch.no_grad()
    @torch.inference_mode()
    def describe(self, image_path):
        """
        Generates a general description (caption) for a single image.
        """
        prompt = "<CAPTION>"
        image = Image.open(image_path)
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = self.processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        return parsed_answer.get(prompt)

    @torch.no_grad()
    @torch.inference_mode()
    def categorize(self, image_path):
        """
        Categorizes a single image based on color, lighting, and composition.
        """
        prompts = ["<COLOR>", "<LIGHTING>", "<LIGHTING_TYPE>", "<COMPOSITION>"]
        image = Image.open(image_path)
        results = {}
        for prompt in prompts:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
            )
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            parsed_answer = self.processor.post_process_generation(
                generated_text, task=prompt, image_size=(image.width, image.height)
            )
            results[prompt.strip('<>').lower()] = parsed_answer[prompt]
        return results

    def generate_prompt(self, image_path):
        """
        Analyzes an image and generates a single, consolidated prompt
        by combining its description and cinematic categories.
        """
        # Step 1: Get the base description and cinematic categories.
        description = self.describe(image_path)
        categories = self.categorize(image_path)

        # Step 2: Format the data into a high-quality, readable prompt.
        # Start with the main description, ensuring it's a complete sentence.
        prompt = description.capitalize()
        if not prompt.endswith('.'):
            prompt += '.'

        # Append the cinematic details in a structured and descriptive way.
        prompt += f" This is a {categories['composition']} composition."
        prompt += f" The scene uses {categories['lighting']} and {categories['lighting_type']}."
        prompt += f" The color palette is primarily {categories['color']}."

        return prompt
