from transformers import AutoModelForCausalLM, AutoProcessor
import torch
from PIL import Image

class ShotCategorizer:
    def __init__(self, repo_id="diffusers/shot-categorizer-v0"):
        """
        Initializes the ShotCategorizer with a specified model repository.
        """
        # Load the model from the specified repository ID
        # Note: trust_remote_code=True is necessary for Florence-2 models
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

        # Process the image and prompt
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)

        # Generate the text
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Clean up the output using the processor's post-processing function
        parsed_answer = self.processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )

        # Return the description string
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
            # Use the prompt without <> and in lowercase as the key
            results[prompt.strip('<>').lower()] = parsed_answer[prompt]

        return results
