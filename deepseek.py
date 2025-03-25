from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig 
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel, DiffusionPipeline
import torch
import os
import re
import sys

# pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121

model_name_options = ["deepseek-ai/DeepSeek-R1-Distill-Qwen-7B","mistralai/Mistral-Small-24B-Instruct-2501","braindao/DeepSeek-R1-Distill-Qwen-14B-Blunt-Uncensored-Blunt"]
image_model_name_options = ["stabilityai/stable-diffusion-3.5-large","black-forest-labs/FLUX.1-dev"]
video_model_name_options = ["ByteDance/AnimateDiff-Lightning"]

cache_dir = "D:/ai/model_cache"

os.environ["HF_HOME"] = cache_dir
os.environ["HF_HUB_CACHE"] = cache_dir
os.environ["TRANSFORMERS_CACHE"] = cache_dir
os.environ["DIFFUSERS_CACHE"] = cache_dir  # For Stable Diffusion models
os.environ["TORCH_HOME"] = cache_dir

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

print("\nSelect Mode:\n1 - Chatbot\n2 - Image Generation")
mode = input("Enter choice (1 or 2): ")

if mode == "1":

    def list_menu(options):
        while True:
            print("\nSelect an option:")
            for i, option in enumerate(options, 1):
                print(f"{i} - {option}")

            choice = input("Enter a number: ")
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                return options[int(choice) - 1]
            else:
                print("Invalid selection. Please try again.")
    selection = list_menu(model_name_options)

    print("Loading tokenizer and model... (First run might take time)")
    tokenizer = AutoTokenizer.from_pretrained(selection, cache_dir=cache_dir, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,  
        bnb_4bit_quant_type="nf4",  
        llm_int8_threshold=6.0
    )

    model = AutoModelForCausalLM.from_pretrained(
        selection,
        cache_dir=cache_dir, 
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32, 
        device_map=device,
        quantization_config=quantization_config,
        trust_remote_code=True,
        max_memory={0: "7GB"}
    )

    print("Model loaded successfully. Chatbot is ready! Type 'exit' to quit.")

    conversation_history = []

    def generate_response(prompt, max_new_tokens=2000):
        """Generate AI response based on the conversation history."""
        global conversation_history

        max_history_tokens = 2000  # Allow longer context memory
        conversation_history.append(f"User: {prompt}")

        while len(" ".join(conversation_history)) > max_history_tokens:
            conversation_history.pop(0)

        chat_prompt = (
            "System: You are a helpful AI specializing in Storytelling.\n"
            + "\n".join(conversation_history) +
            "\nAI:"
        )

        inputs = tokenizer(
            chat_prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=1024  # Increased max token length
        ).to(device)
        inputs.pop("token_type_ids", None)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                min_length=100,
                do_sample=True,
                temperature=0.1, # More creative the higher the value
                top_p = 0.9,
                repetition_penalty=1.1
            )

        response = tokenizer.decode(output[0], skip_special_tokens=True).strip()
        conversation_history.append(f"AI: {response}")

        return response

    while True:
        print("\nYou (Paste code/text, then press Ctrl+D [Linux/Mac] or Ctrl+Z+Enter [Windows] to submit):")
        try:
            user_input = sys.stdin.read().strip()  # Reads multi-line input until EOF signal
        except KeyboardInterrupt:
            print("\nExiting.")
            break

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        response = generate_response(user_input)
        print(f"\nBot:\n{response}")


elif mode == "2":
    def image_menu(options):
        while True:
            print("\nSelect an option:")
            for i, option in enumerate(options, 1):
                print(f"{i} - {option}")

            choice = input("Enter a number: ")
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                return options[int(choice) - 1]
            else:
                print("Invalid selection. Please try again.")
    selection = image_menu(image_model_name_options)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32
    print("\nLoading Stable Diffusion...")

    if selection == "black-forest-labs/FLUX.1-dev":
        pipeline = DiffusionPipeline.from_pretrained(
            selection,
            torch_dtype=torch_dtype
        ).to(device)
        pipeline.load_lora_weights("Jovie/Midjourney")
    else:
        model_nf4 = SD3Transformer2DModel.from_pretrained(
            selection,
            subfolder="transformer",
            torch_dtype=torch_dtype
        )

        pipeline = StableDiffusion3Pipeline.from_pretrained(
            selection,
            cache_dir=cache_dir,
            transformer=model_nf4,
            torch_dtype=torch_dtype
        ).to(device)

    pipeline.enable_model_cpu_offload()

    print("Model loaded successfully. Type a prompt to generate an image.\n")

    def generate_filename(prompt, max_words=5):
        words = re.findall(r'\b\w+\b', prompt)[:max_words]
        filename = "_".join(words).lower()
        filename = re.sub(r'[^a-z0-9_]', '', filename)
        return f"generated_{filename}.png" if filename else "generated_image.png"
    default_style = "In the style of Anime, with a high level of detail and vibrant colors."
    while True:
        user_input = input("Enter image prompt (or type 'exit' to quit): ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        full_prompt = f"{user_input}. {default_style}" 
        
        print("Generating image...")
        image = pipeline(full_prompt,
                     num_inference_steps=15, # The higher the number the better quality but slower
                     guidance_scale=7.5, # The higher the value the closer it matches the prompt
                     max_sequence_length=512
        ).images[0]
    
        filename = generate_filename(user_input)
        image_path = os.path.join(os.getcwd(), filename)
        image.save(image_path)
        print(f"Image saved as: {filename}")

else:
    print("Invalid selection. Please restart the program and choose either '1' or '2'.")