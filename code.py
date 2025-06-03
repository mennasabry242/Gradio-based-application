# Import required libraries
import gradio as gr  # For building the web interface
import torch  # For model loading and GPU support
from PIL import Image  # For handling image input
from transformers import AutoProcessor, LlavaForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM  # For model loading and processing

# ----------- CONFIG -----------

# Model IDs for image captioning and code generation
caption_model_id = "llava-hf/llava-1.5-7b-hf"  # Model used for generating image descriptions
code_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  # Model used for code generation

# Load tokenizer for the code generation model
tokenizer = AutoTokenizer.from_pretrained(code_model_id, trust_remote_code=True)

# ----------- IMAGE DESCRIPTION FUNCTION -----------

def describe_image(image):
    """
    Generates a textual description for the given image using the LLaVA model.
    """
    try:
        # Load processor and model for captioning
        processor = AutoProcessor.from_pretrained(caption_model_id)
        model = LlavaForConditionalGeneration.from_pretrained(
            caption_model_id, torch_dtype=torch.float16, device_map="auto"
        ).eval()

        # Define the prompt for image description
        prompt = "<image>\nDescribe the image."

        # Convert the image to RGB format
        image = image.convert("RGB")

        # Preprocess the image and prompt
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

        # Generate output from the model
        output = model.generate(**inputs, max_new_tokens=100)

        # Decode the generated tokens to text
        caption = processor.batch_decode(output, skip_special_tokens=True)[0]

        # Clean up to free GPU memory
        del model
        torch.cuda.empty_cache()

        return caption
    except Exception as e:
        torch.cuda.empty_cache()
        return f"Error generating description: {e}"

# ----------- CODE GENERATION FUNCTION -----------

def generate_code_only(task_description, temperature=0.7, top_p=0.9, max_tokens=150):
    """
    Generates Python code based on the provided natural language task description.
    """
    try:
        # Load the code generation model
        model = AutoModelForCausalLM.from_pretrained(
            code_model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        ).eval()

        # Prepare prompt to instruct the model to generate code only
        prompt = (
            f"Write only the Python function that solves the following task. "
            f"Do not add comments, explanations, or markdown.\n\n"
            f"Task: {task_description}\n\n"
            f"Code:\n"
        )

        # Tokenize the prompt and move to device
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate code using sampling with given parameters
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True
        )

        # Decode the output and extract the code part
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Clean up model from memory
        del model
        torch.cuda.empty_cache()

        return text.split("Code:")[-1].strip() if "Code:" in text else text.strip()
    except Exception as e:
        torch.cuda.empty_cache()
        return f"Error generating code: {e}"

# ----------- COMBINED FUNCTION -----------

def generate_both(selected, image, task_prompt, temperature, top_p, max_tokens):
    """
    Determines which tasks are selected (image description or code generation),
    then executes the appropriate functions.
    """
    caption, code = "", ""

    if "Image Description" in selected and image:
        caption = describe_image(image)

    if "Python Code" in selected and task_prompt.strip():
        code = generate_code_only(task_prompt, temperature, top_p, max_tokens)

    return caption, code

# ----------- GRADIO INTERFACE -----------

with gr.Blocks() as demo:
    gr.Markdown("## 🔄 Unified Image Description & Code Generator")

    # Task selection options
    with gr.Row():
        task_selector = gr.CheckboxGroup(
            choices=["Image Description", "Python Code"],
            label="Select Tasks",
            value=["Image Description"]
        )

    # Input fields: image and text prompt
    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        task_prompt = gr.Textbox(
            label="Task Prompt (for code generation)",
            lines=2,
            placeholder="E.g., Write a function to reverse a string."
        )

    # Generation parameter controls
    with gr.Row():
        temperature = gr.Slider(0.1, 1.0, value=0.7, step=0.05, label="Temperature")
        top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
        max_tokens = gr.Slider(50, 300, value=150, step=10, label="Max Tokens")

    # Button to run generation
    with gr.Row():
        run_button = gr.Button("Generate")

    # Output fields for image caption and generated code
    with gr.Row():
        caption_output = gr.Textbox(label="Image Description", lines=5)
        code_output = gr.Textbox(label="Generated Python Code", lines=15)

    # Bind the button to the combined function
    run_button.click(
        fn=generate_both,
        inputs=[task_selector, image_input, task_prompt, temperature, top_p, max_tokens],
        outputs=[caption_output, code_output]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch()
