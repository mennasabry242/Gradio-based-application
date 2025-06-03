# 🧠 Gradio-based Image Description & Code Generation App

This project provides a simple web interface using **Gradio** that performs two main tasks:

1. **Image Description** – Generate captions for uploaded images.
2. **Python Code Generation** – Generate Python code based on natural language task descriptions.

The app integrates two powerful transformer models:
- 🔍 **Image Captioning Model**: [`llava-hf/llava-1.5-7b-hf`](https://huggingface.co/llava-hf/llava-1.5-7b-hf)
- 👨‍💻 **Code Generation Model**: [`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)

---

## 🚀 How to Run

1. Open this notebook or script in **Google Colab** (T4 GPU recommended).
2. Install required libraries:

```bash
!pip install gradio transformers torch Pillow
##🧪 Prompt Engineering Strategy
We used task-specific prompts to improve accuracy and relevance:

**Image Description**
Prompt used:

text
Copy
Edit
<image>
Describe the image.
This format works well with the LLaVA model and keeps the image context clear.

⚠️** Warning**: Avoid using the Qwen-VL-Chat model for image captioning on Google Colab.
It’s very slow, consumes too much GPU, and often times out in limited environments like Colab.
llava-hf/llava-1.5-7b-hf is much more suitable and optimized.

##👨‍💻 Code Generation
We use this consistent format to ensure the model outputs pure Python code only:

text
Copy
Edit
Write only the Python function that solves the following task. 
Do not add comments, explanations, or markdown.

Task: {your description here}

Code:
This minimalistic prompt:

Keeps the output focused and clean.

Avoids markdown or explanations that interfere with functional code output.

🔍 I tried using more detailed prompt templates to give extra instructions and structure,
but it confused the model and led to irrelevant or broken outputs.
What worked best was keeping the prompt balanced — not too vague, not too specific.

✅ A great tool to explore is FRESCA(Format, Refine, , which helps Write prompts can lead to better results 
##🎛️ Parameter Tuning Tips
The interface provides sliders to experiment with model behavior. Here are recommended ranges based on testing:

**Parameter	Purpose	Recommended Values**
temperature	Controls randomness. Lower values = more focused output.	0.6 – 0.7
top_p	Nucleus sampling. Filters to top p cumulative probability tokens.	0.8 – 0.95
max_tokens	Limits the number of tokens in the output (i.e., output length).	100 – 200

📌 **Note**: Increasing temperature too much often leads to noisy output — especially for code.
For clean, executable code, keep temperature below 0.75.

##🖼️ User Interface
You can select one or both tasks:

Upload an image for description.

Write a natural language task prompt for code generation.

Adjust generation parameters.

Click Generate and get results instantly.
