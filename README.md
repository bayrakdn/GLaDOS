# GLaDOS: A Chatbot Powered by LLaMA 2

GLaDOS is an advanced chatbot implementation built using the LLaMA 2 model for natural language processing and conversation. This project is developed in Google Colab and allows users to interact with a custom-trained language model. It also supports querying information from uploaded PDF files.

---

## Features
- **Natural Language Interaction**: GLaDOS responds to user queries conversationally.
- **Document Analysis**: Upload PDF files and ask questions based on their content.
- **High-Performance AI Model**: Utilizes the LLaMA 2 model for robust and precise language understanding.
- **Customizable Prompts**: Adjust system and query prompts for varied use cases.

---

## Installation

### Prerequisites
1. Google Colab or a Python environment with GPU support.
2. Hugging Face account and an access token for LLaMA 2.

### Steps
1. Clone the repository:
    ```bash
    git clone https://github.com/YourUsername/GLaDOS-LLaMA2.git
    cd GLaDOS-LLaMA2
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Configure your Hugging Face token in the script:
    ```python
    auth_token = "Your Own Unique Token Right Here"
    ```

4. Run the script in a Google Colab notebook or any compatible environment.

---

## Usage

### Setting Up GLaDOS
1. Import required modules and install dependencies using the script.
2. Load the LLaMA 2 model:
    ```python
    name = "meta-llama/Llama-2-13b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir='./model/', use_auth_token=auth_token)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        cache_dir='./model/',
        use_auth_token=auth_token,
        torch_dtype=torch.float16,
        rope_scaling={"type": "dynamic", "factor": 2},
        load_in_8bit=True
    )
    ```

### Interacting with the Chatbot
Input your query in the `user_query` variable, and the bot will generate a response:
```python
user_query = "### User: What does BBC stands for? ### GLaDOS: "

Upload a PDF file and analyze its contents using the;
file_path = '/content/A.pdf'
load_pdf(file_path)

Ask questions based on the uploaded document:
response = query_engine.query("What was the FY2022 return on equity?")

Acknowledgments
Special thanks to:

Meta for the LLaMA 2 model.
Hugging Face for providing an excellent model repository.
Google Colab for its free GPU compute platform.
