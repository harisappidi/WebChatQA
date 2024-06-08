# WebChatQA

## Description

WebChatQA is a Streamlit-based project that allows users to interact with a web application by providing a URL. The provided URL is processed to create vector embeddings for efficient question answering, enabling users to extract relevant information from web content seamlessly.

This project leverages the LangChain framework to handle the text processing and question answering tasks. The content from the provided URL is processed and converted into smaller text chunks using LangChain's `RecursiveCharacterTextSplitter`. These chunks are then transformed into vector embeddings using the HuggingFace BAAI/bge-small-en model. The vector embeddings are stored and managed using FAISS (Facebook AI Similarity Search) for efficient retrieval and similarity search.

## Installation

Follow these steps to set up the project:

1. **Clone the repository:**

   ```sh
   git clone https://github.com/yourusername/WebChatQA.git
   cd WebChatQA
   ```
2. **Create a virtual environment:**

   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install the required dependencies:**

   ```sh
   pip install -r requirements.txt
   ```
4. **Set up environment variables:**
   Create a `.env` file in the project root directory and add your HuggingFace API token:

   ```env
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
   ```

## Usage

1. **Run the Streamlit application:**

   ```sh
   streamlit run app.py
   ```
2. **Provide a URL:**
   Enter the URL of the web page you want to process in the sidebar.
3. **Ask questions:**
   After processing the URL, you can enter questions in the input box to retrieve relevant answers based on the web content.

## Contributing

We welcome contributions to improve WebChatQA! To contribute:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
   ```sh
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes.
   ```sh
   git commit -m "Add your commit message"
   ```
4. Push to the branch.
   ```sh
   git push origin feature/your-feature-name
   ```
5. Open a pull request describing your changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Contact

For any inquiries or issues, please contact the project maintainer at:

- **Email:** harikrishna.sappidi@gmail.com
- **GitHub:** [harisappidi](https://github.com/harisappidi)

I look forward to your feedback and contributions!
