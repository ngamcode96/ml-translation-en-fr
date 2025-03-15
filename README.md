# Training Transformers from Scratch for Language Translation (English to French)

## Overview
This project focuses on training a Transformer model from scratch to perform English-to-French translation. It follows a structured approach, from data collection to model deployment using Gradio.

![Transformer Architecture](https://dassignies.law/wp-content/uploads/2024/04/DASSIGNIES-avocat-intelligence-artificielle-cybersecurite-strategie-protection-actifs-immateriels-formations-expertises-blog-transformer-architecture.webp)

## Project Steps

1. **Data Collection**  
   - Gather parallel English-French text data for training.

2. **Dataset Creation and Upload to Hugging Face**  
   - Preprocess and structure the dataset.
   - Upload the dataset to the Hugging Face Hub for easy access.

3. **Training Tokenizers**  
   - Train separate tokenizers for English and French.
   - Save and store trained tokenizers.

4. **Creating a Tokenized Dataset**  
   - Tokenize the dataset using the trained tokenizers.
   - Publish the tokenized dataset on Hugging Face.

5. **Building the Transformer Model from Scratch**  
   - Implement custom Transformer components, including:
     - Encoder
     - Decoder
     - Embedding Layer
     - Positional Encoding

6. **Model Training and Evaluation**  
   - Train the model using the prepared dataset.
   - Use Weights & Biases (Wandb) for real-time metric visualization.

7. **Inference**  
   - Test the trained model with sample English inputs.
   - Generate translated French text.

8. **Web Interface with Gradio**  
   - Develop an interactive UI using Gradio for easy model inference.

## Installation

To use the application, install the required dependencies using either `uv` or `pip`:

Using `uv`:
```bash
uv pip install -r requirements.txt
```

Using `pip`:
```bash
pip install -r requirements.txt
```

## Running the Application

To launch the application, run:
```bash
python app.py
```

This will start a Gradio interface where users can input English text and receive French translations.

## Repository Structure
- **trained_tokenizers/** - Folder containing trained tokenizers.
- **data_collector.py** - Script for data collection.
- **tokenize_dataset.py** - Prepares and tokenizes dataset.
- **model.py** - Contains the Transformer model implementation.
- **train.py** - Training script.
- **inference.py** - Inference script for model predictions.
- **app.py** - Web interface with Gradio.
- **requirements.txt** - List of dependencies.
