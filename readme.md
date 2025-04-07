# Sentiment Analysis App

## Overview
This application uses natural language processing to analyze the emotional tone (sentiment) of text input. Built with Python, it leverages Hugging Face's Transformers library for sentiment analysis and Gradio for the interactive web interface.

![Sentiment Analysis App](https://via.placeholder.com/800x450?text=Sentiment+Analysis+App)

## Features
- Real-time sentiment analysis of text input
- Visual confidence score representation
- Pre-loaded example statements to demonstrate different sentiments
- Clean, intuitive user interface
- Support for positive, negative, and neutral sentiment detection

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package installer)

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/sentiment-analysis-app.git
cd sentiment-analysis-app
```

### Step 2: Install dependencies
```bash
pip install --upgrade gradio transformers torch matplotlib
```

### Step 3: Run the application
```bash
python app.py
```

After running the command, the application will be available at `http://localhost:7860` in your web browser.

## Usage

1. Enter text in the input box or select one of the provided examples
2. Click "Analyze Sentiment" or press Enter
3. View the sentiment analysis results and confidence score visualization
4. Use the "Clear" button to reset the input and results

## How It Works

The application uses a pre-trained DistilBERT model fine-tuned on the SST-2 (Stanford Sentiment Treebank) dataset to classify text as either POSITIVE or NEGATIVE. The model provides a confidence score that indicates how certain it is about its prediction.

The sentiment is displayed along with an appropriate emoji:
- 😄 for positive sentiment
- 😞 for negative sentiment
- 😐 for neutral sentiment (if applicable)

## Customization

You can customize the application by:

1. Modifying the example statements in the `EXAMPLE_STATEMENTS` list
2. Changing the color scheme in the `create_confidence_viz` function
3. Adding additional visualizations or analysis metrics

## Troubleshooting

### Common Issues:

1. **ModuleNotFoundError**: Ensure all dependencies are installed
   ```bash
   pip install --upgrade gradio transformers torch matplotlib
   ```

2. **CUDA/GPU Issues**: If you encounter GPU-related errors, you can force CPU usage:
   ```python
   sentiment_pipeline = pipeline("sentiment-analysis", device=-1)  # Forces CPU
   ```

3. **Slow First Analysis**: The first analysis might take longer as the model is loaded into memory

## Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/transformers/)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [DistilBERT Paper](https://arxiv.org/abs/1910.01108)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face for providing the pre-trained models
- Gradio team for the interactive UI framework