import gradio as gr
from transformers import pipeline
import matplotlib.pyplot as plt
import time

# Load sentiment-analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis")

# Example statements for users to try
EXAMPLE_STATEMENTS = [
    "I absolutely loved the movie, it was fantastic!",
    "The service at this restaurant was terrible and the food was cold.",
    "I'm feeling neutral about the whole situation.",
    "The new software update has some bugs, but overall it's an improvement.",
    "This is the worst experience I've ever had with customer service."
]

# Simple prediction function
def analyze_sentiment(text):
    if not text or not text.strip():
        return "Please enter some text to analyze.", None
    
    # Get prediction from model
    try:
        result = sentiment_pipeline(text)[0]
        label = result['label']
        score = result['score']
        
        # Create emoji based on sentiment
        emoji = "😄" if label == "POSITIVE" else "😞" if label == "NEGATIVE" else "😐"
        
        # Format the output text
        output_text = f"Sentiment: {label} {emoji}\nConfidence: {score:.4f} ({score:.2%})"
        
        # Create visualization
        fig = create_confidence_viz(score, label)
        
        return output_text, fig
    except Exception as e:
        return f"Error during analysis: {str(e)}", None

# Create visualization for confidence score
def create_confidence_viz(score, label):
    # Set color based on sentiment
    color = "#28a745" if label == "POSITIVE" else "#dc3545" if label == "NEGATIVE" else "#6c757d"
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 1))
    
    # Create horizontal bar
    ax.barh([0], [score], color=color, height=0.4)
    ax.barh([0], [1], color="#e9ecef", height=0.4, alpha=0.5)
    
    # Add confidence percentage text
    ax.text(score/2, 0, f"{score:.2%}", 
            ha='center', va='center', color='white', fontweight='bold')
    
    # Remove axes and spines
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # Set limits
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    return fig

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 💬 Sentiment Analysis App")
    gr.Markdown("This app analyzes the emotional tone of your text using a pre-trained language model.")
    
    with gr.Row():
        with gr.Column():
            # Input text area
            input_text = gr.Textbox(
                label="Enter your text",
                placeholder="Type or paste text here to analyze its sentiment...",
                lines=5
            )
            
            # Buttons
            with gr.Row():
                clear_btn = gr.Button("Clear")
                analyze_btn = gr.Button("Analyze Sentiment", variant="primary")
            
            # Examples
            gr.Markdown("### Try these examples:")
            examples = gr.Examples(
                examples=EXAMPLE_STATEMENTS,
                inputs=input_text
            )
        
        with gr.Column():
            # Output area
            result_text = gr.Textbox(label="Analysis Result", lines=3)
            confidence_plot = gr.Plot(label="Confidence Score")
    
    # Set up event handlers
    analyze_btn.click(
        fn=analyze_sentiment,
        inputs=input_text,
        outputs=[result_text, confidence_plot]
    )
    
    clear_btn.click(
        fn=lambda: ("", None),
        inputs=[],
        outputs=[result_text, confidence_plot]
    )
    
    input_text.submit(
        fn=analyze_sentiment,
        inputs=input_text,
        outputs=[result_text, confidence_plot]
    )
    
    # Add a clear function for the input text
    clear_btn.click(
        fn=lambda: "",
        inputs=[],
        outputs=[input_text]
    )
    
    gr.Markdown(
        """<div style="margin-top: 20px; text-align: center; font-size: 0.8rem; color: #666;">
        Powered by Hugging Face Transformers and Gradio
        </div>"""
    )

# Launch the app
if __name__ == "__main__":
    app.launch()