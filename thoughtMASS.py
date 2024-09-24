"""
Enhanced Text Semantics Analysis Tool
This tool performs semantic analysis on input text to model the contours of meaningfulness using concepts from an exploratory ontology of thought.
It segments the text into chunks, then quantifies and visualizes metrics including:

Thought Mass - Semantic density calculated using TF-IDF vectors
Local Entropy - Variability of thought mass representing homogenization
Gradients - Changes in thought mass indicating meaningful edges
Similarity - Contextual continuity measured by cosine distance of adjacent chunks
These metrics allow interactively exploring the dynamics of context, novelty, noise reduction, and information contours within the textual ideascape. Users can tune analysis parameters like chunk size, smoothing thresholds, etc.

Advanced features include 3D visualization, Markov boundary detection, and normalized scaling. The tool is intended as a practical demonstration of mapping mindspace.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from matplotlib import gridspec
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import nltk
from nltk.corpus import stopwords
import string

# Download NLTK data if not already present
nltk.download('stopwords', quiet=True)

def read_file(file_path):
    """Reads the contents of a given file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def preprocess_text(text):
    """Preprocess the text by removing punctuation and stopwords."""
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    words = text.lower().split()
    filtered_words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(filtered_words)

def segment_text(text, n=20):
    """Segments the text into chunks of n words each."""
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(0, len(words), n)]

def compute_tfidf(units):
    """Compute the TF-IDF matrix for all units."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(units)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, feature_names, vectorizer

def compute_thought_mass(tfidf_matrix):
    """Compute thought-mass using the sum of TF-IDF values for each unit."""
    return tfidf_matrix.sum(axis=1).A1  # Convert to 1D array

def compute_local_entropy(thought_masses):
    """Compute local entropy based on thought masses."""
    total_mass = np.sum(thought_masses)
    probabilities = thought_masses / total_mass
    # Avoid log(0) by adding a small epsilon where probability is zero
    epsilon = 1e-10
    probabilities = np.where(probabilities == 0, epsilon, probabilities)
    entropy = -probabilities * np.log(probabilities)
    return entropy

def compute_gradient(thought_masses):
    """Compute the gradient of thought-mass for each segment."""
    return np.gradient(thought_masses)

def compute_cosine_similarities(tfidf_matrix):
    """Compute cosine similarity between adjacent units based on their TF-IDF vectors."""
    similarities = []
    for i in range(len(tfidf_matrix) - 1):
        sim = cosine_similarity(tfidf_matrix[i], tfidf_matrix[i + 1])[0][0]
        similarities.append(sim)
    return similarities

def identify_edges(gradients, gradient_threshold=0.5):
    """Identify edges based on gradient thresholds."""
    return gradients > gradient_threshold

def visualize_metrics(units, gradients, entropy, similarities, edges):
    segments = range(len(units))
    
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 0.75])
    fig.suptitle("Analysis of Thought-Mass as an Ontological Primitive", fontsize=16)

    # Plot gradient
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(segments, gradients, marker='o', linestyle='-', color='green', lw=2, label='Gradient')
    ax1.axhline(y=np.mean(gradients), color='gray', linestyle='--', label='Average Gradient')
    for i, edge in enumerate(edges):
        if edge:
            ax1.axvline(x=i, color='red', linestyle='--', alpha=0.5)
    ax1.set_title('Gradient of Thought-Mass: Identifying Shifts in Context')
    ax1.set_ylabel('Gradient Value')
    ax1.legend()

    # Plot entropy
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(segments, entropy, marker='o', linestyle='-', color='blue', lw=2, label='Entropy')
    ax2.axhline(y=np.mean(entropy), color='gray', linestyle='--', label='Average Entropy')
    for i, edge in enumerate(edges):
        if edge:
            ax2.axvline(x=i, color='red', linestyle='--', alpha=0.5)
    ax2.set_title('Local Entropy: Gauge of Information Density')
    ax2.set_ylabel('Entropy Value')
    ax2.legend()

    # Plot cosine similarity
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(segments[:-1], similarities, marker='o', linestyle='-', color='purple', lw=2, label='Cosine Similarity')
    ax3.axhline(y=np.mean(similarities), color='gray', linestyle='--', label='Average Similarity')
    for i, edge in enumerate(edges[:-1]):
        if edge:
            ax3.axvline(x=i, color='red', linestyle='--', alpha=0.5)
    ax3.set_title('Cosine Similarity: Comparing Adjacent Segments for Contextual Continuity')
    ax3.set_xlabel('Segment Index')
    ax3.set_ylabel('Similarity Value')
    ax3.legend()
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.show()

def visualize_3d(entropy, gradient, similarity):
    """3D visualization of entropy, gradient, and similarity."""
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(entropy, gradient, similarity, c=similarity, cmap='viridis',
                         s=200 * similarity, alpha=0.7, edgecolors='w', linewidth=1)
    
    ax.set_xlabel('Entropy', fontsize=14)
    ax.set_ylabel('Gradient', fontsize=14)
    ax.set_zlabel('Cosine Similarity', fontsize=14)
    ax.set_title('3D Analysis of Entropy, Gradient, and Similarity in Thought-Mass', fontsize=16)
    
    cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20, fontsize=12)
    
    ax.grid(True)
    plt.show()

def display_processed_text(units, edges):
    """Display the processed text with suppression and edge indicators."""
    print("\nProcessed Text:\n")
    for i, (segment, edge) in enumerate(zip(units, edges)):
        edge_status = " [EDGE]" if edge else ""
        print(f"Segment {i+1}: {segment}{edge_status}\n")

def main():
    file_path = input("Please enter the path to your text file: ")
    raw_text = read_file(file_path)
    preprocessed_text = preprocess_text(raw_text)
    units = segment_text(preprocessed_text, n=25)
    
    tfidf_matrix, feature_names, vectorizer = compute_tfidf(units)
    thought_masses = compute_thought_mass(tfidf_matrix)
    entropy = compute_local_entropy(thought_masses)
    gradients = compute_gradient(thought_masses)
    similarities = compute_cosine_similarities(tfidf_matrix)
    edges = identify_edges(gradients, gradient_threshold=0.3)
    
    display_processed_text(units, edges)
    print(f"Number of edges identified: {np.sum(edges)}")
    
    visualize_metrics(units, gradients, entropy, similarities, edges)
    visualize_3d(entropy, gradients, similarities)

if __name__ == "__main__":
    main()
