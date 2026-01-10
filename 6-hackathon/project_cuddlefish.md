# Kadaster Akte Classifier: AI-Powered Legal Document Classification

Our hackathon project aimed to solve a real-world problem: how can we automatically classify Dutch legal documents (aktes) from the Kadaster with their appropriate "rechtsfeitcodes" (legal fact codes)?

## The Goal

The primary objective was to build a machine learning model that could analyze anonymized legal documents and predict the appropriate legal fact codes from 200+ possible categories. This is a challenging multi-label classification task with a highly imbalanced dataset of approximately 20,000 documents.

## The Model

We developed a hybrid approach combining three complementary techniques:

1. **Regex-based Classification**: Automatically generated regex patterns from code descriptions, with domain-specific improvements including bidirectional logic for legal language patterns and smart compounding for compound words.

2. **Text Vectorization with Transformers**: Used pre-trained HuggingFace models (like BERT) with automatic pooling strategy detection and dynamic max-length configuration to create semantic embeddings of the legal texts.

3. **Hybrid Neural Architecture**: Built a sophisticated dual-pathway transformer architecture that processes:
   - Text embeddings through a Transformer encoder
   - Regex features through a separate embedding layer and Transformer encoder
   - Combined representations through a final MLP classifier

The model evolved throughout the hackathon, progressing from simple regex matching → regex + neural network → text vectorizer + neural network → hybrid transformer architecture combining both modalities.

## Key Technical Achievements

- **Smart Vectorization Caching**: Implemented hash-based caching system to speed up repeated training runs
- **Automatic Pooling Detection**: Built intelligent model configuration detection from HuggingFace models
- **Separate Neural Pathways**: Architected independent processing streams for regex and text features, allowing the model to leverage both rule-based and learned patterns
- **Configurable Training Pipeline**: Created flexible CLI with MLflow tracking, early stopping, and comprehensive evaluation metrics
- **Long-tail Handling**: Designed strategies for dealing with rare classes through threshold-based LLM fallback

## Outcome

The hybrid model successfully combines the precision of domain-specific regex patterns with the semantic understanding of transformer-based text embeddings. The separate transformer encoders for text and regex features allow the model to learn complex interactions between rule-based and contextual information, which is particularly effective for legal document classification where both explicit keywords and contextual understanding matter.

You can find all the code in https://github.com/dhussaini/Hackathon

[Go back to Homepage](../README.md)