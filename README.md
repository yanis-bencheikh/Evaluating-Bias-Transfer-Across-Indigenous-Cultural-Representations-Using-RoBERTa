# Parliamentary Language Bias Analysis: A Comparative Study of Canadian and Nunavut Hansard Data

## 📋 Project Overview

This project investigates potential biases in language models fine-tuned on parliamentary proceedings from Canada and Nunavut. By comparing embedding spaces and language representations across different training datasets, we analyze how political discourse shapes AI model understanding of Indigenous-related concepts, governance, and social issues.

### 🎯 Research Questions
- How do models trained on different parliamentary datasets represent Indigenous-related concepts?
- What biases emerge when models are trained on predominantly settler vs. Indigenous political discourse?
- How does balanced vs. imbalanced training data affect model representations?

## 🗂️ Project Structure

```
COMP550_final_project/
├── data/
│   ├── canadian_hansard/          # Canadian parliamentary proceedings (1999-2017)
│   ├── nunavut_hansard/           # Nunavut parliamentary proceedings
│   └── untarred/                  # Extracted archive files
├── models/                        # Fine-tuned RoBERTa models
│   ├── canadian_hansard/          # Model trained on Canadian data only
│   ├── nunavut_hansard/           # Model trained on Nunavut data only
│   ├── balanced_multilingual/     # 50/50 balanced training
│   ├── imbalanced_multilingual/   # 80/20 Canadian/Nunavut ratio
│   ├── canadian_hansard_mlm/      # MLM version for Canadian data
│   ├── nunavut_hansard_mlm/       # MLM version for Nunavut data
│   ├── balanced_multilingual_mlm/ # MLM version for balanced training
│   └── imbalanced_multilingual_mlm/ # MLM version for imbalanced training
├── tokenized_datasets/            # Preprocessed and tokenized data
├── tokenized_mlm_datasets/        # MLM-specific tokenized data
├── results/                       # Analysis outputs and visualizations
│   ├── bias_analysis_results.csv
│   ├── full_vocab_ranking_results.csv
│   ├── open_ended_mlm_results.csv
│   └── binary_mlm_results.csv
└── data/                          # Raw and benchmark datasets
    ├── validated_binary_benchmark_nunavut.csv
    └── indigenous_political_bias_benchmark_comp550.csv
```

## 🛠️ Methodology

### Data Sources
1. **Canadian Hansard (1999-2017)**: Parliamentary proceedings from the House of Commons
2. **Nunavut Hansard**: Inuktitut-English parallel corpus from Nunavut Legislative Assembly

### Model Training Approaches
- **Baseline**: Pre-trained RoBERTa-base
- **Canadian-only**: Fine-tuned exclusively on Canadian parliamentary data
- **Nunavut-only**: Fine-tuned exclusively on Nunavut parliamentary data
- **Balanced Multilingual**: 50/50 mix of Canadian and Nunavut data
- **Imbalanced Multilingual**: 80/20 Canadian/Nunavut ratio

### Analysis Methods

#### 1. **Embedding Space Analysis**
- UMAP visualization of politically charged concepts
- Cosine similarity matrices for key terms
- Clustering analysis of embedding representations

#### 2. **Word Association Bias Testing**
- Word Embedding Association Test (WEAT) scores
- Pairwise cosine similarity analysis
- Full vocabulary ranking for target concepts

#### 3. **Masked Language Modeling (MLM) Benchmarking**
- **Open-ended MLM**: Models predict masked tokens in Indigenous-related contexts
- **Binary MLM**: Models choose between favorable/unfavorable completions
- Custom benchmark dataset with validated Indigenous political bias prompts
- Comparative probability analysis across training approaches

## 🔧 Technical Implementation

### Dependencies
```python
transformers==4.x
torch>=1.9.0
datasets
pandas
numpy
scikit-learn
umap-learn
matplotlib
seaborn
wordcloud
```

### Key Components

#### Data Processing Pipeline
```python
# Extract and preprocess parliamentary data
canadian_dataset = load_dataset_from_csv(canadian_file, "speechtext")
nunavut_dataset = load_dataset_from_txt(nunavut_file)

# Create balanced/imbalanced combinations
balanced_dataset = create_combined_dataset(canadian, nunavut, ratio=0.5)
imbalanced_dataset = create_combined_dataset(canadian, nunavut, ratio=0.8)
```

#### Model Fine-tuning
```python
# Fine-tune RoBERTa for sequence classification
model = RobertaForSequenceClassification.from_pretrained("roberta-base")
trainer = Trainer(model=model, train_dataset=dataset, ...)
trainer.train()
```

#### Bias Analysis
```python
# WEAT score computation
def compute_weat_score(target_X, target_Y, attribute_A, attribute_B):
    # Association-based bias measurement
    return s_X - s_Y

# Word pair similarity analysis
cos_sim = cosine_similarity([embedding1], [embedding2])

# MLM benchmarking
def evaluate_binary_mlm(benchmark_data, model_paths):
    # Compare favorable vs unfavorable completions
    favorable_prob = probs[0, favorable_id].item()
    unfavorable_prob = probs[0, unfavorable_id].item()
```

## 📊 Key Findings

### Embedding Space Differences
- Models show distinct clustering patterns for Indigenous vs. settler concepts
- Nunavut-trained models demonstrate different associative patterns
- Balanced training reduces some systematic biases

### Word Association Patterns
- Significant variations in how models associate Indigenous groups with positive/negative concepts
- Canadian-only models show different bias patterns compared to Nunavut-inclusive models
- WEAT scores reveal measurable bias differences across training approaches

### Vocabulary Ranking Insights
- Models prioritize different semantic relationships based on training data
- Indigenous concepts show varying proximity to governance vs. social issue terms
- Multilingual training affects conceptual clustering

### MLM Benchmarking Results
- **Binary choice patterns**: Models show distinct preferences for favorable vs unfavorable completions
- **Cross-model consistency**: Different training approaches lead to measurable differences in MLM predictions
- **Probability distributions**: Quantitative assessment of bias through completion probabilities
- **Theme-based analysis**: Systematic evaluation across different Indigenous-related topics

## 🚀 Usage

### 1. Data Preparation
```bash
# Extract parliamentary data archives
python extract_hansard_data.py

# Preprocess and align datasets
python preprocess_data.py
```

### 2. Model Training
```bash
# Train individual models
python fine_tune_models.py --dataset canadian
python fine_tune_models.py --dataset nunavut
python fine_tune_models.py --dataset balanced
python fine_tune_models.py --dataset imbalanced
```

### 3. Bias Analysis
```bash
# Run embedding analysis
python analyze_embeddings.py

# Compute WEAT scores
python compute_weat_scores.py

# Generate vocabulary rankings
python rank_vocabulary.py

# MLM benchmarking evaluation
python benchmark_mlm_models.py
```

## 📈 Results and Visualization

The project generates several types of analysis outputs:

- **UMAP plots**: 2D visualizations of embedding spaces
- **Similarity matrices**: Heatmaps showing cosine similarities
- **Bias score tables**: Quantitative measurements of associative biases
- **Word clouds**: Visual representations of vocabulary emphasis
- **Ranking tables**: Closest word associations for target concepts
- **MLM probability charts**: Comparative analysis of favorable vs unfavorable completions
- **Cross-model consistency plots**: Tracking prediction patterns across training approaches

## 🔍 Ethical Considerations

This research addresses important questions about:
- **Representation bias** in AI systems trained on political discourse
- **Indigenous voice preservation** in computational models
- **Balanced dataset creation** for more equitable AI
- **Transparency** in bias detection and measurement

## 📚 Academic Context

This work contributes to:
- Computational linguistics bias detection
- Indigenous NLP research
- Political discourse analysis
- Ethical AI development

## 🤝 Contributing

This project was developed for COMP 550 (Natural Language Processing) and explores critical intersections of AI, politics, and Indigenous representation. The methodologies can be adapted for other multilingual or cross-cultural bias studies.

## 📄 License

Research project for academic purposes. Data sources subject to their respective licensing terms.

## 🙏 Acknowledgments

- Canadian Parliament for Hansard data availability
- Nunavut Legislative Assembly for parallel corpus access
- COMP 550 course instruction and guidance
- Indigenous communities whose voices are represented in this data

---

*This project aims to contribute to more equitable and representative AI systems by understanding and measuring bias in parliamentary language models.*
