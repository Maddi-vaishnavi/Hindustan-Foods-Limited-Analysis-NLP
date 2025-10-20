# 📊 Hindustan Foods Limited – Report Analyzer

An interactive NLP dashboard for analyzing annual reports using sentiment analysis, word clouds, TF-IDF, and topic modeling (LDA).

---

## 🎯 Project Overview

This Streamlit web application performs comprehensive Natural Language Processing on PDF annual reports, providing:

- ✅ **PDF Text Extraction** - Extract text from all pages
- ✅ **Sentiment Analysis** - Analyze sentiment for each sentence using TextBlob
- ✅ **Word Cloud Generation** - Visual representation of frequent words
- ✅ **TF-IDF Matrix** - Term Frequency-Inverse Document Frequency analysis
- ✅ **Topic Modeling** - Discover 10 latent topics using LDA (Latent Dirichlet Allocation)

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+ installed
- pip package manager

### Installation

1. **Clone or download this project**

2. **Install dependencies:**
```powershell
pip install streamlit PyPDF2==3.0.1 nltk textblob wordcloud matplotlib scikit-learn pandas
```

3. **Download NLTK data** (one-time setup):
```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Running the Application

```powershell
# Navigate to project directory
cd "d:\B.tech{CS-DS}\Semister 7\DE-IV(Speech and NLP)\project"

# Run Streamlit app
streamlit run assignment.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📋 How to Use

1. **Upload PDF**: Click "Upload PDF file" in the left sidebar
2. **View Analysis**: Navigate through the 4 tabs:
   - **Tab 1**: Sentiment Analysis with charts
   - **Tab 2**: Frequent Words & Word Cloud
   - **Tab 3**: TF-IDF Matrix Preview
   - **Tab 4**: Topic Modeling (10 Topics)
3. **Download Results**: Click download buttons to export CSV files

---

## 🛠️ Technical Stack

| Component | Technology |
|-----------|-----------|
| **Framework** | Streamlit |
| **PDF Processing** | PyPDF2 |
| **NLP** | NLTK, TextBlob |
| **Machine Learning** | Scikit-learn (TF-IDF, LDA) |
| **Visualization** | Matplotlib, WordCloud |
| **Data Manipulation** | Pandas |

---

## 📊 Features

### 1. Sentiment Analysis
- Sentence-level sentiment classification (Positive/Negative/Neutral)
- Polarity scores (-1 to +1)
- Visual distribution charts (bar & pie)
- Downloadable CSV results

### 2. Word Analysis
- Top 20 most frequent words
- Frequency bar charts
- Beautiful word clouds with color schemes

### 3. TF-IDF Analysis
- Document-term matrix creation
- Unigrams and bigrams support
- Feature importance visualization

### 4. Topic Modeling (LDA)
- 10 topics with top 10 words each
- Topic weight distribution
- Gibbs sampling-based (online learning)
- Visual topic importance chart

---

## 📁 Project Structure

```
project/
├── assignment.py          # Main Streamlit application
├── PROJECT_REPORT.md      # Detailed project report
├── README.md              # This file
└── requirements.txt       # Python dependencies (optional)
```

---

## 🎓 Project Requirements (Completed)

- [x] **Task 1**: Import PDF and read all pages
- [x] **Task 2**: Save into DataFrame
- [x] **Task 3**: Preprocess (lowercase, remove punctuation, digits, special chars, stopwords)
- [x] **Task 4**: Sentence tokenize and calculate sentiment (TextBlob)
- [x] **Task 5**: Word tokenize and preprocess
- [x] **Task 6**: Frequent words and word cloud
- [x] **Task 7**: Convert to TF-IDF / Document-Term Matrix
- [x] **Task 8**: Build Topic Modeling (LDA with 10 topics)

---

## 📈 Sample Output

### Sentiment Distribution
```
Positive: 45.6%
Neutral:  41.5%
Negative: 12.9%
```

### Top Words
```
1. company    (87)
2. revenue    (65)
3. growth     (52)
4. market     (48)
5. operations (43)
```

### Sample Topics
```
Topic 1: financial, performance, revenue, growth, profit
Topic 2: operations, manufacturing, production, facilities
Topic 3: market, competition, strategy, expansion
...
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'PyPDF2'"
**Solution:**
```powershell
pip install PyPDF2==3.0.1
```

### Issue: NLTK data not found
**Solution:**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

### Issue: PDF upload error
**Solution:** Ensure PDF is text-based (not scanned images)

---

## 🔮 Future Enhancements

- [ ] Named Entity Recognition (NER)
- [ ] Multi-report comparison
- [ ] Time-series sentiment tracking
- [ ] OCR for scanned PDFs
- [ ] Export to PowerPoint/PDF
- [ ] Advanced visualizations (Plotly)
- [ ] Fine-tuned BERT models

---

## 📚 References

- **LDA Paper**: Blei et al. (2003) - "Latent Dirichlet Allocation"
- **Streamlit Docs**: https://docs.streamlit.io/
- **NLTK Book**: "Natural Language Processing with Python"
- **Scikit-learn**: https://scikit-learn.org/

---

## 👨‍💻 Course Information

**Course:** DE-IV (Speech and Natural Language Processing)  
**Semester:** 7 | B.Tech (CS-DS)  
**Institution:** [Your Institution]  
**Date:** October 20, 2025

---

## 📄 License

This project is created for educational purposes as part of the NLP course curriculum.

---

## 🤝 Contributing

This is an academic project. For suggestions or improvements, please contact the course instructor.

---

**Built with ❤️ using Python & Streamlit**
