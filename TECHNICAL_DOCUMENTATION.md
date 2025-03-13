# Social Media Analysis API Technical Documentation

## Architecture Overview

The application is built using FastAPI and implements a RESTful API for analyzing social media content, particularly Reddit data. The system combines natural language processing, sentiment analysis, and vector similarity search to provide comprehensive social media analytics.

## Core Components

### 1. Data Processing
- Uses Pandas DataFrame for efficient data manipulation
- Loads data from JSONL format
- Implements text cleaning and preprocessing using regex
- Handles missing values and data type conversions

### 2. Machine Learning Models
- **Sentiment Analysis**: Uses Hugging Face's transformers pipeline
- **Chat Bot**: Uses Vector store for Chatbot database
- **Text Embeddings**: Implements SentenceTransformer for semantic search
- **Vector Store**: Uses FAISS for efficient similarity search
- **Topic Modeling**: Implements Latent Dirichlet Allocation (LDA)

### 3. API Endpoints

#### Chat Functionality
- Implements context-aware chatbot using Groq API
- Uses vector similarity search for relevant context retrieval
- Handles SSL verification and API error cases

#### Sentiment Analysis
- Batch processing with configurable batch sizes
- Provides sentiment distribution statistics
- Includes confidence scores for sentiment predictions
- Handles both individual and bulk sentiment analysis

#### Content Analysis
- Implements text summarization
- Provides word frequency analysis
- Generates topic models
- Tracks trending topics over time

#### Subreddit Analysis
- Provides subreddit-specific metrics
- Implements sentiment analysis per subreddit
- Generates subreddit-specific summaries
- Analyzes top words and trends

## Technical Implementation Details

### Vector Store Implementation
- Uses FAISS IndexFlatL2 for L2 distance-based similarity search
- Dimension: 384 (based on SentenceTransformer model)
- Implements batch processing for efficient indexing

### Text Processing Pipeline
1. URL removal
2. Special character handling
3. Hashtag and mention preservation
4. Tokenization with length constraints
5. Text summarization for long content

### Performance Optimizations
- Implements vectorized operations using Pandas
- Uses batch processing for model inference
- Implements efficient data structures for quick lookups
- Caches processed data where appropriate

### Error Handling
- Implements comprehensive exception handling
- Provides detailed error messages
- Handles edge cases in data processing
- Implements proper HTTP status codes

### Security Features
- Implements CORS middleware
- Handles SSL verification
- Secures API keys using environment variables
- Implements proper input validation

## Data Flow

1. **Input Processing**
   - Data loading from JSONL
   - Text cleaning and preprocessing
   - Vector embedding generation

2. **Analysis Pipeline**
   - Sentiment analysis
   - Topic modeling
   - Trend analysis
   - Summary generation

3. **Response Generation**
   - Data aggregation
   - Statistical analysis
   - JSON response formatting

## Dependencies

- FastAPI: Web framework
- Pandas: Data manipulation
- Transformers: NLP models
- SentenceTransformer: Text embeddings
- FAISS: Vector similarity search
- scikit-learn: Machine learning utilities
- aiohttp: Async HTTP client
- python-dotenv: Environment variable management

## Performance Considerations

- Batch processing for large datasets
- Efficient vector indexing
- Memory management for large text processing
- Async operations for API calls

## Future Improvements

1. Implement caching for frequently accessed data
2. Add rate limiting
3. Implement more sophisticated text summarization
4. Add more advanced topic modeling
5. Implement real-time data processing
6. Add user authentication and authorization
7. Implement data persistence layer
8. Add more sophisticated error handling
9. Implement logging and monitoring
10. Add automated testing
