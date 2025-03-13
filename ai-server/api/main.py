from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Query
from typing import List, Dict, Any, Optional
import pandas as pd
import json
import re
from transformers import pipeline, AutoTokenizer
import os
import aiohttp
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from pydantic import BaseModel
import ssl
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_message: str
    context: List[str] = []

# Update the API URL and key configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"  # Updated API endpoint

# Update the call_groq_api function
async def call_groq_api(prompt: str) -> str:
    """Calls the Groq API with the prompt."""
    if not GROQ_API_KEY:
        return "Error: GROQ_API_KEY not found in environment variables"
        
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "You are a chatbot specialized in analyzing and discussing social media trends and Reddit content."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1024
    }

    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            
            async with session.post(GROQ_API_URL, headers=headers, json=payload) as resp:
                response_json = await resp.json()
                
                if "error" in response_json:
                    return f"API Error: {response_json['error'].get('message', 'Unknown error')}"
                
                return response_json.get("choices", [{}])[0].get("message", {}).get("content", "No response content available.")
    except Exception as e:
        return f"Error calling Groq API: {str(e)}"

# Load sentiment analysis model
# Load models and initialize them before use
try:
    sentiment_model = pipeline("sentiment-analysis")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Move this section up, after the imports and before build_vector_store
def clean_text(text: str) -> str:
    """Cleans text by removing URLs and special characters while preserving hashtags and mentions."""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z0-9#@\s]", "", text)  # Preserve hashtags & mentions
    return text.strip()

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def truncate_text(text):
    tokenized_input = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    return tokenizer.decode(tokenized_input["input_ids"][0], skip_special_tokens=True)

# Load data from JSONL file
def load_data(file_path: str) -> pd.DataFrame:
    data = []
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = [json.loads(line) for line in file]
        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame([item.get('data', {}) for item in data])
        return df
    except Exception as e:
        raise RuntimeError(f"Error loading data: {e}")

# Load the data before building vector store
df = load_data("./data.jsonl")

# Build Vector Store (FAISS)
def build_vector_store():
    posts = []
    embeddings_list = []
    
    # Use the DataFrame directly
    for _, row in df.iterrows():
        text = clean_text(str(row.get('selftext', '')))
        if text.strip():  # Only include non-empty posts
            posts.append({
                "id": row.get("id"),
                "subreddit": row.get("subreddit"),
                "text": text
            })
            embeddings_list.append(embedding_model.encode(text))
    
    # Build FAISS index
    d = 384  # Embedding size
    index = faiss.IndexFlatL2(d)
    if embeddings_list:
        embeddings = np.array(embeddings_list)
        index.add(embeddings)
    
    return index, posts

# Initialize vector store with DataFrame data
index, reddit_posts = build_vector_store()

def retrieve_reddit_context(query: str, top_k: int = 2) -> List[Dict[str, Any]]:
    if not reddit_posts:  # Check if we have any posts
        return []
        
    query_embedding = embedding_model.encode(query)
    distances, indices = index.search(np.array([query_embedding]), min(top_k, len(reddit_posts)))
    
    # Filter out invalid indices and ensure we don't exceed list bounds
    valid_posts = [reddit_posts[i] for i in indices[0] if 0 <= i < len(reddit_posts)]
    return valid_posts

# Update the retrieve_reddit_context function
def retrieve_reddit_context(query: str, top_k: int = 2) -> List[Dict[str, Any]]:
    query_embedding = embedding_model.encode(query)
    distances, indices = index.search(np.array([query_embedding]), top_k)
    
    # Filter out invalid indices
    valid_posts = [reddit_posts[i] for i in indices[0] if i < len(reddit_posts)]
    return valid_posts

# Update chat endpoint to use actual data
@app.post("/chat")
async def chat_with_reddit(request: ChatRequest):
    relevant_posts = retrieve_reddit_context(request.user_message)
    retrieved_texts = "\n".join([post["text"] for post in relevant_posts])
    
    prompt = f"""
    You are a chatbot that provides brief, concise responses about social media trends.
    Keep your responses under 50 words and focus on the key points.
    
    Context from similar discussions:
    {retrieved_texts}
    
    User: {request.user_message}
    Provide a brief response:
    """
    
    try:
        response = await call_groq_api(prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Pydantic Model for API
class ChatRequest(BaseModel):
    user_message: str
    context: List[str] = []

# Function to retrieve similar posts
def retrieve_reddit_context(query: str, top_k: int = 2) -> List[Dict[str, Any]]:
    query_embedding = embedding_model.encode(query)
    distances, indices = index.search(np.array([query_embedding]), top_k)
    return [reddit_posts[i] for i in indices[0] if i < len(reddit_posts)]


async def call_groq_api(prompt: str) -> str:
    """Calls the Groq API with the legal analysis prompt."""
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    payload = {
        "model": "mixtral-8x7b-32768",
        "messages": [
            {"role": "system", "content": "You are a chatbot specialized in analyzing and discussing social media trends and Reddit content."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 1,
        "stream": False
    }

    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(GROQ_API_URL, headers=headers, json=payload) as resp:
                response_json = await resp.json()
                
                if "error" in response_json:
                    return f"API Error: {response_json['error'].get('message', 'Unknown error')}"
                
                if "choices" in response_json and len(response_json["choices"]) > 0:
                    message = response_json["choices"][0].get("message", {})
                    return message.get("content", "No response content available.")
                
                return "Failed to get a valid response from the API."
    except Exception as e:
        return f"Error calling Groq API: {str(e)}"

@app.get("/sentiment/all")
def get_all_sentiments():
    try:
        # Use vectorized operations instead of loop
        filtered_df = df.copy()
        
        # Combine and clean text in vectorized operations
        filtered_df['full_text'] = filtered_df['title'].fillna('') + ' ' + filtered_df['selftext'].fillna('')
        filtered_df['clean_text'] = filtered_df['full_text'].apply(clean_text)
        
        # Filter out empty texts
        filtered_df = filtered_df[filtered_df['clean_text'].str.strip().astype(bool)]
        
        # Batch process sentiments
        texts = filtered_df['clean_text'].apply(truncate_text).tolist()
        sentiment_results = sentiment_model(texts, batch_size=32)
        
        # Process results in vectorized operations
        sentiment_labels = [result['label'].upper() for result in sentiment_results]
        confidence_scores = [round(result['score'], 4) for result in sentiment_results]
        
        # Create results using list comprehension
        results = [
            {
                "post_id": str(row['id']),
                "title": str(row['title']),
                "sentiment": label,
                "confidence": score
            }
            for row, label, score in zip(filtered_df.itertuples(), sentiment_labels, confidence_scores)
        ]
        
        # Calculate sentiment counts using Counter
        from collections import Counter
        sentiment_counts = Counter(sentiment_labels)
        total_posts = len(results)
        
        # Calculate distribution
        sentiment_distribution = {
            sentiment: {
                "count": count,
                "percentage": round((count / total_posts) * 100, 2)
            }
            for sentiment, count in sentiment_counts.items()
        }
        
        return {
            "overall_sentiment": max(sentiment_counts, key=sentiment_counts.get),
            "total_posts": total_posts,
            "sentiment_distribution": sentiment_distribution,
            "details": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiments: {str(e)}")

@app.get("/sentiment/{post_id}")
def get_sentiment_by_id(post_id: str):
    try:
        # Find post in DataFrame
        post = df[df['subreddit'] == post_id]
        
        if post.empty:
            raise HTTPException(status_code=404, detail="Post not found")
        
        # Combine title and selftext for better sentiment analysis
        title = str(post['title'].iloc[0]) if not pd.isna(post['title'].iloc[0]) else ''
        selftext = str(post['selftext'].iloc[0]) if not pd.isna(post['selftext'].iloc[0]) else ''
        full_text = f"{title} {selftext}".strip()
        
        if not full_text:
            raise HTTPException(status_code=400, detail="Empty text in post")
        
        # Clean and analyze text
        cleaned_text = clean_text(full_text)
        truncated_text = truncate_text(cleaned_text)
        sentiment_result = sentiment_model(truncated_text)[0]
        sentiment_label = sentiment_result["label"].upper()
        
        # Calculate sentiment distribution
        sentiment_distribution = {
            "POSITIVE": 0,
            "NEGATIVE": 0,
            "NEUTRAL": 0
        }
        sentiment_distribution[sentiment_label] = 100
        
        return {
            "post_id": post_id,
            "title": title,
            "text": selftext,
            "sentiment": sentiment_label,
            "confidence": round(sentiment_result["score"], 4),
            "subreddit": str(post['subreddit'].iloc[0]) if not pd.isna(post['subreddit'].iloc[0]) else '',
            "dominant_sentiment": sentiment_label,
            "sentiment_distribution": sentiment_distribution,
            "explanation": f"This post shows a {sentiment_label.lower()} sentiment with {round(sentiment_result['score'] * 100, 2)}% confidence.",
            "total_analyzed": 1
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing sentiment: {str(e)}")

@app.get("/posts")
def get_all_posts():
    dataset = data if isinstance(data, list) else [data]
    subreddits = [post.get("data", {}) for post in dataset]
    return list(subreddits)

@app.get("/subreddits")
def get_all_subreddits():
    try:
        filtered_df = df.copy()
        
        if 'subreddit' not in filtered_df.columns:
            raise HTTPException(status_code=400, detail="Subreddit column not found in data")
        
        # Group by subreddit and aggregate data
        subreddit_stats = filtered_df.groupby('subreddit').agg({
            'id': 'count',
            'subreddit_subscribers': 'first'
        }).reset_index()
        
        # Format the response
        subreddits_list = [
            {
                "name": f"r/{row['subreddit']}",
                "id": row['subreddit'],
                "subscribers": int(row['subreddit_subscribers']) if not pd.isna(row['subreddit_subscribers']) else 0,
                "posts": int(row['id'])
            }
            for _, row in subreddit_stats.iterrows()
        ]
        
        return subreddits_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/subreddits/{subreddit_id}")
def get_subreddit(subreddit_id: str):
    try:
        # Filter DataFrame for the specific subreddit
        subreddit_df = df[df['subreddit'] == subreddit_id].copy()
        
        if len(subreddit_df) == 0:
            raise HTTPException(status_code=404, detail="Subreddit not found")
        
        # Get subscriber count from first post
        subscribers = 0
        if not subreddit_df['subreddit_subscribers'].empty:
            first_value = subreddit_df['subreddit_subscribers'].iloc[0]
            if pd.notna(first_value):
                subscribers = int(first_value)
        
        # Process posts with proper type handling
        posts = []
        for _, row in subreddit_df.iterrows():
            post = {
                "id": str(row['id']) if pd.notna(row.get('id')) else None,
                "title": str(row['title']) if pd.notna(row.get('title')) else None,
                "selftext": str(row['selftext']) if pd.notna(row.get('selftext')) else None,
                "score": int(row['score']) if pd.notna(row.get('score')) else 0,
                "created_utc": int(row['created_utc']) if pd.notna(row.get('created_utc')) else None,
                "author": str(row['author']) if pd.notna(row.get('author')) else None,
                "num_comments": int(row['num_comments']) if pd.notna(row.get('num_comments')) else 0
            }
            posts.append(post)
        
        return {
            "name": f"r/{subreddit_id}",
            "id": subreddit_id,
            "subscribers": subscribers,
            "posts": posts,
            "total_posts": len(posts)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching subreddit data: {str(e)}")

from typing import List
@app.get("/summary")
async def summarize_all_posts() -> dict:
    """Summarizes all posts and provides a high-level overview."""
    try:
        # Convert DataFrame to list of dictionaries if needed
        if isinstance(df, pd.DataFrame):
            dataset = df.to_dict('records')
        else:
            dataset = data if isinstance(data, list) else [data]
        
        # Extract and clean text from all posts
        all_texts = []
        for post in dataset:
            title = str(post.get('title', '')) if isinstance(post, dict) else ''
            selftext = str(post.get('selftext', '')) if isinstance(post, dict) else ''
            
            full_text = f"{title} {selftext}".strip()
            if full_text:
                cleaned_text = clean_text(full_text)
                if cleaned_text and len(cleaned_text) > 10:  # Ensure meaningful content
                    all_texts.append(cleaned_text)

        if not all_texts:
            raise HTTPException(status_code=400, detail="No valid text content found in posts")

        # Continue with the rest of the function
        summarized_texts = [summarize_text(text) for text in all_texts]
        merged_summary = merge_summaries(summarized_texts, max_words=500)

        if not merged_summary:
            raise HTTPException(status_code=400, detail="Failed to generate summary from texts")

        prompt = f"""
        Analyze the following Reddit content and provide a structured summary in bullet points:

        {merged_summary}

        Provide your response in the following format:
        Main Topics:
        • [Topic 1]
        • [Topic 2]
        • [Topic 3]

        Key Trends:
        • [Trend 1]
        • [Trend 2]

        User Engagement:
        • [Point 1]
        • [Point 2]

        Notable Insights:
        • [Insight 1]
        • [Insight 2]

        Use only bullet points (•) and keep each point brief and clear.
        """

        final_summary = await call_groq_api(prompt)
        if not final_summary:
            raise HTTPException(status_code=500, detail="Failed to generate AI summary")

        return {
            "summary": final_summary,
            "post_count": len(all_texts),
            "analyzed_posts": len(summarized_texts)
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/summarize/{subreddit_id}")
async def summarize_subreddit(subreddit_id: str) -> dict:
    """Summarizes all posts from a subreddit and provides a high-level overview."""
    try:
        # Filter DataFrame for specific subreddit
        subreddit_df = df[df['subreddit'] == subreddit_id].copy()
        
        if len(subreddit_df) == 0:
            raise HTTPException(status_code=404, detail="Subreddit not found")

        # Extract and clean text from all posts
        all_texts = []
        for _, row in subreddit_df.iterrows():
            title = str(row['title']) if pd.notna(row.get('title')) else ''
            selftext = str(row['selftext']) if pd.notna(row.get('selftext')) else ''
            full_text = f"{title} {selftext}".strip()
            
            if full_text:
                cleaned_text = clean_text(full_text)
                if cleaned_text and len(cleaned_text) > 10:  # Ensure meaningful content
                    all_texts.append(cleaned_text)

        if not all_texts:
            raise HTTPException(status_code=400, detail="No valid text content in subreddit posts")

        # Summarize each post individually
        summarized_texts = [summarize_text(text) for text in all_texts]
        
        # Merge summaries to fit within token limits
        merged_summary = merge_summaries(summarized_texts)

        # Generate explanation using Groq API
        prompt = f"""
        Subreddit Summary:

        The following is a summary of discussions in the subreddit "{subreddit_id}":

        {merged_summary}

        Task: Provide a high-level explanation of the main themes, trends, and discussions in this subreddit. Focus on clarity and conciseness.
        """

        final_summary = await call_groq_api(prompt)
        if not final_summary:
            raise HTTPException(status_code=500, detail="Failed to generate summary")

        return {
            "subreddit_id": subreddit_id,
            "summary": final_summary,
            "total_posts": len(all_texts),
            "analyzed_posts": len(summarized_texts)
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")

@app.get("/subreddit/{subreddit_id}/sentiment")
async def explain_sentiment(subreddit_id: str) -> dict:
    """Analyzes sentiment distribution in a subreddit and provides explanations for sentiment trends."""
    try:
        # Filter DataFrame for specific subreddit
        subreddit_df = df[df['subreddit'] == subreddit_id].copy()
        
        if subreddit_df.empty:
            raise HTTPException(status_code=404, detail="Subreddit not found")

        # Combine and clean text
        subreddit_df['full_text'] = subreddit_df['title'].fillna('') + ' ' + subreddit_df['selftext'].fillna('')
        subreddit_df['clean_text'] = subreddit_df['full_text'].apply(clean_text)
        
        # Filter out empty texts
        subreddit_df = subreddit_df[subreddit_df['clean_text'].str.strip().astype(bool)]
        
        if subreddit_df.empty:
            raise HTTPException(status_code=400, detail="No valid text content in subreddit posts")

        # Batch process sentiments
        texts = subreddit_df['clean_text'].apply(truncate_text).tolist()
        sentiment_results = sentiment_model(texts, batch_size=32)
        
        # Process results
        sentiment_labels = [result['label'].upper() for result in sentiment_results]
        from collections import Counter
        sentiment_counts = Counter(sentiment_labels)
        
        # Calculate percentages
        total_posts = len(sentiment_labels)
        sentiment_distribution = {
            sentiment: {
                "count": count,
                "percentage": round((count / total_posts) * 100, 2)
            }
            for sentiment, count in sentiment_counts.items()
        }
        
        # Get dominant sentiment
        dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get)
        
        # Prepare example texts for AI explanation
        example_texts = subreddit_df.head(3)[['clean_text']].values.flatten().tolist()
        example_sentiments = sentiment_labels[:3]
        examples = [f"Text: {text}\nSentiment: {sent}" for text, sent in zip(example_texts, example_sentiments)]

        # Build prompt for sentiment explanation
        prompt = f"""
        Sentiment Analysis Explanation:

        Subreddit: {subreddit_id}

        Sentiment Distribution:
        {', '.join(f"{k}: {v['count']} ({v['percentage']}%)" for k, v in sentiment_distribution.items())}

        Example Posts:
        {examples}

        Task: Explain why this subreddit shows a {dominant_sentiment} sentiment trend. 
        Consider the topics discussed, common themes, and user interactions.
        Keep the explanation concise and focused on key patterns.
        """

        sentiment_explanation = await call_groq_api(prompt)

        return {
            "subreddit_id": subreddit_id,
            "dominant_sentiment": dominant_sentiment,
            "total_analyzed": total_posts,
            "sentiment_distribution": sentiment_distribution,
            "explanation": sentiment_explanation
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing subreddit sentiment: {str(e)}")

@app.post("/chat")
async def chat_with_reddit(request: ChatRequest):
    relevant_posts = retrieve_reddit_context(request.user_message)
    retrieved_texts = "\n".join([post["text"] for post in relevant_posts])
    
    prompt = f"""
    You are a chatbot specialized in social media discussions, particularly on Reddit.
    
    Context from similar Reddit discussions:
    {retrieved_texts}
    
    User: {request.user_message}
    Chatbot Response:
    """
    
    try:
        response = await call_groq_api(prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
df = load_data("./data.jsonl")

@app.get("/subreddit/{subreddit_id}/top-words")
async def get_subreddit_top_words(subreddit_id: str, limit: int = 10):
    """Returns the most frequently used words in a subreddit's posts."""
    try:
        # Filter DataFrame for specific subreddit
        subreddit_df = df[df['subreddit'] == subreddit_id].copy()
        
        if len(subreddit_df) == 0:
            raise HTTPException(status_code=404, detail="Subreddit not found")

        # Combine title and selftext
        subreddit_df['full_text'] = subreddit_df['title'].fillna('') + ' ' + subreddit_df['selftext'].fillna('')
        
        # Create vectorizer for word frequency analysis
        vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            stop_words='english',
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # Match words with 2 or more letters
        )
        
        # Fit and transform the text data
        X = vectorizer.fit_transform(subreddit_df['full_text'])
        
        # Get word frequencies
        word_freq = zip(vectorizer.get_feature_names_out(), X.sum(axis=0).A1)
        
        # Sort by frequency and get top words
        top_words = sorted(word_freq, key=lambda x: x[1], reverse=True)[:limit]
        
        return {
            "subreddit_id": subreddit_id,
            "total_posts_analyzed": len(subreddit_df),
            "top_words": [
                {
                    "word": word,
                    "frequency": int(count),
                    "percentage": round((count / len(subreddit_df)) * 100, 2)
                }
                for word, count in top_words
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing top words: {str(e)}")

from sklearn.decomposition import LatentDirichletAllocation

@app.get("/top-trends")
async def get_top_trends():
    try:
        filtered_df = df.copy()
        
        # Ensure we have text data to analyze
        if 'selftext' not in filtered_df.columns and 'title' not in filtered_df.columns:
            raise HTTPException(status_code=400, detail="Required text columns not found in data")
        
        # Combine title and selftext for analysis
        filtered_df['text_content'] = filtered_df['title'].fillna('') + ' ' + filtered_df['selftext'].fillna('')
        
        # Clean text
        filtered_df['clean_text'] = filtered_df['text_content'].apply(lambda x: clean_text(x))
        
        # Create date column if it doesn't exist
        if 'created_utc' in filtered_df.columns:
            filtered_df['date'] = pd.to_datetime(filtered_df['created_utc'], unit='s').dt.date
        else:
            # Use current date if no timestamp available
            filtered_df['date'] = pd.Timestamp.now().date()
        
        # Vectorize text
        vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
        X = vectorizer.fit_transform(filtered_df['clean_text'])
        
        # Apply LDA for topic modeling
        lda = LatentDirichletAllocation(n_components=5, random_state=42)
        lda.fit(X)
        
        # Extract topics
        words = vectorizer.get_feature_names_out()
        topics = {i: [words[idx] for idx in topic.argsort()[-10:]] for i, topic in enumerate(lda.components_)}
        
        # Assign topics to posts
        filtered_df['topic'] = lda.transform(X).argmax(axis=1)
        
        # Aggregate trends over time
        topic_trends = filtered_df.groupby(['date', 'topic']).size().unstack(fill_value=0)
        
        # Rename topics with their key terms
        topic_names = {i: f"Topic{i+1}" for i in range(5)}
        topic_trends.columns = [topic_names[col] for col in topic_trends.columns]
        
        # Get topic keywords for frontend display
        topic_keywords = {f"Topic{i+1}": ", ".join(topics[i][-5:]) for i in range(5)}
        
        return {
            'dates': [str(date) for date in topic_trends.index],
            'trends': {
                topic: values.tolist() 
                for topic, values in topic_trends.items()
            },
            'topic_keywords': topic_keywords
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/top-contributors")
async def get_top_contributors():
    try:
        filtered_df = df.copy()
        
        if 'author' not in filtered_df.columns:
            raise HTTPException(status_code=400, detail="Author column not found in data")
        
        # Get top 5 contributors and their post counts
        top_contributors = filtered_df['author'].value_counts()\
            .head(5)\
            .to_frame(name='count')\
            .reset_index()\
            .rename(columns={'index': 'author'})
        
        return {
            'contributors': [{
                'author': row['author'],
                'posts': int(row['count'])
            } for _, row in top_contributors.iterrows()]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/top-hashtags")
async def get_top_hashtags():
    try:
        filtered_df = df.copy()
        
        # Extract hashtags from title or content if hashtags column doesn't exist
        if 'hashtags' not in filtered_df.columns:
            filtered_df['hashtags'] = filtered_df['title'].fillna('').str.findall(r'#\w+') + \
                                    filtered_df['selftext'].fillna('').str.findall(r'#\w+')
        
        # Flatten hashtags list and get top 7
        hashtags = [tag for tags in filtered_df['hashtags'] for tag in tags if tags]
        if not hashtags:
            return {'hashtags': []}
            
        hashtag_counts = pd.Series(hashtags).value_counts().head(7)
        
        return {
            'hashtags': [{
                'tag': tag,
                'count': int(count)
            } for tag, count in hashtag_counts.items()]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)
