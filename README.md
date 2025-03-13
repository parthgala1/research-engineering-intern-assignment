# Social Media Analysis Dashboard

An interactive web platform for analyzing and visualizing social media data patterns, trends, and user behaviors.

## Live Demo

🔗 [View Live Demo](https://youtu.be/M-n5d21Ib9Y)

## Features

### 1. Trending Topics Analysis

![Trending Topics](/images/timeseries.png)

- Real-time visualization of trending topics over time
- Interactive time series charts
- Topic categorization (Technology, Politics, Entertainment, Sports, Health)

### 2. User Engagement Metrics

![User Engagement](/images/charts.png)

- Top contributors visualization
- User activity patterns
- Community participation analysis

### 3. Content Analysis

![Content Analysis](/images//chatbot.mov)

- Hashtag frequency analysis
- Word cloud visualization
- Sentiment analysis of posts
- Network visualization of content sharing patterns

### 4. AI-Powered Features

![AI Features](/images/ai-summary.png)

- AI-generated trend summaries
- Interactive data insights chatbot
- Smart content categorization

## Technology Stack

### Frontend

- React.js
- TailwindCSS
- D3.js for visualizations
- React Query for data fetching

### Backend

- Flask (Python)
- Natural Language Processing
- Machine Learning algorithms
- RESTful API architecture

## Local Development

1. Clone the repository:

```bash
git clone https://github.com/yourusername/social-media-dashboard.git
cd social-media-dashboard
```

2. Install dependencies:

```bash
cd frontend
npm install
cd ../backend
pip install -r requirements.txt
```

3. Run the development servers:

```bash
# Frontend
cd frontend
npm start
# Backend
cd backend
python api/main.py
```

4. Environment Variables
   Create a `.env` file in the `backend` directory with the following variables:

```bash
GROQ_API_KEY=your_groq_api_key
```

# API Documentation

## Base URL

```
http://localhost:5000
```

## Endpoints

### 1. Chat Endpoint

```http
POST /chat
```

Interactive chatbot for discussing social media trends and Reddit content.

**Request Body:**

```json
{
  "user_message": "string",
  "context": ["string"]
}
```

### 2. Sentiment Analysis

#### Get All Sentiments

```http
GET /sentiment/all
```

Returns sentiment analysis for all posts with distribution statistics.

**Response:**

```json
{
  "overall_sentiment": "string",
  "total_posts": "number",
  "sentiment_distribution": {
    "POSITIVE": { "count": "number", "percentage": "number" },
    "NEGATIVE": { "count": "number", "percentage": "number" },
    "NEUTRAL": { "count": "number", "percentage": "number" }
  },
  "details": [...]
}
```

#### Get Sentiment by Post ID

```http
GET /sentiment/{post_id}
```

Returns sentiment analysis for a specific post.

### 3. Posts and Subreddits

#### Get All Posts

```http
GET /posts
```

Returns all available posts in the dataset.

#### Get All Subreddits

```http
GET /subreddits
```

Returns list of all subreddits with basic statistics.

**Response:**

```json
[
  {
    "name": "string",
    "id": "string",
    "subscribers": "number",
    "posts": "number"
  }
]
```

#### Get Specific Subreddit

```http
GET /subreddits/{subreddit_id}
```

Returns detailed information about a specific subreddit.

### 4. Content Analysis

#### Get Summary

```http
GET /summary
```

Provides AI-generated summary of all posts with key insights.

#### Get Subreddit Summary

```http
GET /summarize/{subreddit_id}
```

Generates summary for a specific subreddit's content.

#### Get Subreddit Sentiment Analysis

```http
GET /subreddit/{subreddit_id}/sentiment
```

Detailed sentiment analysis for a specific subreddit.

#### Get Top Words in Subreddit

```http
GET /subreddit/{subreddit_id}/top-words
```

Returns most frequently used words in a subreddit.

**Query Parameters:**

- `limit` (optional): Number of top words to return (default: 10)

### 5. Trend Analysis

#### Get Top Trends

```http
GET /top-trends
```

Returns trending topics and their evolution over time.

#### Get Top Contributors

```http
GET /top-contributors
```

Returns the most active users and their post counts.

#### Get Top Hashtags

```http
GET /top-hashtags
```

Returns most frequently used hashtags.

## Error Responses

All endpoints may return the following error responses:

```json
{
  "detail": "Error message"
}
```

**Status Codes:**

- `200`: Success
- `400`: Bad Request
- `404`: Not Found
- `500`: Internal Server Error

## Rate Limiting

No rate limiting is currently implemented.
