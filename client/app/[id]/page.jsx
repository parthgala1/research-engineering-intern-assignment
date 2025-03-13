"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import DashboardHeader from "@/components/dashboard/DashboardHeader";
import TimeSeriesChart from "@/components/charts/TimeSeriesChart";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";

// Mock data for demonstration

export default function SubredditAnalysis() {
  const params = useParams();
  const [loading, setLoading] = useState(true);
  // Update the initial state
  const [subredditData, setSubredditData] = useState({
    name: "",
    subscribers: 0,
    posts: [],
    sentiment: {
      dominant_sentiment: "",
      sentiment_distribution: {},
      explanation: "",
      total_analyzed: 0,
    },
    summary: "",
    dateRange: {
      start: new Date(2024, 0, 1),
      end: new Date(),
    },
    subreddit_id: "",
    total_posts_analyzed: 0,
    top_words: [],
  });
  // Add these state variables after other state declarations
  const [currentPage, setCurrentPage] = useState(1);
  const postsPerPage = 10;

  // Add this before the return statement
  const indexOfLastPost = currentPage * postsPerPage;
  const indexOfFirstPost = indexOfLastPost - postsPerPage;
  const currentPosts =
    subredditData.posts?.slice(indexOfFirstPost, indexOfLastPost) || [];
  const totalPages = Math.ceil(
    (subredditData.posts?.length || 0) / postsPerPage
  );

  // Update the useEffect fetch
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);

        // Parallel API calls
        const [
          subredditResponse,
          sentimentResponse,
          summaryResponse,
          topWordsResponse,
        ] = await Promise.all([
          fetch(`http://localhost:8000/subreddits/${params.id}`),
          fetch(`http://localhost:8000/subreddit/${params.id}/sentiment`),
          fetch(`http://localhost:8000/summarize/${params.id}`),
          fetch(`http://localhost:8000/subreddit/${params.id}/top-words`),
        ]);

        const [subredditInfo, sentimentInfo, summaryInfo, topWordsInfo] =
          await Promise.all([
            subredditResponse.json(),
            sentimentResponse.json(),
            summaryResponse.json(),
            topWordsResponse.json(),
          ]);

        setSubredditData({
          name: subredditInfo.name,
          subscribers: subredditInfo.subscribers || 0,
          posts: subredditInfo.posts || [],
          sentiment: {
            dominant_sentiment: sentimentInfo.dominant_sentiment,
            sentiment_distribution: sentimentInfo.sentiment_distribution,
            explanation: sentimentInfo.explanation,
            total_analyzed: sentimentInfo.total_analyzed,
          },
          summary: summaryInfo.summary,
          dateRange: {
            start: new Date(2024, 0, 1),
            end: new Date(),
          },
          subreddit_id: topWordsInfo.subreddit_id,
          total_posts_analyzed: topWordsInfo.total_posts_analyzed,
          top_words: topWordsInfo.top_words,
        });
      } catch (error) {
        console.error("Error fetching data:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [params.id]);

  // Update the Top Words card rendering
  <Card>
    <CardHeader>
      <CardTitle>Top Words</CardTitle>
    </CardHeader>
    <CardContent>
      <div className="space-y-3">
        {subredditData.top_words?.map((item, index) => (
          <div key={index} className="flex items-center">
            <span className="w-24 font-medium">{item.word}</span>
            <div className="flex-1">
              <Progress value={item.percentage} className="h-3 bg-gray-200" />
            </div>
            <span className="w-16 text-right text-sm text-gray-500">
              {item.frequency}
            </span>
          </div>
        ))}
      </div>
    </CardContent>
  </Card>;
  console.log(subredditData);

  // Update the loading check
  if (loading) {
    return (
      <div className="min-h-screen bg-background">
        <DashboardHeader />
        <main className="container mx-auto px-4 py-6">
          <div className="flex justify-center items-center h-64">
            <p className="text-lg">Loading subreddit analysis...</p>
          </div>
        </main>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <DashboardHeader />
      <main className="container mx-auto px-4 py-6">
        {/* Header section */}
        <div className="mb-6">
          <Link
            href="/"
            className="flex items-center text-sm text-blue-500 hover:underline"
          >
            <ArrowLeft className="h-4 w-4 mr-1" />
            Back to Dashboard
          </Link>

          <div className="mt-4 flex flex-col md:flex-row md:items-center md:justify-between">
            <div>
              <h1 className="text-3xl font-bold">{subredditData.name}</h1>
              <div className="flex gap-2 mt-2">
                <Badge variant="outline">
                  {subredditData.subscribers.toLocaleString()} subscribers
                </Badge>
                <Badge variant="outline">
                  {subredditData.total_posts_analyzed.toLocaleString()} posts
                  analyzed
                </Badge>
              </div>
            </div>
          </div>
        </div>

        <div className="grid gap-6">
          {/* Summary and Sentiment Analysis */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Summary</CardTitle>
              </CardHeader>
              <CardContent>
                <p>{subredditData.summary}</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Sentiment Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {Object.entries(
                    subredditData.sentiment.sentiment_distribution
                  ).map(([sentiment, data]) => (
                    <div key={sentiment}>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm">
                          {sentiment.charAt(0) +
                            sentiment.slice(1).toLowerCase()}{" "}
                          ({data.percentage}%)
                        </span>
                      </div>
                      <Progress
                        value={data.percentage}
                        className={`h-4 ${
                          sentiment === "POSITIVE"
                            ? "text-green-200"
                            : sentiment === "NEGATIVE"
                            ? "text-red-200"
                            : "text-yellow-200"
                        }`}
                      />
                    </div>
                  ))}
                </div>

                {subredditData.sentiment.explanation && (
                  <div className="mt-6">
                    <h3 className="font-medium mb-2">Analysis</h3>
                    <p className="text-sm text-gray-600">
                      {subredditData.sentiment.explanation}
                    </p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          {/* Word Analysis */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Top Words</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {subredditData.top_words?.map((item, index) => (
                    <div key={index} className="flex items-center">
                      <span className="w-24 font-medium">{item.word}</span>
                      <div className="flex-1">
                        <Progress
                          value={item.percentage}
                          className="h-3 bg-gray-200"
                        />
                      </div>
                      <span className="w-16 text-right text-sm text-gray-500">
                        {item.frequency}
                      </span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Overview</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div>
                    <h3 className="font-medium mb-2">Subreddit Analysis</h3>
                    <p className="text-sm text-gray-600">
                      r/{params.id} is a community with{" "}
                      {subredditData.subscribers.toLocaleString()} subscribers.
                      Based on{" "}
                      {subredditData.total_posts_analyzed.toLocaleString()}{" "}
                      analyzed posts, the discussions primarily focus on{" "}
                      {subredditData.top_words
                        ?.slice(0, 3)
                        .map((w) => w.word)
                        .join(", ")}
                      .
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Posts Table */}

          <Card className="col-span-full">
            <CardHeader>
              <CardTitle>Recent Posts</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="rounded-md border">
                <table className="w-full">
                  <thead>
                    <tr className="border-b bg-muted/50">
                      <th className="p-2 text-left">Title</th>
                      <th className="p-2 text-left">Score</th>
                      <th className="p-2 text-left">Comments</th>
                      <th className="p-2 text-left">Author</th>
                      <th className="p-2 text-left">Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {currentPosts.map((post, index) => (
                      <tr key={index} className="border-b hover:bg-muted/50">
                        <td className="p-2">
                          <div className="max-w-sm">
                            <p className="truncate" title={post.title}>
                              {post.title}
                            </p>
                          </div>
                        </td>
                        <td className="p-2 text-sm">{post.score}</td>
                        <td className="p-2 text-sm">{post.num_comments}</td>
                        <td className="p-2 text-sm">{post.author}</td>
                        <td className="p-2 text-sm text-gray-600">
                          {new Date(
                            post.created_utc * 1000
                          ).toLocaleDateString()}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                {/* Pagination Controls */}
                <div className="flex items-center justify-between px-4 py-3 bg-white border-t">
                  <div className="flex items-center gap-2">
                    <p className="text-sm text-gray-700">
                      Showing {indexOfFirstPost + 1} to{" "}
                      {Math.min(
                        indexOfLastPost,
                        subredditData.posts?.length || 0
                      )}{" "}
                      of {subredditData.posts?.length || 0} posts
                    </p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                      disabled={currentPage === 1}
                      className={`px-3 py-1 text-sm rounded border ${
                        currentPage === 1
                          ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                          : "bg-white text-gray-700 hover:bg-gray-50"
                      }`}
                    >
                      Previous
                    </button>
                    {[...Array(totalPages)].map((_, i) => {
                      const pageNumber = i + 1;
                      
                      // Always show first two and last two pages
                      if (
                        pageNumber <= 2 ||
                        pageNumber > totalPages - 2 ||
                        Math.abs(currentPage - pageNumber) <= 1
                      ) {
                        return (
                          <button
                            key={pageNumber}
                            onClick={() => setCurrentPage(pageNumber)}
                            className={`px-3 py-1 text-sm rounded border ${
                              currentPage === pageNumber
                                ? "bg-blue-50 text-blue-600 border-blue-500"
                                : "bg-white text-gray-700 hover:bg-gray-50"
                            }`}
                          >
                            {pageNumber}
                          </button>
                        );
                      }
                      
                      // Show ellipsis for skipped pages
                      if (
                        pageNumber === 3 ||
                        pageNumber === totalPages - 2
                      ) {
                        return (
                          <span
                            key={pageNumber}
                            className="px-2 text-gray-500"
                          >
                            ...
                          </span>
                        );
                      }
                      
                      return null;
                    })}
                    
                    <button
                      onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                      disabled={currentPage === totalPages}
                      className={`px-3 py-1 text-sm rounded border ${
                        currentPage === totalPages
                          ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                          : "bg-white text-gray-700 hover:bg-gray-50"
                      }`}
                    >
                      Next
                    </button>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}
