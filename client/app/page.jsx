"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import DashboardHeader from "@/components/dashboard/DashboardHeader";
import TimeSeriesChart from "@/components/charts/TimeSeriesChart";
import TopContributorsChart from "@/components/charts/TopContributorsChart";
import HashtagsChart from "@/components/charts/HashtagsChart";
import ChatbotInterface from "@/components/dashboard/ChatbotInterface";
import InsightsSummary from "@/components/dashboard/InsightsSummary";
import RoadmapBoard from "@/components/dashboard/RoadmapBoard";
import { Card, CardContent, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Badge } from "@/components/ui/badge";
import { Loader2 } from "lucide-react";

export default function Dashboard() {
  const [dateRange, setDateRange] = useState({
    start: new Date(2024, 0, 1),
    end: new Date(),
  });
  const [selectedSubreddit, setSelectedSubreddit] = useState(null);
  const [selectedHashtag, setSelectedHashtag] = useState(null);
  const [subreddits, setSubreddits] = useState([]);
  const [loading, setLoading] = useState(true);
  const [posts, setPosts] = useState([]);
  // Add this state with other state declarations at thetop
  const [currentPage, setCurrentPage] = useState(1);
  // Add these calculations before the return statement
  const postsPerPage = 10;
  const indexOfLastPost = currentPage * postsPerPage;
  const indexOfFirstPost = indexOfLastPost - postsPerPage;
  const currentPosts = posts.slice(indexOfFirstPost, indexOfLastPost);
  const totalPages = Math.ceil(posts.length / postsPerPage);

  // Add new loading state
  const [sentimentLoading, setSentimentLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    setSentimentLoading(true);

    // Fetch subreddits data
    fetch("http://localhost:8000/subreddits", {
      method: "GET",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
    })
      .then((response) => {
        if (!response.ok) throw new Error("Failed to fetch subreddits");
        return response.json();
      })
      .then((data) => {
        setSubreddits(data); // Direct assignment as the API returns array of subreddits
      })
      .catch((error) => {
        console.error("Error loading subreddits:", error);
        setSubreddits([]);
      })
      .finally(() => setLoading(false));

    // Fetch sentiment analysis data
    fetch("http://localhost:8000/sentiment/all", {
      method: "GET",
      headers: {
        Accept: "application/json",
        // "Content-Type": "application/json",
      },
    })
      .then((response) => {
        if (!response.ok) throw new Error("Failed to fetch sentiment data");
        return response.json();
      })
      .then((data) => setPosts(data.details || []))
      .catch((error) => {
        console.error("Error fetching sentiment data:", error);
        setPosts([]);
      })
      .finally(() => setSentimentLoading(false));
  }, []);

  console.log(posts);

  return (
    <div className="min-h-screen bg-background">
      <DashboardHeader />
      <main className="container mx-auto px-4 py-6">
        <div className="grid gap-6">
          {/* Subreddits Card Section */}
          <Card className="p-6">
            <CardTitle className="mb-4">Subreddits</CardTitle>
            <CardContent className="p-0">
              {selectedSubreddit ? (
                <div className="mb-4">
                  <button
                    onClick={() => setSelectedSubreddit(null)}
                    className="text-sm text-blue-500 hover:underline flex items-center"
                  >
                    ← Back to all subreddits
                  </button>
                  <h3 className="text-xl font-bold mt-2">
                    {selectedSubreddit.name}
                  </h3>
                  <div className="flex gap-2 mt-1">
                    <Badge variant="outline">
                      {selectedSubreddit.subscribers.toLocaleString()}{" "}
                      subscribers
                    </Badge>
                    <Badge variant="outline">
                      {selectedSubreddit.posts.toLocaleString()} posts
                    </Badge>
                  </div>
                  <div className="mt-4">
                    <Link
                      href={`/${selectedSubreddit.name.replace("r/", "")}`}
                      className="inline-block px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
                    >
                      View Detailed Analysis
                    </Link>
                  </div>
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                  {loading ? (
                    <div className="col-span-full flex justify-center items-center p-8">
                      <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
                      <span className="ml-2">Loading data...</span>
                    </div>
                  ) : (
                    subreddits.map((subreddit, index) => {
                      // Extract the subreddit ID from the name (remove "r/" prefix)
                      const subredditId = subreddit.name.replace("r/", "");

                      return (
                        <Link
                          key={index}
                          href={`/${subredditId}`}
                          className="block"
                        >
                          <Card className="cursor-pointer hover:bg-gray-50 transition-colors h-full">
                            <CardContent className="p-4">
                              <h3 className="font-medium">{subreddit.name}</h3>
                              <p className="text-sm text-gray-500">
                                {subreddit.subscribers.toLocaleString()}{" "}
                                subscribers
                              </p>
                              <p className="text-xs text-gray-400">
                                {subreddit.posts.toLocaleString()} posts
                              </p>
                            </CardContent>
                          </Card>
                        </Link>
                      );
                    })
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Main Analytics Section */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <Card className="col-span-full p-6">
              <TimeSeriesChart
                dateRange={dateRange}
                onDateRangeChange={setDateRange}
                selectedSubreddit={selectedSubreddit}
                selectedHashtag={selectedHashtag}
              />
            </Card>

            <Card className="p-6">
              <TopContributorsChart
                dateRange={dateRange}
                selectedSubreddit={selectedSubreddit}
                selectedHashtag={selectedHashtag}
              />
            </Card>

            <Card className="p-6">
              <HashtagsChart
                dateRange={dateRange}
                selectedSubreddit={selectedSubreddit}
                onHashtagSelect={setSelectedHashtag}
              />
            </Card>
          </div>
          <Card className="p-6">
            <InsightsSummary
              dateRange={dateRange}
              selectedSubreddit={selectedSubreddit}
              selectedHashtag={selectedHashtag}
            />
          </Card>

          {/* Data and Interaction Section */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <Card className="lg:col-span-2 p-6">
              <Tabs defaultValue="sentiment">
                <TabsList>
                  <TabsTrigger value="sentiment">
                    Sentiment Analysis
                  </TabsTrigger>
                  <TabsTrigger value="roadmap">Roadmap</TabsTrigger>
                </TabsList>
                <TabsContent value="table"></TabsContent>

                <TabsContent value="sentiment">
                  {sentimentLoading ? (
                    <div className="flex justify-center items-center p-8">
                      <Loader2 className="h-8 w-8 animate-spin text-blue-500" />
                      <span className="ml-2">Loading sentiment data...</span>
                    </div>
                  ) : (
                    <div className="space-y-4">
                      <div className="flex justify-between items-center">
                        <h3 className="text-lg font-semibold">
                          Posts Sentiment Analysis
                        </h3>
                        <div className="flex gap-2">
                          <Badge variant="success">Positive</Badge>
                          <Badge variant="warning">Neutral</Badge>
                          <Badge variant="destructive">Negative</Badge>
                        </div>
                      </div>

                      {/* Overall Statistics */}
                      <div className="grid grid-cols-4 gap-4 mb-4">
                        <Card className="p-4">
                          <p className="text-sm text-muted-foreground">
                            Total Posts
                          </p>
                          <p className="text-2xl font-bold">{posts.length}</p>
                        </Card>
                        <Card className="p-4">
                          <p className="text-sm text-muted-foreground">
                            Positive
                          </p>
                          <p className="text-2xl font-bold text-green-600">
                            {
                              posts.filter(
                                (post) => post.sentiment === "POSITIVE"
                              ).length
                            }
                          </p>
                        </Card>
                        <Card className="p-4">
                          <p className="text-sm text-muted-foreground">
                            Neutral
                          </p>
                          <p className="text-2xl font-bold text-yellow-600">
                            {
                              posts.filter(
                                (post) => post.sentiment === "NEUTRAL"
                              ).length
                            }
                          </p>
                        </Card>
                        <Card className="p-4">
                          <p className="text-sm text-muted-foreground">
                            Negative
                          </p>
                          <p className="text-2xl font-bold text-red-600">
                            {
                              posts.filter(
                                (post) => post.sentiment === "NEGATIVE"
                              ).length
                            }
                          </p>
                        </Card>
                      </div>

                      {/* Detailed Table */}
                      <div className="rounded-md border">
                        <table className="w-full">
                          <thead>
                            <tr className="border-b bg-muted/50">
                              <th className="p-2 text-left">ID</th>
                              <th className="p-2 text-left">Title</th>
                              <th className="p-2 text-left">Sentiment</th>
                              <th className="p-2 text-left">Confidence</th>
                            </tr>
                          </thead>

                          <tbody>
                            {currentPosts.map((post, index) => (
                              <tr
                                key={index}
                                className="border-b hover:bg-muted/50"
                              >
                                <td className="p-2 font-mono text-sm">
                                  {post.post_id}
                                </td>
                                <td className="p-2">
                                  <div className="max-w-sm">
                                    <p className=" truncate" title={post.text}>
                                      {post.title}
                                    </p>
                                  </div>
                                </td>
                                <td className="p-2">
                                  <Badge
                                    variant={
                                      post.sentiment === "POSITIVE"
                                        ? "success"
                                        : post.sentiment === "NEGATIVE"
                                        ? "destructive"
                                        : "warning"
                                    }
                                  >
                                    {post.sentiment}
                                  </Badge>
                                </td>
                                <td className="p-2">
                                  <div className="w-full bg-gray-200 rounded-full h-2.5">
                                    <div
                                      className={`h-2.5 rounded-full ${
                                        post.sentiment === "POSITIVE"
                                          ? "bg-green-600"
                                          : post.sentiment === "NEGATIVE"
                                          ? "bg-red-600"
                                          : "bg-yellow-600"
                                      }`}
                                      style={{
                                        width: `${post.confidence * 100}%`,
                                      }}
                                    ></div>
                                  </div>
                                  <span className="text-xs text-gray-500 mt-1">
                                    {(post.confidence * 100).toFixed(1)}%
                                  </span>
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                        {/* Add this pagination controls after the table */}
                        <div className="flex items-center justify-between p-4 border-t">
                          <button
                            onClick={() =>
                              setCurrentPage((prev) => Math.max(prev - 1, 1))
                            }
                            disabled={currentPage === 1}
                            className="px-3 py-1 rounded border disabled:opacity-50 hover:bg-gray-100"
                          >
                            Previous
                          </button>
                          <span className="text-sm text-gray-600">
                            Page {currentPage} of {totalPages}
                          </span>
                          <button
                            onClick={() =>
                              setCurrentPage((prev) =>
                                Math.min(prev + 1, totalPages)
                              )
                            }
                            disabled={currentPage === totalPages}
                            className="px-3 py-1 rounded border disabled:opacity-50 hover:bg-gray-100"
                          >
                            Next
                          </button>
                        </div>
                      </div>
                    </div>
                  )}
                </TabsContent>
                <TabsContent value="roadmap">
                  <RoadmapBoard />
                </TabsContent>
              </Tabs>
            </Card>

            <Card className="p-6">
              <ChatbotInterface />
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}
