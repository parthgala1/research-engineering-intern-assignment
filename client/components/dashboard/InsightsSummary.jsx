"use client";

import { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";

export default function InsightsSummary({
  dateRange,
  selectedSubreddit,
  selectedHashtag,
}) {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchSummary = async () => {
      try {
        setLoading(true);
        const response = await fetch("http://localhost:8000/summary");
        if (!response.ok) {
          throw new Error("Failed to fetch summary");
        }
        const data = await response.json();
        setSummary(data.summary);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchSummary();
  }, [dateRange, selectedSubreddit, selectedHashtag]);

  const renderContent = () => {
    if (loading) {
      return (
        <div className="space-y-3">
          <Skeleton className="h-4 w-[250px]" />
          <Skeleton className="h-4 w-[200px]" />
          <Skeleton className="h-4 w-[300px]" />
        </div>
      );
    }

    if (error) {
      return (
        <p className="text-sm text-red-500">
          Failed to load insights. Please try again later.
        </p>
      );
    }

    if (!summary) {
      return (
        <p className="text-sm text-muted-foreground">
          No insights available at the moment.
        </p>
      );
    }

    return (
      <div className="space-y-6">
        {summary.split('\n\n').map((section, index) => {
          const [title, ...points] = section.split('\n');
          if (!title || !points.length) return null;
          
          return (
            <div key={index} className="space-y-2">
              <h3 className="font-semibold text-sm">{title.replace(':', '')}</h3>
              <ul className="list-disc pl-4 space-y-1.5 text-lg">
                {points.map((point, pointIndex) => (
                  <li key={pointIndex} className="text-sm text-muted-foreground">
                    {point.replace('•', '').trim()}
                  </li>
                ))}
              </ul>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center justify-between text-xl">
          AI-Generated Insights
          {loading && (
            <span className="text-xs text-muted-foreground">Analyzing...</span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>{renderContent()}</CardContent>
    </Card>
  );
}
