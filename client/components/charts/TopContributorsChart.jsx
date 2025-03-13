"use client";

import { useState, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

export default function TopContributorsChart({
  dateRange,
  selectedSubreddit,
  selectedHashtag,
}) {
  const [contributors, setContributors] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("http://localhost:8000/top-contributors");
        if (!response.ok) {
          throw new Error("Failed to fetch contributors");
        }
        const data = await response.json();
        
        // Transform the data for the chart
        const transformedData = data.contributors.map(contributor => ({
          name: contributor.author,
          posts: contributor.posts
        }));
        
        setContributors(transformedData);
      } catch (error) {
        console.error("Error fetching contributors:", error);
      }
    };
    fetchData();
  }, [dateRange, selectedSubreddit, selectedHashtag]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Top Contributors</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={contributors}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="name"
                angle={-45}
                textAnchor="end"
                height={60}
                interval={0}
                tick={{ fontSize: 8 }}
              />
              <YAxis
                allowDecimals={false}
                label={{ value: 'Number of Posts', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip />
              <Bar 
                dataKey="posts" 
                fill="hsl(var(--primary))"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
