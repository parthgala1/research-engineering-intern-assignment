"use client";

import { useState, useEffect } from "react";
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";

const COLORS = ["#8884d8", "#82ca9d", "#ffc658", "#ff7300", "#ff0000", "#00C49F", "#FFBB28"];

export default function HashtagsChart({ dateRange, selectedSubreddit, onHashtagSelect }) {
  const [hashtags, setHashtags] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("http://localhost:8000/top-hashtags");
        if (!response.ok) {
          throw new Error("Failed to fetch hashtags");
        }
        const data = await response.json();
        
        // Transform the data for the chart
        const transformedData = data.hashtags.map(item => ({
          name: item.tag,
          value: item.count
        }));
        
        setHashtags(transformedData);
      } catch (error) {
        console.error("Error fetching hashtags:", error);
      }
    };
    fetchData();
  }, [dateRange, selectedSubreddit]);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Top Hashtags</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="h-[300px]">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={hashtags}
                dataKey="value"
                nameKey="name"
                cx="50%"
                cy="50%"
                outerRadius={80}
                onClick={(data) => onHashtagSelect(data.name)}
              >
                {hashtags.map((entry, index) => (
                  <Cell 
                    key={`cell-${index}`} 
                    fill={COLORS[index % COLORS.length]}
                  />
                ))}
              </Pie>
              <Tooltip formatter={(value) => `${value} posts`} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}