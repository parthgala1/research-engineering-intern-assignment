"use client";

import { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { DatePickerWithRange } from "@/components/ui/date-range-picker";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const COLORS = ["#8884d8", "#82ca9d", "#ffc658", "#ff7300", "#ff0000"];

export default function TimeSeriesChart({
  dateRange,
  onDateRangeChange,
  selectedSubreddit,
  selectedHashtag,
}) {
  const [data, setData] = useState([]);
  const [topics, setTopics] = useState([]);
  const [activeTopics, setActiveTopics] = useState({});

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch("http://localhost:8000/top-trends");
        const result = await response.json();

        // Transform the data for the chart
        const transformedData = result.dates.map((date, index) => {
          const dataPoint = { date };
          Object.entries(result.trends).forEach(([topic, values]) => {
            dataPoint[topic] = values[index];
          });
          return dataPoint;
        });

        const topicsList = Object.keys(result.trends);
        setData(transformedData);
        setTopics(topicsList);

        // Initialize all topics as active
        const initialActiveState = {};
        topicsList.forEach((topic) => {
          initialActiveState[topic] = true;
        });
        setActiveTopics(initialActiveState);
      } catch (error) {
        console.error("Error fetching trend data:", error);
      }
    };

    fetchData();
  }, [dateRange]);

  const toggleTopic = (topic) => {
    setActiveTopics((prev) => ({
      ...prev,
      [topic]: !prev[topic],
    }));
  };

  // Format topic name for display
  const formatTopicName = (topic) => {
    // Extract index from topic name or use array index
    const topicIndex = topic.startsWith("Topic ") 
      ? parseInt(topic.split(" ")[1]) + 1 
      : topics.indexOf(topic) + 1;
    return `Topic${topicIndex}`;
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle>Trending Topics Over Time</CardTitle>
        <DatePickerWithRange
          date={dateRange}
          onDateChange={onDateRangeChange}
        />
      </CardHeader>
      <CardContent>
        <div className="flex flex-wrap gap-2 mb-4 px-4">
          {topics.map((topic, index) => (
            <Badge
              key={topic}
              variant={activeTopics[topic] ? "default" : "outline"}
              className="cursor-pointer truncate px-4"
              style={{
                backgroundColor: activeTopics[topic]
                  ? COLORS[index % COLORS.length]
                  : "transparent",
                color: activeTopics[topic] ? "white" : "inherit",
                borderColor: COLORS[index % COLORS.length],
              }}
              onClick={() => toggleTopic(topic)}
            >
              {formatTopicName(topic)}
            </Badge>
          ))}
        </div>
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="date"
                allowDuplicatedCategory={false}
                scale="auto"
                padding={{ left: 0, right: 0 }}
              />
              <YAxis
                allowDecimals={false}
                scale="auto"
                padding={{ top: 20, bottom: 20 }}
              />
              <Tooltip
                formatter={(value, name) => [value, formatTopicName(name)]}
              />
              {topics.map(
                (topic, index) =>
                  activeTopics[topic] && (
                    <Line
                      key={topic}
                      type="monotone"
                      dataKey={topic}
                      name={formatTopicName(topic)}
                      stroke={COLORS[index % COLORS.length]}
                      strokeWidth={2}
                    />
                  )
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </CardContent>
    </Card>
  );
}
