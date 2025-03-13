"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import HashtagsChart from "@/components/charts/HashtagsChart";
import DashboardHeader from "@/components/dashboard/DashboardHeader";
import { Badge } from "@/components/ui/badge";

export default function HashtagsPage() {
  const [dateRange, setDateRange] = useState({
    start: new Date(2024, 0, 1),
    end: new Date(),
  });
  const [selectedHashtag, setSelectedHashtag] = useState(null);

  return (
    <div className="min-h-screen bg-background">
      <DashboardHeader />
      <main className="container mx-auto px-4 py-6">
        <h1 className="text-3xl font-bold mb-6">Hashtag Analysis</h1>
        <div className="grid gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Popular Hashtags</CardTitle>
            </CardHeader>
            <CardContent>
              <HashtagsChart
                dateRange={dateRange}
                onHashtagSelect={setSelectedHashtag}
              />
            </CardContent>
          </Card>

          {selectedHashtag && (
            <Card>
              <CardHeader>
                <CardTitle>
                  #{selectedHashtag}
                  <Badge className="ml-2" variant="secondary">
                    Selected Hashtag
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <TimeSeriesChart
                  dateRange={dateRange}
                  selectedHashtag={selectedHashtag}
                />
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </div>
  );
}