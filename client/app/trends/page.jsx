"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import TimeSeriesChart from "@/components/charts/TimeSeriesChart";
import DashboardHeader from "@/components/dashboard/DashboardHeader";

export default function TrendsPage() {
  const [dateRange, setDateRange] = useState({
    start: new Date(2024, 0, 1),
    end: new Date(),
  });

  return (
    <div className="min-h-screen bg-background">
      <DashboardHeader />
      <main className="container mx-auto px-4 py-6">
        <h1 className="text-3xl font-bold mb-6">Trend Analysis</h1>
        <div className="grid gap-6">
          <Card className="col-span-full">
            <CardHeader>
              <CardTitle>Engagement Over Time</CardTitle>
            </CardHeader>
            <CardContent>
              <TimeSeriesChart
                dateRange={dateRange}
                onDateRangeChange={setDateRange}
              />
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}