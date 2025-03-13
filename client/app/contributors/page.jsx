"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import TopContributorsChart from "@/components/charts/TopContributorsChart";
import DashboardHeader from "@/components/dashboard/DashboardHeader";
import DataTable from "@/components/dashboard/DataTable";

export default function ContributorsPage() {
  const [dateRange, setDateRange] = useState({
    start: new Date(2024, 0, 1),
    end: new Date(),
  });

  return (
    <div className="min-h-screen bg-background">
      <DashboardHeader />
      <main className="container mx-auto px-4 py-6">
        <h1 className="text-3xl font-bold mb-6">Top Contributors</h1>
        <div className="grid gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Contribution Analysis</CardTitle>
            </CardHeader>
            <CardContent>
              <TopContributorsChart dateRange={dateRange} />
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Contributors List</CardTitle>
            </CardHeader>
            <CardContent>
              <DataTable dateRange={dateRange} />
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}