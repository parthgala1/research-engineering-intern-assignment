"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { LineChart, BarChart3, PieChart } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

export default function DashboardHeader() {
  const [dashboardData, setDashboardData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch dashboard data from the API
    // // // fetch('http://localhost:8000/api/dashboard-data', {
    // // //   headers: {
    // // //     'Accept': 'application/json',
    // // //   },
    // // // })
    // // //   .then((response) => {
    // // //     if (!response.ok) {
    // // //       throw new Error('Network response was not ok');
    // // //     }
    // // //     return response.json();
    // // //   })
    // // //   .then((data) => {
    // // //     setDashboardData(data);
    // // //     setLoading(false);
    // // //   })
    // // //   .catch((error) => {
    // // //     console.error("Error loading dashboard data:", error);
    // // //     // Fallback data in case of error
    // // //     setDashboardData({
    // // //       title: "Social Media Analysis Dashboard",
    // // //       sections: {
    // // //         trends: "Trend Analysis",
    // // //         contributors: "Top Contributors",
    // // //         hashtags: "Popular Hashtags"
    // // //       },
    // // //       exportButton: "Export Data",
    // // //       exportOptions: [
    // // //         "Export as CSV",
    // // //         "Export as PDF",
    // // //         "Share Dashboard"
    // // //       ],
    // // //     });
    // //     setLoading(false);
    //   });
  }, []);

  return (
    <header className="border-b">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h1 className="text-2xl font-bold">
              {loading
                ? "Loading..."
                : dashboardData?.title || "Social Media Analysis"}
            </h1>
            <div className="hidden md:flex items-center space-x-2">
              <Link href="/trends">
                <Button variant="ghost" size="sm">
                  <LineChart className="h-4 w-4 mr-2" />
                  {dashboardData?.sections?.trends || "Trends"}
                </Button>
              </Link>
              <Link href="/contributors">
                <Button variant="ghost" size="sm">
                  <BarChart3 className="h-4 w-4 mr-2" />
                  {dashboardData?.sections?.contributors || "Contributors"}
                </Button>
              </Link>
              <Link href="/hashtags">
                <Button variant="ghost" size="sm">
                  <PieChart className="h-4 w-4 mr-2" />
                  {dashboardData?.sections?.hashtags || "Hashtags"}
                </Button>
              </Link>
            </div>
          </div>

          <div className="flex items-center space-x-4">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline">
                  {dashboardData?.exportButton || "Export"}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                {dashboardData?.exportOptions?.map((option, index) => (
                  <DropdownMenuItem key={index}>{option}</DropdownMenuItem>
                )) || (
                  <>
                    <DropdownMenuItem>Export as CSV</DropdownMenuItem>
                    <DropdownMenuItem>Export as PDF</DropdownMenuItem>
                    <DropdownMenuItem>Share Dashboard</DropdownMenuItem>
                  </>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
      </div>
    </header>
  );
}
