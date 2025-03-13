"use client";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

const mockData = [
  { id: 1, platform: "Reddit", topic: "AI", engagement: 1200, sentiment: "Positive" },
  { id: 2, platform: "Twitter", topic: "Tech", engagement: 850, sentiment: "Neutral" },
  { id: 3, platform: "Facebook", topic: "News", engagement: 650, sentiment: "Negative" },
];

export default function DataTable({ dateRange, selectedSubreddit, selectedHashtag }) {
  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Platform</TableHead>
            <TableHead>Topic</TableHead>
            <TableHead>Engagement</TableHead>
            <TableHead>Sentiment</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {mockData.map((row) => (
            <TableRow key={row.id}>
              <TableCell>{row.platform}</TableCell>
              <TableCell>{row.topic}</TableCell>
              <TableCell>{row.engagement}</TableCell>
              <TableCell>{row.sentiment}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}