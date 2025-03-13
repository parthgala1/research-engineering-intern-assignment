"use client";

import { Card, CardContent } from "@/components/ui/card";

const roadmapData = {
  todo: [
    "Implement real-time data updates",
    "Add more social media platforms",
  ],
  inProgress: [
    "Enhanced AI analysis features",
    "Advanced filtering options",
  ],
  completed: [
    "Basic dashboard layout",
    "Data visualization components",
  ],
};

export default function RoadmapBoard() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {Object.entries(roadmapData).map(([status, items]) => (
        <Card key={status}>
          <CardContent className="p-4">
            <h3 className="font-semibold capitalize mb-3">{status}</h3>
            <ul className="space-y-2">
              {items.map((item, index) => (
                <li
                  key={index}
                  className="text-sm p-2 bg-muted rounded-md"
                >
                  {item}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}