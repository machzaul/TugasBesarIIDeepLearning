import React from "react";
import { Card } from "@/components/ui/card";
import { Clock } from "lucide-react";

interface AttendanceRecord {
  id: number;
  label: string;
  date: string;
  time: string;
  confidence?: number;
}

type Props = {
  attendanceList?: AttendanceRecord[];
};

const AttendanceHistory: React.FC<Props> = ({ attendanceList = [] }) => {
  return (
    <Card className="p-4">
      <h3 className="text-base font-normal mb-3 flex items-center gap-2">
        <Clock className="w-4 h-4" />
        Riwayat Absensi
      </h3>

      {attendanceList.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          Belum ada absensi hari ini.
        </p>
      ) : (
        <div className="space-y-2 max-h-[60vh] overflow-auto pr-2">
          {attendanceList.map((rec) => (
            <div
              key={rec.id}
              className="flex items-center justify-between px-3 py-2 rounded-lg bg-muted/30 border border-border/50"
            >
              <div className="flex items-center gap-2 min-w-0">
                <div className="text-sm font-normal truncate">{rec.label}</div>
              </div>
              <div className="text-sm font-normal text-green-600 flex-shrink-0">
                {rec.time}
              </div>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
};

export default AttendanceHistory;
