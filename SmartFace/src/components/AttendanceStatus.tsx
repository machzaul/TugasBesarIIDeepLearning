import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { CheckCircle2, Clock, User, Award } from "lucide-react";

interface Prediction {
  label: string;
  confidence: number;
}

interface AttendanceStatusProps {
  isPresent: boolean;
  studentName?: string;
  timestamp?: string;
  confidence?: number;
  predictions?: Prediction[];
  onMarkAttendance?: () => void;
}

const AttendanceStatus = ({
  isPresent,
  studentName,
  timestamp,
  confidence = 0,
  predictions = [],
  onMarkAttendance,
}: AttendanceStatusProps) => {
  const currentTime =
    timestamp ||
    new Date().toLocaleTimeString("id-ID", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

  return (
    <Card className="p-8">
      <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
        <div
          className="w-1 h-8 rounded-full"
          style={{ backgroundColor: "hsl(var(--scan-primary))" }}
        />
        Status Kehadiran
      </h2>

      <div className="space-y-6">
        {isPresent ? (
          <div className="fade-in">
            {/* Status Badge */}
            <div
              className="flex items-center justify-center gap-3 p-6 rounded-2xl mb-6"
              style={{
                background:
                  "linear-gradient(135deg, hsl(var(--scan-success) / 0.1), hsl(var(--scan-success) / 0.2))",
                border: "2px solid hsl(var(--scan-success) / 0.3)",
              }}
            >
              <CheckCircle2
                className="w-12 h-12"
                style={{ color: "hsl(var(--scan-success))" }}
              />
              <div className="text-left">
                <p
                  className="text-3xl font-bold"
                  style={{ color: "hsl(var(--scan-success))" }}
                >
                  HADIR
                </p>
                {studentName && (
                  <p className="text-2xl font-bold text-foreground mt-1">
                    {studentName}
                  </p>
                )}
              </div>
            </div>

            {/* Confidence */}
            {confidence > 0 && (
              <div className="p-4 rounded-lg bg-muted/50 mb-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-muted-foreground">
                    Confidence
                  </span>
                  <span className="text-lg font-bold">
                    {confidence.toFixed(2)}%
                  </span>
                </div>
                <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{
                      width: `${confidence}%`,
                      background: "var(--gradient-primary)",
                    }}
                  />
                </div>
              </div>
            )}

            {/* Top 3 Predictions */}
            {predictions.length > 0 && (
              <div className="mb-6">
                <h3 className="text-sm font-semibold mb-3 flex items-center gap-2">
                  <Award className="w-4 h-4" />
                  Top 3 Prediksi
                </h3>
                <div className="space-y-2">
                  {predictions.map((pred, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between p-3 rounded-lg bg-muted/30"
                    >
                      <div className="flex items-center gap-2">
                        <span className="w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center text-xs font-bold">
                          {idx + 1}
                        </span>
                        <span className="text-sm">{pred.label}</span>
                      </div>
                      <span className="text-sm font-semibold">
                        {pred.confidence.toFixed(2)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Time Info */}
            <div className="flex items-center justify-center gap-2 text-muted-foreground mb-6">
              <Clock className="w-5 h-5" />
              <span className="text-sm">Terdeteksi pada {currentTime}</span>
            </div>

            {/* Mark Attendance Button */}
            <Button
              onClick={onMarkAttendance}
              className="w-full gap-2 py-6"
              style={{
                background: "var(--gradient-primary)",
                color: "hsl(var(--primary-foreground))",
              }}
            >
              <CheckCircle2 className="w-5 h-5" />
              Catat Kehadiran
            </Button>
          </div>
        ) : (
          <div className="text-center py-12">
            <div className="w-24 h-24 mx-auto mb-6 rounded-full bg-muted flex items-center justify-center">
              <CheckCircle2 className="w-12 h-12 text-muted-foreground" />
            </div>
            <p className="text-xl font-semibold text-muted-foreground mb-2">
              Belum Ada Kehadiran
            </p>
            <p className="text-sm text-muted-foreground">
              Upload foto atau gunakan kamera untuk absensi
            </p>
          </div>
        )}
      </div>
    </Card>
  );
};

export default AttendanceStatus;
