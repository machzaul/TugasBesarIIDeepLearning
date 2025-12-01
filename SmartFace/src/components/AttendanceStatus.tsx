import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import AttendanceButton from "@/components/ui/AttendanceButton";
import { CheckCircle2, Clock, User, Award, IdCard } from "lucide-react";

interface Prediction {
  label: string;
  confidence: number;
}

interface StudentInfo {
  name: string;
  nim: string;
  kelas: string;
}

interface AttendanceStatusProps {
  isPresent: boolean;
  studentName?: string;
  studentInfo?: StudentInfo | null;
  timestamp?: string;
  confidence?: number;
  predictions?: Prediction[];
  onMarkAttendance?: () => void;
}

const AttendanceStatus = ({
  isPresent,
  studentName,
  studentInfo,
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
    <Card className="pt-3 px-6 pb-6">
      <h2 className="text-xl font-medium mb-2 flex items-center gap-2">
        <div
          className="w-1 h-5 rounded-full"
          style={{ backgroundColor: "hsl(var(--scan-primary))" }}
        />
        Status Kehadiran
      </h2>

      <div className="space-y-4">
        {isPresent ? (
          <div className="fade-in">
            {/* Status Badge */}
            <div
              className="px-4 py-3 rounded-xl mb-4 border-2 flex flex-col gap-1"
              style={{
                backgroundColor: "hsl(var(--scan-success) / 0.1)",
                borderColor: "hsl(var(--scan-success) / 0.4)",
              }}
            >
              {/* Header dengan Icon dan Status */}
              <div className="flex items-center gap-2 mb-1">
                <CheckCircle2
                  className="w-5 h-5 flex-shrink-0"
                  style={{ color: "hsl(var(--scan-success))" }}
                />
                <p
                  className="text-lg font-bold leading-tight tracking-wide"
                  style={{ color: "hsl(var(--scan-success))" }}
                >
                  Hadir
                </p>
              </div>
              {/* Student Info */}
              {studentName && (
                <div className="flex flex-col gap-1">
                  <p className="text-base font-semibold text-foreground mb-0">
                    {studentName}
                  </p>

                  {studentInfo && (
                    <div className="pt-1 border-t border-border/50 flex flex-col gap-1">
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-muted-foreground">
                          NIM:
                        </span>
                        <span className="text-sm font-medium text-foreground">
                          {studentInfo.nim}
                        </span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-muted-foreground">
                          Kelas:
                        </span>
                        <span className="text-sm font-medium text-foreground">
                          {studentInfo.kelas}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Confidence */}
            {confidence > 0 && (
              <div className="mb-3">
                <div className="flex items-center justify-between text-sm mb-1">
                  <span className="text-sm text-muted-foreground">
                    Confidence
                  </span>
                  <span className="text-sm font-semibold text-foreground">
                    {Math.round(confidence)}%
                  </span>
                </div>
                <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-300"
                    style={{
                      width: `${confidence}%`,
                      backgroundColor: "hsl(var(--scan-primary))",
                    }}
                  />
                </div>
              </div>
            )}

            {/* Top 3 Predictions */}
            {predictions.length > 0 && (
              <div className="mb-4">
                <h3 className="text-sm font-semibold mb-3 text-foreground">
                  Top 3 Prediksi
                </h3>
                <div className="space-y-2">
                  {predictions.map((pred, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between px-3 py-2 rounded-lg bg-muted/30 border border-border/50 hover:border-border transition-colors"
                    >
                      <div className="flex items-center gap-2 flex-1 min-w-0">
                        <div
                          className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold text-white flex-shrink-0"
                          style={{
                            backgroundColor:
                              idx === 0
                                ? "hsl(var(--scan-primary))"
                                : idx === 1
                                ? "hsl(var(--scan-primary) / 0.8)"
                                : "hsl(var(--scan-primary) / 0.6)",
                          }}
                        >
                          {idx + 1}
                        </div>
                        <span className="text-sm font-normal text-foreground truncate">
                          {pred.label}
                        </span>
                      </div>
                      <span
                        className="text-sm font-normal ml-2 flex-shrink-0"
                        style={{
                          color: "hsl(var(--scan-primary))",
                          fontWeight: 400,
                        }}
                      >
                        {pred.confidence.toFixed(2)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Time Info */}

            {/* (Removed duplicate time info; history panel shows records) */}

            {/* Mark Attendance Button */}
            <div className="flex justify-center mt-4">
              <AttendanceButton onClick={onMarkAttendance} />
            </div>
          </div>
        ) : (
          <div className="text-center py-12">
            <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-muted/40 flex items-center justify-center">
              <CheckCircle2 className="w-5 h-5 text-muted-foreground/40" />
            </div>
            <p className="text-base font-semibold text-foreground mb-1">
              Belum Ada Kehadiran
            </p>
            <p className="text-xs text-muted-foreground leading-relaxed">
              Aktifkan kamera atau upload foto untuk memulai proses absensi
            </p>
          </div>
        )}

        {/* Attendance history removed from this panel per user request */}
      </div>
    </Card>
  );
};

export default AttendanceStatus;
