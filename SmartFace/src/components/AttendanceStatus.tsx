import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
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
              className="p-8 rounded-3xl mb-6 border-2"
              style={{
                background:
                  "linear-gradient(135deg, hsl(var(--scan-success) / 0.08), hsl(var(--scan-success) / 0.15))",
                borderColor: "hsl(var(--scan-success) / 0.4)",
              }}
            >
              {/* Header dengan Icon dan Status */}
              <div className="flex items-start gap-4 mb-4">
                <CheckCircle2
                  className="w-14 h-14 flex-shrink-0 mt-1"
                  style={{ color: "hsl(var(--scan-success))" }}
                />
                <div>
                  <p
                    className="text-4xl font-bold leading-tight"
                    style={{ color: "hsl(var(--scan-success))" }}
                  >
                    HADIR
                  </p>
                </div>
              </div>

              {/* Student Info */}
              {studentName && (
                <div className="space-y-3 ml-0">
                  {/* Nama */}
                  <div>
                    <p className="text-2xl font-bold text-foreground">
                      {studentName}
                    </p>
                  </div>

                  {/* NIM dan Kelas */}
                  {studentInfo && (
                    <div className="space-y-2 pt-2 border-t border-border/50">
                      <div className="flex items-center gap-3">
                        <IdCard className="w-5 h-5 text-muted-foreground flex-shrink-0" />
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-muted-foreground">NIM:</span>
                          <span className="text-base font-semibold text-foreground">
                            {studentInfo.nim}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-3">
                        <User className="w-5 h-5 text-muted-foreground flex-shrink-0" />
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-muted-foreground">Kelas:</span>
                          <span className="text-base font-semibold text-foreground">
                            {studentInfo.kelas}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Confidence */}
            {confidence > 0 && (
              <div className="p-4 rounded-2xl bg-gradient-to-r from-blue-50 to-transparent mb-6 border border-blue-100">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Award className="w-4 h-4 text-blue-600" />
                    <span className="text-sm font-semibold text-foreground">
                      Confidence
                    </span>
                  </div>
                  <span className="text-lg font-bold text-blue-600">
                    {confidence.toFixed(2)}%
                  </span>
                </div>
                <div className="w-full h-2.5 bg-blue-100 rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all duration-500"
                    style={{
                      width: `${confidence}%`,
                      background: "linear-gradient(90deg, hsl(var(--scan-primary)), hsl(var(--scan-success)))",
                    }}
                  />
                </div>
              </div>
            )}

            {/* Top 3 Predictions */}
            {predictions.length > 0 && (
              <div className="mb-6">
                <h3 className="text-sm font-semibold mb-3 flex items-center gap-2 text-foreground">
                  <Award className="w-4 h-4" />
                  Top 3 Prediksi
                </h3>
                <div className="space-y-2">
                  {predictions.map((pred, idx) => (
                    <div
                      key={idx}
                      className="flex items-center justify-between p-4 rounded-xl bg-gradient-to-r from-muted/40 to-transparent border border-border/50 hover:border-border transition-colors"
                    >
                      <div className="flex items-center gap-3 flex-1 min-w-0">
                        <div 
                          className="w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold text-white flex-shrink-0"
                          style={{
                            background: idx === 0 
                              ? "linear-gradient(135deg, hsl(var(--scan-primary)), hsl(var(--scan-success)))"
                              : idx === 1
                              ? "hsl(var(--scan-primary))"
                              : "hsl(var(--scan-primary) / 0.6)"
                          }}
                        >
                          {idx + 1}
                        </div>
                        <span className="text-sm font-medium text-foreground truncate">
                          {pred.label}
                        </span>
                      </div>
                      <span className="text-sm font-bold ml-2 flex-shrink-0" style={{ color: "hsl(var(--scan-primary))" }}>
                        {pred.confidence.toFixed(2)}%
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Time Info */}
            <div className="flex items-center justify-center gap-2 text-muted-foreground mb-6 py-3 bg-muted/30 rounded-lg">
              <Clock className="w-4 h-4" />
              <span className="text-sm font-medium">Terdeteksi pada {currentTime}</span>
            </div>

            {/* Mark Attendance Button */}
            <Button
              onClick={onMarkAttendance}
              className="w-full gap-2 py-6 text-base font-semibold rounded-xl transition-all hover:shadow-lg"
              style={{
                background: "linear-gradient(135deg, hsl(var(--scan-primary)), hsl(var(--scan-success)))",
                color: "hsl(var(--primary-foreground))",
              }}
            >
              <CheckCircle2 className="w-5 h-5" />
              Catat Kehadiran
            </Button>
          </div>
        ) : (
          <div className="text-center py-16">
            <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-gradient-to-br from-muted/50 to-muted flex items-center justify-center">
              <CheckCircle2 className="w-10 h-10 text-muted-foreground/40" />
            </div>
            <p className="text-lg font-semibold text-foreground mb-2">
              Belum Ada Kehadiran
            </p>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Aktifkan kamera atau upload foto untuk memulai proses absensi
            </p>
          </div>
        )}
      </div>
    </Card>
  );
};

export default AttendanceStatus;
