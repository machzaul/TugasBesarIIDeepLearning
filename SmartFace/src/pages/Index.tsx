import { useState, useEffect } from "react";
import { Calendar } from "lucide-react";
import CameraView from "@/components/CameraView";
import AttendanceStatus from "@/components/AttendanceStatus";
import { useToast } from "@/hooks/use-toast";
import { parseStudentCSV, StudentInfo, StudentDatabase } from "@/lib/studentData";
import labelsData from "@/labels-nim.csv?raw";

interface Prediction {
  label: string;
  confidence: number;
}

const Index = () => {
  const { toast } = useToast();
  const [isPresent, setIsPresent] = useState(false);
  const [studentName, setStudentName] = useState<string>("");
  const [studentInfo, setStudentInfo] = useState<StudentInfo | null>(null);
  const [attendanceTime, setAttendanceTime] = useState<string>("");
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [confidence, setConfidence] = useState<number>(0);
  const [studentDatabase, setStudentDatabase] = useState<StudentDatabase>({});

  // Parse CSV and create database
  useEffect(() => {
    const database = parseStudentCSV(labelsData);
    setStudentDatabase(database);
  }, []);

  const handleDetection = (detected: boolean, preds?: Prediction[]) => {
    if (detected && preds && preds.length > 0) {
      const topPrediction = preds[0];
      const info = studentDatabase[topPrediction.label];

      setIsPresent(true);
      setStudentName(topPrediction.label);
      setStudentInfo(info || null);
      setConfidence(topPrediction.confidence);
      setPredictions(preds);
      setAttendanceTime(
        new Date().toLocaleTimeString("id-ID", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        })
      );

      toast({
        title: "Wajah Terdeteksi!",
        description: `${topPrediction.label} (${topPrediction.confidence.toFixed(
          2
        )}%)`,
      });
    }
  };

  const handleMarkAttendance = async () => {
    if (!studentName || !confidence) return;

    try {
      const response = await fetch("http://localhost:5000/mark-attendance", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          label: studentName,
          confidence: confidence,
          image: null, // Could include face image here
        }),
      });

      const data = await response.json();

      if (data.success) {
        toast({
          title: "Absensi Berhasil!",
          description: data.message,
          variant: "default",
        });
      } else {
        toast({
          title: "Gagal!",
          description: data.message,
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "Error!",
        description: "Gagal menghubungi server",
        variant: "destructive",
      });
    }
  };

  const currentDate = new Date().toLocaleDateString("id-ID", {
    weekday: "long",
    year: "numeric",
    month: "long",
    day: "numeric",
  });

  return (
    <div className="h-screen flex flex-col overflow-hidden bg-background">
      {/* Header */}
      <header className="border-b border-border bg-card/50 backdrop-blur-sm flex-shrink-0">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <img src="/Scan.png" alt="Logo" className="w-16 h-16" />
              <div>
                <h1 className="text-2xl font-bold">Deep Learning RA</h1>
                <p className="text-sm text-muted-foreground">
                  Sistem Absensi Face Recognition
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-muted">
              <Calendar className="w-5 h-5 text-muted-foreground" />
              <span className="text-sm font-medium">{currentDate}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 container mx-auto px-6 py-8 overflow-auto">
        <div className="grid lg:grid-cols-3 gap-8 h-full">
          {/* Left Column - Camera */}
          <div className="lg:col-span-2">
            <CameraView onDetection={handleDetection} />
          </div>

          {/* Right Column - Status */}
          <div className="lg:col-span-1">
            <AttendanceStatus
              isPresent={isPresent}
              studentName={studentName}
              studentInfo={studentInfo}
              timestamp={attendanceTime}
              confidence={confidence}
              predictions={predictions}
              onMarkAttendance={handleMarkAttendance}
            />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border py-4 flex-shrink-0">
        <div className="container mx-auto px-6 text-center text-sm text-muted-foreground">
          <p>© 2024 Deep Learning RA - Face Recognition Attendance System</p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
