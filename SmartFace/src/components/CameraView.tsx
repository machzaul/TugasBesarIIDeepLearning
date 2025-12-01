import { useRef, useEffect, useState } from "react";
import Webcam from "react-webcam";
import { Camera, CheckCircle2, User, Upload, X } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import StartAttendanceButton from "@/components/ui/StartAttendanceButton";
import { useToast } from "@/hooks/use-toast";

interface Prediction {
  label: string;
  confidence: number;
}

interface CameraViewProps {
  onDetection?: (detected: boolean, predictions?: Prediction[]) => void;
}

const CameraView = ({ onDetection }: CameraViewProps) => {
  const webcamRef = useRef<Webcam>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const [mode, setMode] = useState<"camera" | "upload">("camera");
  const [isScanning, setIsScanning] = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [annotatedImage, setAnnotatedImage] = useState<string | null>(null);
  const [faceImage, setFaceImage] = useState<string | null>(null);
  const [detectedName, setDetectedName] = useState<string | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);

  // Removed useEffect - camera activation now handled in handleModeChange

  const handleStartScan = async () => {
    if (mode === "upload" && !uploadedImage) {
      toast({
        title: "Belum ada foto",
        description: "Silakan upload foto terlebih dahulu",
        variant: "destructive",
      });
      return;
    }

    setIsScanning(true);
    setFaceDetected(false);

    try {
      // Get image data
      let imageData = "";

      if (mode === "camera" && webcamRef.current) {
        imageData = webcamRef.current.getScreenshot() || "";
      } else if (mode === "upload" && uploadedImage) {
        imageData = uploadedImage;
      }

      if (!imageData) {
        throw new Error("Tidak bisa mengambil gambar");
      }

      // Call backend API
      const response = await fetch("http://localhost:5000/recognize", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: imageData }),
      });

      if (!response.ok) {
        throw new Error("Gagal mengenali wajah");
      }

      const data = await response.json();

      if (data.success && data.predictions && data.predictions.length > 0) {
        setFaceDetected(true);
        setAnnotatedImage(data.annotated_image);
        setFaceImage(data.face_image);
        setDetectedName(data.predictions[0]?.label || null);
        onDetection?.(true, data.predictions);

        toast({
          title: "Wajah Terdeteksi!",
          description: `${
            data.predictions[0].label
          } - ${data.predictions[0].confidence.toFixed(2)}%`,
        });

        // Hasil deteksi tetap ditampilkan, tidak hilang otomatis
        setIsScanning(false);
      } else {
        throw new Error(data.error || "Tidak ada wajah terdeteksi");
      }
    } catch (error) {
      setIsScanning(false);
      setFaceDetected(false);
      setDetectedName(null);

      let errorMessage = "Terjadi kesalahan";
      if (error instanceof Error) {
        if (error.message.includes("fetch")) {
          errorMessage =
            "Tidak dapat terhubung ke server. Pastikan backend sudah berjalan di http://localhost:5000";
        } else {
          errorMessage = error.message;
        }
      }

      toast({
        title: "Gagal Mendeteksi",
        description: errorMessage,
        variant: "destructive",
      });
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith("image/")) {
      toast({
        title: "Format tidak valid",
        description: "Silakan upload file gambar (JPG, PNG, dll)",
        variant: "destructive",
      });
      return;
    }

    // Validate file size (max 5MB)
    if (file.size > 5 * 1024 * 1024) {
      toast({
        title: "File terlalu besar",
        description: "Ukuran file maksimal 5MB",
        variant: "destructive",
      });
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const result = e.target?.result as string;
      setUploadedImage(result);
      toast({
        title: "Foto berhasil diupload",
        description: "Klik tombol 'Mulai Absensi' untuk memindai wajah",
      });
    };
    reader.readAsDataURL(file);
  };

  const handleRemoveImage = () => {
    setUploadedImage(null);
    setIsScanning(false);
    setFaceDetected(false);
    setDetectedName(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleCameraReady = () => {
    setCameraActive(true);
    console.log("Camera ready and active");
    // Only show toast if camera was not previously active
    if (!cameraActive) {
      toast({
        title: "Kamera Siap",
        description: "Kamera berhasil diaktifkan",
      });
    }
  };

  const handleCameraError = (error: string | DOMException) => {
    console.error("Camera error:", error);
    setCameraActive(false);
    const errorMessage =
      error instanceof DOMException ? error.message : String(error);
    setCameraError(errorMessage);
    toast({
      title: "Gagal Mengakses Kamera",
      description: `Error: ${errorMessage}. Pastikan kamera tidak digunakan aplikasi lain dan izinkan akses kamera.`,
      variant: "destructive",
    });
  };

  const handleModeChange = (newMode: "camera" | "upload") => {
    setMode(newMode);
    setIsScanning(false);
    setFaceDetected(false);
    setDetectedName(null);
    setAnnotatedImage(null);
    if (newMode === "camera") {
      setUploadedImage(null);
      // Check camera permission before activating
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices
          .getUserMedia({ video: true })
          .then(() => {
            console.log("Camera permission granted");
            setCameraActive(false); // Will be set to true by onUserMedia
          })
          .catch((error) => {
            console.error("Camera permission denied:", error);
            toast({
              title: "Izin Kamera Ditolak",
              description: "Silakan izinkan akses kamera di browser Anda",
              variant: "destructive",
            });
          });
      }
    } else {
      setCameraActive(false);
    }
  };

  return (
    <Card className="relative overflow-hidden bg-card border-2 border-border">
      {/* Mode Toggle */}
      <div className="p-3 border-b border-border bg-muted/30">
        <div className="flex gap-2 max-w-md mx-auto">
          <StartAttendanceButton
            onClick={() => handleModeChange("upload")}
            className="flex-1 gap-2 transition-all duration-200 hover:scale-[1.02]"
            style={
              mode === "upload"
                ? {
                    backgroundColor: "hsl(var(--scan-primary))",
                    color: "white",
                  }
                : {
                    backgroundColor: "white",
                    color: "hsl(var(--scan-primary))",
                    border: "1px solid hsl(var(--scan-primary))",
                  }
            }
          >
            <Upload className="w-5 h-5" />
            Upload Foto
          </StartAttendanceButton>
          <StartAttendanceButton
            onClick={() => handleModeChange("camera")}
            className="flex-1 gap-2 transition-all duration-200 hover:scale-[1.02]"
            style={
              mode === "camera"
                ? {
                    backgroundColor: "hsl(var(--scan-primary))",
                    color: "white",
                  }
                : {
                    backgroundColor: "white",
                    color: "hsl(var(--scan-primary))",
                    border: "1px solid hsl(var(--scan-primary))",
                  }
            }
          >
            <Camera className="w-5 h-5" />
            Gunakan Kamera
          </StartAttendanceButton>
        </div>
      </div>

      <div
        className="relative bg-muted mx-auto"
        style={{ height: "280px", width: "400px", maxWidth: "100%" }}
      >
        {mode === "camera" ? (
          annotatedImage && faceDetected ? (
            <div className="w-full h-full relative">
              <img
                src={annotatedImage}
                alt="Detected Face"
                className="w-full h-full object-contain"
              />
              {/* Removed baked-in name mask (no white overlay) */}
              {detectedName && (
                <div className="absolute left-1/2 -translate-x-1/2 top-4">
                  <span
                    className="px-2 py-1 rounded-full text-xs font-semibold tracking-tight max-w-[90%] truncate"
                    style={{
                      backgroundColor: "hsl(var(--scan-success))",
                      color: "white",
                      boxShadow: "0 4px 12px rgba(34,197,94,0.14)",
                    }}
                  >
                    {detectedName}
                  </span>
                </div>
              )}
            </div>
          ) : cameraError ? (
            <div className="w-full h-full flex items-center justify-center">
              <div className="text-center space-y-4 p-6">
                <div className="w-16 h-16 mx-auto mb-4 rounded-full bg-destructive/10 flex items-center justify-center">
                  <X className="w-8 h-8 text-destructive" />
                </div>
                <div>
                  <p className="text-sm font-semibold mb-2 text-destructive">
                    Kamera Tidak Dapat Diakses
                  </p>
                  <p className="text-xs text-muted-foreground mb-4">
                    {cameraError}
                  </p>
                  <Button
                    onClick={() => {
                      setCameraError(null);
                      setCameraActive(false);
                      // Trigger camera activation again
                      handleModeChange("camera");
                    }}
                    className="px-3 py-2 text-xs rounded-full gap-2"
                    style={{
                      backgroundColor: "hsl(var(--scan-primary))",
                      color: "white",
                    }}
                  >
                    <Camera className="w-4 h-4" />
                    Coba Lagi
                  </Button>
                </div>
              </div>
            </div>
          ) : (
            <Webcam
              key={`webcam-${mode}`} // Force re-render when mode changes
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/jpeg"
              className="w-full h-full object-cover"
              videoConstraints={{
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: "user",
                frameRate: { ideal: 30 },
              }}
              onUserMedia={handleCameraReady}
              onUserMediaError={handleCameraError}
            />
          )
        ) : uploadedImage ? (
          faceDetected && annotatedImage ? (
            <div className="w-full h-full relative">
              <img
                src={annotatedImage}
                alt="Detected Face with BBox"
                className="w-full h-full object-contain"
              />
              {/* Removed baked-in name mask (no white overlay) */}
              {detectedName && (
                <div className="absolute left-1/2 -translate-x-1/2 top-4">
                  <span
                    className="px-2 py-1 rounded-full text-xs font-semibold tracking-tight max-w-[90%] truncate"
                    style={{
                      backgroundColor: "hsl(var(--scan-success))",
                      color: "white",
                      boxShadow: "0 4px 12px rgba(34,197,94,0.14)",
                    }}
                  >
                    {detectedName}
                  </span>
                </div>
              )}
            </div>
          ) : (
            <div className="relative w-full h-full">
              <img
                src={uploadedImage}
                alt="Uploaded"
                className="w-full h-full object-contain"
              />
              {/* temporary placeholder removed; name overlay will appear when detection completes */}
              <button
                onClick={handleRemoveImage}
                className="absolute top-3 right-3 p-1.5 rounded-full bg-destructive hover:bg-destructive/90 transition-colors"
                aria-label="Hapus foto"
              >
                <X className="w-4 h-4 text-destructive-foreground" />
              </button>
            </div>
          )
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <div className="text-center space-y-4">
              {/* upload icon removed per request */}
              <div>
                <p className="text-sm font-semibold mb-1">
                  Upload Foto untuk Absensi
                </p>
                <p className="text-xs text-muted-foreground mb-2">
                  Format: JPG, PNG (Max 5MB)
                </p>
                <Button
                  onClick={() => fileInputRef.current?.click()}
                  className="px-2.5 py-1.5 text-xs rounded-full gap-1.5 transition-all duration-200 hover:scale-105 hover:shadow-lg"
                  style={{
                    backgroundColor: "hsl(var(--scan-primary))",
                    color: "white",
                  }}
                >
                  <Upload className="w-3.5 h-3.5" />
                  Pilih Foto
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </div>
            </div>
          </div>
        )}

        {/* Face Detection Overlay */}
        {isScanning && (
          <div className="absolute inset-0 flex items-center justify-center">
            {/* Scanning Frame */}
            <div
              className="relative w-48 h-64 border-4 rounded-lg transition-all duration-300"
              style={{
                borderColor: faceDetected
                  ? "hsl(var(--scan-success))"
                  : "hsl(var(--scan-primary))",
                boxShadow: faceDetected
                  ? "0 0 40px hsl(var(--scan-success) / 0.5)"
                  : "0 0 30px hsl(var(--scan-primary) / 0.4)",
              }}
            >
              {/* Corner Markers */}
              <div
                className="absolute -top-1 -left-1 w-8 h-8 border-t-4 border-l-4 rounded-tl-lg"
                style={{
                  borderColor: faceDetected
                    ? "hsl(var(--scan-success))"
                    : "hsl(var(--scan-primary))",
                }}
              />
              <div
                className="absolute -top-1 -right-1 w-8 h-8 border-t-4 border-r-4 rounded-tr-lg"
                style={{
                  borderColor: faceDetected
                    ? "hsl(var(--scan-success))"
                    : "hsl(var(--scan-primary))",
                }}
              />
              <div
                className="absolute -bottom-1 -left-1 w-8 h-8 border-b-4 border-l-4 rounded-bl-lg"
                style={{
                  borderColor: faceDetected
                    ? "hsl(var(--scan-success))"
                    : "hsl(var(--scan-primary))",
                }}
              />
              <div
                className="absolute -bottom-1 -right-1 w-8 h-8 border-b-4 border-r-4 rounded-br-lg"
                style={{
                  borderColor: faceDetected
                    ? "hsl(var(--scan-success))"
                    : "hsl(var(--scan-primary))",
                }}
              />

              {/* Scanning Line */}
              {!faceDetected && (
                <div
                  className="absolute inset-x-0 h-1 bg-gradient-to-r from-transparent via-scan-primary to-transparent scan-animation"
                  style={{
                    background:
                      "linear-gradient(90deg, transparent, hsl(var(--scan-primary)), transparent)",
                  }}
                />
              )}

              {/* Detection Success */}
              {faceDetected && (
                <div className="absolute inset-0 flex items-center justify-center fade-in">
                  <div className="bg-background/90 backdrop-blur-sm rounded-full p-3">
                    <CheckCircle2
                      className="w-12 h-12"
                      style={{ color: "hsl(var(--scan-success))" }}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Status Indicator - hidden when annotatedImage is present so it doesn't overlap the box */}
        {!annotatedImage &&
          (isScanning ||
            faceDetected ||
            (mode === "camera" && !uploadedImage) ||
            (mode === "upload" && uploadedImage)) && (
            <div className="absolute top-3 left-3 flex items-center gap-2 bg-background/80 backdrop-blur-sm px-3 py-1.5 rounded-full">
              <div
                className={`w-2.5 h-2.5 rounded-full ${
                  isScanning
                    ? "bg-scan-primary animate-pulse"
                    : faceDetected
                    ? "bg-green-500"
                    : cameraError && mode === "camera"
                    ? "bg-red-500"
                    : cameraActive && mode === "camera"
                    ? "bg-green-500"
                    : uploadedImage && mode === "upload"
                    ? "bg-blue-500"
                    : "bg-yellow-500 animate-pulse"
                }`}
                style={
                  isScanning
                    ? { backgroundColor: "hsl(var(--scan-primary))" }
                    : {}
                }
              />
              <span className="text-xs font-medium">
                {faceDetected
                  ? "Wajah Terdeteksi"
                  : isScanning
                  ? "Memindai..."
                  : cameraActive && mode === "camera"
                  ? "Kamera Siap"
                  : cameraError && mode === "camera"
                  ? "Error Kamera"
                  : uploadedImage && mode === "upload"
                  ? "Foto Siap"
                  : mode === "camera"
                  ? "Mengaktifkan Kamera..."
                  : "Siap Upload"}
              </span>
            </div>
          )}
      </div>

      {/* Control Button */}
      <div className="p-4 flex justify-center">
        <StartAttendanceButton
          onClick={handleStartScan}
          loading={isScanning}
          disabled={isScanning}
        >
          {isScanning ? (
            <>
              <User className="w-5 h-5 animate-pulse" />
              Memindai Wajah...
            </>
          ) : (
            "Mulai Absensi"
          )}
        </StartAttendanceButton>
      </div>
    </Card>
  );
};

export default CameraView;
