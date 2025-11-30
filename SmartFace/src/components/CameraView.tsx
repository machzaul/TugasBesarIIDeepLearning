import { useRef, useEffect, useState } from "react";
import Webcam from "react-webcam";
import { Camera, CheckCircle2, User, Upload, X } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
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
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleModeChange = (newMode: "camera" | "upload") => {
    setMode(newMode);
    setIsScanning(false);
    setFaceDetected(false);
    if (newMode === "camera") {
      setUploadedImage(null);
    }
  };

  return (
    <Card className="relative overflow-hidden bg-card border-2 border-border">
      {/* Mode Toggle */}
      <div className="p-4 border-b border-border bg-muted/30">
        <div className="flex gap-2 max-w-md mx-auto">
          <Button
            onClick={() => handleModeChange("upload")}
            variant={mode === "upload" ? "default" : "outline"}
            className="flex-1 gap-2"
            style={
              mode === "upload"
                ? {
                    background: "var(--gradient-primary)",
                    color: "hsl(var(--primary-foreground))",
                  }
                : {}
            }
          >
            <Upload className="w-5 h-5" />
            Upload Foto
          </Button>
          <Button
            onClick={() => handleModeChange("camera")}
            variant={mode === "camera" ? "default" : "outline"}
            className="flex-1 gap-2"
            style={
              mode === "camera"
                ? {
                    background: "var(--gradient-primary)",
                    color: "hsl(var(--primary-foreground))",
                  }
                : {}
            }
          >
            <Camera className="w-5 h-5" />
            Gunakan Kamera
          </Button>
        </div>
      </div>

      <div
        className="relative bg-muted mx-auto"
        style={{ height: "350px", width: "500px", maxWidth: "100%" }}
      >
        {mode === "camera" ? (
          annotatedImage && faceDetected ? (
            <div className="w-full h-full">
              <img
                src={annotatedImage}
                alt="Detected Face"
                className="w-full h-full object-contain"
              />
            </div>
          ) : (
            <Webcam
              ref={webcamRef}
              audio={false}
              screenshotFormat="image/jpeg"
              className="w-full h-full object-cover"
              videoConstraints={{
                facingMode: "user",
              }}
            />
          )
        ) : uploadedImage ? (
          faceDetected && annotatedImage ? (
            <div className="w-full h-full">
              <img
                src={annotatedImage}
                alt="Detected Face with BBox"
                className="w-full h-full object-contain"
              />
            </div>
          ) : (
            <div className="relative w-full h-full">
              <img
                src={uploadedImage}
                alt="Uploaded"
                className="w-full h-full object-contain"
              />
              <button
                onClick={handleRemoveImage}
                className="absolute top-4 right-4 p-2 rounded-full bg-destructive hover:bg-destructive/90 transition-colors"
                aria-label="Hapus foto"
              >
                <X className="w-5 h-5 text-destructive-foreground" />
              </button>
            </div>
          )
        ) : (
          <div className="w-full h-full flex items-center justify-center">
            <div className="text-center space-y-4">
              <div className="w-24 h-24 mx-auto rounded-full bg-primary/10 flex items-center justify-center">
                <Upload className="w-12 h-12 text-primary" />
              </div>
              <div>
                <p className="text-lg font-semibold mb-2">
                  Upload Foto untuk Absensi
                </p>
                <p className="text-sm text-muted-foreground mb-4">
                  Format: JPG, PNG (Max 5MB)
                </p>
                <Button
                  onClick={() => fileInputRef.current?.click()}
                  className="gap-2"
                  style={{
                    background: "var(--gradient-primary)",
                    color: "hsl(var(--primary-foreground))",
                  }}
                >
                  <Upload className="w-5 h-5" />
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
              className="relative w-64 h-80 border-4 rounded-lg transition-all duration-300"
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
                  <div className="bg-background/90 backdrop-blur-sm rounded-full p-4">
                    <CheckCircle2
                      className="w-16 h-16"
                      style={{ color: "hsl(var(--scan-success))" }}
                    />
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Status Indicator */}
        {(isScanning || faceDetected || mode === "camera") && (
          <div className="absolute top-4 left-4 flex items-center gap-2 bg-background/80 backdrop-blur-sm px-4 py-2 rounded-full">
            <div
              className={`w-3 h-3 rounded-full ${
                isScanning
                  ? "bg-scan-primary animate-pulse"
                  : "bg-muted-foreground"
              }`}
              style={
                isScanning
                  ? { backgroundColor: "hsl(var(--scan-primary))" }
                  : {}
              }
            />
            <span className="text-sm font-medium">
              {faceDetected
                ? "Wajah Terdeteksi"
                : isScanning
                ? "Memindai..."
                : "Siap"}
            </span>
          </div>
        )}
      </div>

      {/* Control Button */}
      <div className="p-6 flex justify-center">
        <button
          onClick={handleStartScan}
          disabled={isScanning}
          className="flex items-center gap-3 px-8 py-4 rounded-xl font-semibold text-lg transition-all duration-300 disabled:opacity-50"
          style={{
            background: isScanning
              ? "hsl(var(--muted))"
              : "var(--gradient-primary)",
            color: "hsl(var(--primary-foreground))",
            boxShadow: !isScanning
              ? "0 8px 24px hsl(var(--scan-primary) / 0.3)"
              : "none",
          }}
        >
          {isScanning ? (
            <>
              <User className="w-6 h-6 animate-pulse" />
              Memindai Wajah...
            </>
          ) : (
            <>
              <Camera className="w-6 h-6" />
              Mulai Absensi
            </>
          )}
        </button>
      </div>
    </Card>
  );
};

export default CameraView;
