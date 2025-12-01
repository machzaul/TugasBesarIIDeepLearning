import React from "react";
import "./start-attendance-button.css";

type Props = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  loading?: boolean;
  children?: React.ReactNode;
};

const StartAttendanceButton: React.FC<Props> = ({
  children = "Mulai Absensi",
  loading = false,
  disabled,
  className,
  ...rest
}) => {
  const classNames = ["button", className].filter(Boolean).join(" ");

  return (
    <button className={classNames} disabled={disabled || loading} {...rest}>
      <span className="label">{loading ? "Memindai..." : children}</span>
    </button>
  );
};

export default StartAttendanceButton;
