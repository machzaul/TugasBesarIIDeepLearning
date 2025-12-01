import React from "react";
import "./attendance-button.css?v=2";

type Props = {
  onClick?: () => void;
  children?: React.ReactNode;
  className?: string;
  disabled?: boolean;
};

const AttendanceButton: React.FC<Props> = ({
  onClick,
  children = "Catat Kehadiran",
  className = "",
  disabled = false,
}) => {
  return (
    <div className={`ui-attendance-wrapper ${className}`}>
      <button
        type="button"
        className="ui-attendance-btn"
        onClick={onClick}
        disabled={disabled}
      >
        <span className="ui-attendance-text">{children}</span>
      </button>
    </div>
  );
};

export default AttendanceButton;
