export interface StudentInfo {
  name: string;
  nim: string;
  kelas: string;
}

export interface StudentDatabase {
  [key: string]: StudentInfo;
}

export const parseStudentCSV = (csvContent: string): StudentDatabase => {
  const database: StudentDatabase = {};
  const lines = csvContent.trim().split("\n");

  // Skip header row
  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (!line.trim()) continue;

    // Parse CSV line - handle quoted fields
    const fields = parseCsvLine(line);
    if (fields.length >= 4) {
      const filename = fields[0];
      const label = fields[1];
      const nim = fields[2];
      const kelas = fields[3];

      // Store unique student info by label (name)
      if (!database[label]) {
        database[label] = {
          name: label,
          nim,
          kelas,
        };
      }
    }
  }

  return database;
};

// Helper function to parse CSV line properly
const parseCsvLine = (line: string): string[] => {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;

  for (let i = 0; i < line.length; i++) {
    const char = line[i];

    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === "," && !inQuotes) {
      result.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }

  result.push(current.trim());
  return result;
};
