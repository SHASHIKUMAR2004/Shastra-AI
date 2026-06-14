export type JobStatus = "queued" | "running" | "succeeded" | "failed";

export interface GenerationJob {
  id: string;
  prompt: string;
  status: JobStatus;
  created_at: string;
  updated_at: string;
  project_dir?: string | null;
  final_status?: string | null;
  test_summary?: string | null;
  issues: string[];
  logs: string;
  error?: string | null;
}

export interface ProjectSummary {
  name: string;
  path: string;
  updated_at: string;
  file_count: number;
}

export interface FileSummary {
  path: string;
  size: number;
  updated_at: string;
}

export interface ProjectFilesResponse {
  project: ProjectSummary;
  files: FileSummary[];
}

export interface ProjectChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
}

export interface ProjectValidationResult {
  passed: boolean;
  summary: string;
  issues: string[];
  commands_run: string[];
  stdout: string;
  stderr: string;
}

export interface ProjectChatResponse {
  message: ProjectChatMessage;
  history: ProjectChatMessage[];
  changed_files: string[];
  validation: ProjectValidationResult;
}
