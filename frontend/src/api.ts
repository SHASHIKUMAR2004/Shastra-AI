import type {
  GenerationJob,
  ProjectChatMessage,
  ProjectChatResponse,
  ProjectFilesResponse,
  ProjectSummary,
} from "./types";

const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL;
const API_BASE_URL =
  configuredApiBaseUrl !== undefined
    ? configuredApiBaseUrl.replace(/\/$/, "")
    : import.meta.env.PROD
      ? ""
      : `${window.location.protocol}//${window.location.hostname}:8000`;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
    ...init,
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export async function checkHealth(): Promise<boolean> {
  try {
    await request<{ status: string }>("/api/health");
    return true;
  } catch {
    return false;
  }
}

export interface GenerationAttachment {
  name: string;
  mime_type: string;
  size: number;
  content_text?: string;
}

export async function createGeneration(
  prompt: string,
  recursionLimit: number,
  attachments: GenerationAttachment[] = [],
) {
  return request<{ job_id: string; status: string }>("/api/generations", {
    method: "POST",
    body: JSON.stringify({
      prompt,
      recursion_limit: recursionLimit,
      attachments,
    }),
  });
}

export function getGeneration(jobId: string): Promise<GenerationJob> {
  return request<GenerationJob>(`/api/generations/${jobId}`);
}

export function listGenerations(): Promise<GenerationJob[]> {
  return request<GenerationJob[]>("/api/generations");
}

export function listProjects(): Promise<ProjectSummary[]> {
  return request<ProjectSummary[]>("/api/projects");
}

export function listProjectFiles(projectName: string): Promise<ProjectFilesResponse> {
  return request<ProjectFilesResponse>(`/api/projects/${projectName}/files`);
}

export function listProjectChat(projectName: string): Promise<ProjectChatMessage[]> {
  return request<ProjectChatMessage[]>(`/api/projects/${projectName}/chat`);
}

export function chatWithProject(
  projectName: string,
  message: string,
  attachments: GenerationAttachment[] = [],
): Promise<ProjectChatResponse> {
  return request<ProjectChatResponse>(`/api/projects/${projectName}/chat`, {
    method: "POST",
    body: JSON.stringify({
      message,
      attachments,
    }),
  });
}

export function projectDownloadUrl(projectName: string): string {
  return `${API_BASE_URL}/api/projects/${encodeURIComponent(projectName)}/download`;
}
