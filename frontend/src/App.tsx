import {
  ChangeEvent,
  ClipboardEvent,
  FormEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  Code2,
  Download,
  Files,
  FolderGit2,
  Image,
  Loader2,
  Moon,
  PanelLeftClose,
  PanelLeftOpen,
  Paperclip,
  Play,
  Plus,
  RefreshCcw,
  SlidersHorizontal,
  Sparkles,
  Sun,
  X,
  XCircle,
} from "lucide-react";
import {
  checkHealth,
  chatWithProject,
  createGeneration,
  type GenerationAttachment,
  getGeneration,
  listGenerations,
  listProjectChat,
  listProjectFiles,
  listProjects,
  projectDownloadUrl,
} from "./api";
import type {
  FileSummary,
  GenerationJob,
  JobStatus,
  ProjectChatMessage,
  ProjectSummary,
} from "./types";

const samplePrompts = [
  "Create a polished task management app with React, local storage, filters, and responsive design.",
  "Build a FastAPI blog API with SQLite, Pydantic schemas, CRUD endpoints, and README instructions.",
  "Create a weather dashboard using HTML, CSS, and JavaScript with search, cards, and graceful errors.",
];

function statusLabel(status: JobStatus) {
  if (status === "succeeded") return "Succeeded";
  if (status === "failed") return "Failed";
  if (status === "running") return "Running";
  return "Queued";
}

type Theme = "light" | "dark";

interface ComposerAttachment {
  id: string;
  file: File;
  name: string;
  type: string;
  size: number;
  contentText?: string;
  previewUrl?: string;
}

function getInitialTheme(): Theme {
  const storedTheme =
    window.localStorage.getItem("shastra-ai-theme") ??
    window.localStorage.getItem("coder-buddy-theme");
  if (storedTheme === "light" || storedTheme === "dark") {
    return storedTheme;
  }

  return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function App() {
  const [apiOnline, setApiOnline] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  const [prompt, setPrompt] = useState("");
  const [recursionLimit, setRecursionLimit] = useState(150);
  const [attachments, setAttachments] = useState<ComposerAttachment[]>([]);
  const [composerMenuOpen, setComposerMenuOpen] = useState(false);
  const [jobs, setJobs] = useState<GenerationJob[]>([]);
  const [activeJobId, setActiveJobId] = useState<string>();
  const [projects, setProjects] = useState<ProjectSummary[]>([]);
  const [selectedProject, setSelectedProject] = useState<string>();
  const [projectMessages, setProjectMessages] = useState<ProjectChatMessage[]>([]);
  const [files, setFiles] = useState<FileSummary[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string>();
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const activeJob = useMemo(
    () => jobs.find((job) => job.id === activeJobId),
    [activeJobId, jobs],
  );

  const runningJob = jobs.some((job) => job.status === "queued" || job.status === "running");

  useEffect(() => {
    void refreshDashboard();
  }, []);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    window.localStorage.setItem("shastra-ai-theme", theme);
  }, [theme]);

  useEffect(() => {
    if (!runningJob) return;
    const timer = window.setInterval(() => {
      void refreshJobs();
      void refreshProjects();
    }, 2000);
    return () => window.clearInterval(timer);
  }, [runningJob]);

  async function refreshDashboard() {
    const healthy = await checkHealth();
    setApiOnline(healthy);
    if (!healthy) {
      setError("Start the Shastra AI API on http://localhost:8000.");
      return;
    }

    setError(undefined);
    await Promise.all([refreshJobs(), refreshProjects()]);
  }

  async function refreshJobs() {
    const nextJobs = await listGenerations();
    setJobs(nextJobs);
    if (!activeJobId && nextJobs.length) {
      setActiveJobId(nextJobs[0].id);
    }
  }

  async function refreshProjects() {
    const nextProjects = await listProjects();
    setProjects(nextProjects);
  }

  async function selectProject(projectName: string) {
    setSelectedProject(projectName);
    const [response, chatHistory] = await Promise.all([
      listProjectFiles(projectName),
      listProjectChat(projectName),
    ]);
    setFiles(response.files);
    setProjectMessages(chatHistory);
    setPrompt("");
    setSidebarOpen(false);
  }

  function clearAttachments() {
    setAttachments((current) => {
      current.forEach((attachment) => {
        if (attachment.previewUrl) {
          URL.revokeObjectURL(attachment.previewUrl);
        }
      });
      return [];
    });
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const trimmedPrompt = prompt.trim();
    if (!trimmedPrompt && attachments.length === 0) return;
    const promptToSend =
      trimmedPrompt || "Use the attached files as context and generate the project from them.";

    setIsSubmitting(true);
    setError(undefined);

    try {
      const generationAttachments: GenerationAttachment[] = attachments.map((attachment) => ({
        name: attachment.name,
        mime_type: attachment.type,
        size: attachment.size,
        content_text: attachment.contentText,
      }));

      if (selectedProject) {
        const optimisticMessage: ProjectChatMessage = {
          id: crypto.randomUUID(),
          role: "user",
          content: promptToSend,
          created_at: new Date().toISOString(),
        };
        setProjectMessages((current) => [...current, optimisticMessage]);
        setPrompt("");
        clearAttachments();

        const response = await chatWithProject(selectedProject, promptToSend, generationAttachments);
        setProjectMessages(response.history);
        const projectFiles = await listProjectFiles(selectedProject);
        setFiles(projectFiles.files);
        await refreshProjects();
      } else {
        const response = await createGeneration(promptToSend, recursionLimit, generationAttachments);
        setActiveJobId(response.job_id);
        const job = await getGeneration(response.job_id);
        setJobs((current) => [job, ...current.filter((item) => item.id !== job.id)]);
        setPrompt("");
        clearAttachments();
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to start generation.");
    } finally {
      setIsSubmitting(false);
    }
  }

  async function createAttachmentsFromFiles(files: File[]): Promise<ComposerAttachment[]> {
    const nextAttachments = await Promise.all(
      files.slice(0, 8).map(async (file) => {
        const isTextLike =
          file.type.startsWith("text/") ||
          /\.(md|txt|csv|json|yaml|yml|xml|html|css|js|jsx|ts|tsx|py|java|go|rs|sql)$/i.test(
            file.name,
          );
        const isImage = file.type.startsWith("image/");
        const contentText = isTextLike ? (await file.text()).slice(0, 30000) : undefined;

        return {
          id: crypto.randomUUID(),
          file,
          name: file.name,
          type: file.type || "application/octet-stream",
          size: file.size,
          contentText,
          previewUrl: isImage ? URL.createObjectURL(file) : undefined,
        };
      }),
    );

    return nextAttachments;
  }

  async function addFiles(files: File[]) {
    if (!files.length) return;
    const nextAttachments = await createAttachmentsFromFiles(files);
    setAttachments((current) => [...current, ...nextAttachments].slice(0, 8));
  }

  async function handleFilesSelected(event: ChangeEvent<HTMLInputElement>) {
    const selectedFiles = Array.from(event.target.files ?? []);
    event.target.value = "";
    setComposerMenuOpen(false);
    await addFiles(selectedFiles);
  }

  async function handleComposerPaste(event: ClipboardEvent<HTMLTextAreaElement>) {
    const pastedFiles = Array.from(event.clipboardData.items)
      .filter((item) => item.kind === "file")
      .map((item) => item.getAsFile())
      .filter((file): file is File => file !== null);

    if (!pastedFiles.length) return;

    event.preventDefault();
    await addFiles(pastedFiles);
  }

  function removeAttachment(attachmentId: string) {
    setAttachments((current) => {
      const attachment = current.find((item) => item.id === attachmentId);
      if (attachment?.previewUrl) {
        URL.revokeObjectURL(attachment.previewUrl);
      }
      return current.filter((item) => item.id !== attachmentId);
    });
  }

  function startNewBuild() {
    setSelectedProject(undefined);
    setProjectMessages([]);
    setFiles([]);
    setPrompt("");
    clearAttachments();
  }

  return (
    <div className={`app-shell ${sidebarOpen ? "sidebar-expanded" : "sidebar-collapsed"}`}>
      {sidebarOpen && (
        <aside className="sidebar">
          <div className="sidebar-header">
            <div className="brand">
              <Code2 size={28} />
              <div>
                <span>AI engineering workspace</span>
                <h1>Shastra AI</h1>
              </div>
            </div>
            <button
              className="icon-button"
              type="button"
              aria-label="Close history sidebar"
              title="Close history sidebar"
              onClick={() => setSidebarOpen(false)}
            >
              <PanelLeftClose size={18} />
            </button>
          </div>

          <div className={`health ${apiOnline ? "online" : "offline"}`}>
            {apiOnline ? <CheckCircle2 size={17} /> : <XCircle size={17} />}
            API {apiOnline ? "online" : "offline"}
          </div>

          <button
            className="sidebar-refresh-button"
            type="button"
            aria-label="Refresh dashboard"
            title="Refresh dashboard"
            onClick={() => void refreshDashboard()}
          >
            <RefreshCcw size={17} />
            Refresh dashboard
          </button>

          <section className="panel compact">
            <div className="panel-title">
              <Activity size={18} />
              Recent runs
            </div>
            <div className="job-list">
              {jobs.map((job) => (
                <button
                  key={job.id}
                  className={`job-item ${job.id === activeJobId ? "active" : ""}`}
                  type="button"
                  onClick={() => setActiveJobId(job.id)}
                >
                  <span>{job.prompt}</span>
                  <small className={`status ${job.status}`}>{statusLabel(job.status)}</small>
                </button>
              ))}
              {!jobs.length && <p className="muted">Runs will appear here.</p>}
            </div>
          </section>

          <section className="panel compact">
            <div className="panel-title">
              <FolderGit2 size={18} />
              Generated projects
            </div>
            <div className="project-list">
              {projects.map((project) => (
                <button
                  key={project.name}
                  className={project.name === selectedProject ? "active" : ""}
                  type="button"
                  onClick={() => void selectProject(project.name)}
                >
                  <span>{project.name}</span>
                  <small>{project.file_count} files</small>
                </button>
              ))}
              {!projects.length && <p className="muted">No projects generated yet.</p>}
            </div>
          </section>

          {selectedProject && (
            <section className="panel project-inspector">
              <div className="section-heading split-heading">
                <div>
                  <Files size={18} />
                  <h3>{selectedProject}</h3>
                </div>
                <a
                  className="download-button icon-only-mobile"
                  href={projectDownloadUrl(selectedProject)}
                  download={`${selectedProject}.zip`}
                  title="Download project ZIP"
                >
                  <Download size={17} />
                  Download
                </a>
              </div>
              <div className="file-list compact-files">
                {files.slice(0, 8).map((file) => (
                  <div className="file-row" key={file.path}>
                    <span>{file.path}</span>
                    <small>{file.size.toLocaleString()} bytes</small>
                  </div>
                ))}
              </div>
            </section>
          )}
        </aside>
      )}

      <main className="workspace">
        <header className="topbar">
          <div className="topbar-title">
            {!sidebarOpen && (
              <button
                className="icon-button"
                type="button"
                aria-label="Open history sidebar"
                title="Open history sidebar"
                onClick={() => setSidebarOpen(true)}
              >
                <PanelLeftOpen size={18} />
              </button>
            )}
            <div>
              <span className="eyebrow">Shastra AI</span>
              <h2>AI engineering workspace</h2>
            </div>
          </div>
          <div className="topbar-actions">
            {selectedProject && (
              <button className="new-build-button" type="button" onClick={startNewBuild}>
                New build
              </button>
            )}
            <button
              className="icon-button"
              type="button"
              aria-label={theme === "dark" ? "Switch to light theme" : "Switch to dark theme"}
              title={theme === "dark" ? "Light theme" : "Dark theme"}
              onClick={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
            >
              {theme === "dark" ? <Sun size={18} /> : <Moon size={18} />}
            </button>
          </div>
        </header>

        {error && (
          <div className="error-banner">
            <AlertCircle size={18} />
            {error}
          </div>
        )}

        {selectedProject ? (
          <section className="project-chat-stage">
            <div className="project-chat-header">
              <div>
                <span className="eyebrow">Editing project</span>
                <h1>{selectedProject}</h1>
                <p>Ask for changes, fixes, polish, new features, or debugging. Shastra AI will edit the generated files directly.</p>
              </div>
              <a
                className="download-button"
                href={projectDownloadUrl(selectedProject)}
                download={`${selectedProject}.zip`}
                title="Download latest project ZIP"
              >
                <Download size={17} />
                Download latest
              </a>
            </div>

            <div className="project-chat-body">
              {projectMessages.length === 0 ? (
                <div className="empty-chat-state">
                  <Sparkles size={22} />
                  <h2>What should change?</h2>
                  <p>Try: "make the UI premium", "fix this error", "add login", or paste a screenshot and ask Shastra AI to match it.</p>
                </div>
              ) : (
                <>
                  {projectMessages.map((message) => (
                    <article className={`chat-message ${message.role}`} key={message.id}>
                      <span>{message.role === "assistant" ? "Shastra AI" : "You"}</span>
                      <p>{message.content}</p>
                    </article>
                  ))}
                  {isSubmitting && selectedProject && (
                    <article className="chat-message assistant pending">
                      <span>Shastra AI</span>
                      <p>
                        <Loader2 className="spin inline-spinner" size={16} />
                        Updating the project files...
                      </p>
                    </article>
                  )}
                </>
              )}
            </div>
          </section>
        ) : (
          <section className="welcome-stage">
            <div className="mascot-wrap" aria-hidden="true">
              <div className="mascot-glow" />
              <div className="buddy-mascot">
                <span className="antenna left" />
                <span className="antenna right" />
                <span className="ear left" />
                <span className="ear right" />
                <span className="eye left" />
                <span className="eye right" />
                <span className="foot left" />
                <span className="foot right" />
              </div>
            </div>

            <h1 className="hero-title">Shastra AI</h1>
            <p className="hero-kicker">THE AI THAT TURNS PROMPTS INTO PROJECTS.</p>
            <p className="hero-copy">
              Share an idea, spec, screenshot, or file. I will plan it, architect it, write it, validate it, and package it for download.
            </p>

            {activeJob && (
              <div className="live-status-card">
                <div>
                  <span className={`status ${activeJob.status}`}>{statusLabel(activeJob.status)}</span>
                  <strong>{activeJob.test_summary || activeJob.final_status || "Generation in progress"}</strong>
                </div>
                {activeJob.project_dir && <small>{activeJob.project_dir}</small>}
                {activeJob.error && <p>{activeJob.error}</p>}
              </div>
            )}

            <div className="quick-prompts">
              {samplePrompts.map((item) => (
                <button key={item} type="button" onClick={() => setPrompt(item)}>
                  <Sparkles size={16} />
                  {item.split(" ").slice(0, 5).join(" ")}
                </button>
              ))}
            </div>
          </section>
        )}

        <form className="prompt-dock" onSubmit={(event) => void handleSubmit(event)}>
          <input
            ref={fileInputRef}
            className="hidden-file-input"
            type="file"
            multiple
            accept="image/*,.txt,.md,.csv,.json,.yaml,.yml,.xml,.html,.css,.js,.jsx,.ts,.tsx,.py,.java,.go,.rs,.sql,.pdf,.doc,.docx"
            onChange={(event) => void handleFilesSelected(event)}
          />
          {attachments.length > 0 && (
            <div className="attachment-tray">
              {attachments.map((attachment) => (
                <div className="attachment-chip" key={attachment.id}>
                  {attachment.previewUrl ? (
                    <img src={attachment.previewUrl} alt="" />
                  ) : attachment.type.startsWith("image/") ? (
                    <Image size={16} />
                  ) : (
                    <Paperclip size={16} />
                  )}
                  <span>{attachment.name}</span>
                  <button
                    type="button"
                    aria-label={`Remove ${attachment.name}`}
                    title={`Remove ${attachment.name}`}
                    onClick={() => removeAttachment(attachment.id)}
                  >
                    <X size={14} />
                  </button>
                </div>
              ))}
            </div>
          )}
          <div className="composer-footer">
            <div className="composer-left">
              <button
                className="composer-plus"
                type="button"
                aria-label="Open composer options"
                title="Attach files and settings"
                onClick={() => setComposerMenuOpen((current) => !current)}
              >
                <Plus size={20} />
              </button>
              {attachments.length > 0 && (
                <span className="attachment-count">{attachments.length} attached</span>
              )}
              {composerMenuOpen && (
                <div className="composer-menu">
                  <button type="button" onClick={() => fileInputRef.current?.click()}>
                    <Paperclip size={16} />
                    Attach files, images, screenshots
                  </button>
                  <label>
                    <SlidersHorizontal size={16} />
                    Recursion limit
                    <input
                      type="number"
                      min={20}
                      max={500}
                      value={recursionLimit}
                      onChange={(event) => setRecursionLimit(Number(event.target.value))}
                    />
                  </label>
                </div>
              )}
            </div>
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              onPaste={(event) => void handleComposerPaste(event)}
              placeholder={
                selectedProject
                  ? `Ask Shastra AI to update ${selectedProject}`
                  : "Ask Shastra AI to build an app"
              }
              rows={1}
            />
            <button className="run-button" type="submit" disabled={!apiOnline || isSubmitting}>
              {isSubmitting ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
              {selectedProject ? "Update" : "Build"}
            </button>
          </div>
        </form>

        <p className="site-credit">
          Built with <span className="credit-heart">{"\u2665"}</span> by{" "}
          <span className="credit-name">Shashi</span>.
        </p>
      </main>
    </div>
  );
}

export default App;
