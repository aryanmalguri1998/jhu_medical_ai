import {
  ChangeEvent,
  DragEvent,
  KeyboardEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  AgentPatientResult,
  GroundTruthRow,
  OutcomeCounters,
  OutcomeEnvironment,
  OUTCOME_ENVIRONMENTS,
  PatientOutcomeEntry,
  derivePatientOutcomes,
  scenarioLabel,
} from "./lib/outcomePrompts";

type PatientRecord = Record<string, string | number | boolean | undefined>;

const TABLE_COLUMNS = [
  "Patient#",
  "Age",
  "Race",
  "Sex",
  "Insurance",
  "Suden Onset Vertigo",
  "Positional Vertigo",
  "Dizziness that is reproducible with standing",
  "BMI",
  "Years of Diabetes",
  "Atrial Fibrillation?",
  "Smoker?",
  "Prior Stroke?",
  "Ataxia on finger-nose-finger?",
  "Direction-changing nystagmus?",
  "Skew Devaition?",
  "Head Impulse Test?",
];

const DEFAULT_INSTRUCTIONS = `You are an AI-powered clinical decision support system who will be assisting in diagnosing patients in the emergency room presenting with dizziness who may have a stroke or may not have a stroke.

Background:
Patients presenting to the emergency department with dizziness are a challenging clinical problem. They may have strokes causing their dizziness (posterior circulation ischemic strokes) or non-stroke causes (mostly benign inner ear diseases). Based on the past medical history, presentation, and exam you should determine the most likely category for each patient and highlight what data points were most influential.

Output:
- Return JSON with structured fields for each patient mirroring the payload schema.
- Provide a short narrative summary for each case and list risk drivers.
- Assign a probability for stroke (0-100) for each patient.
- Compare your assessment against the provided ground truth and explain disagreements.`;

const MAX_AGENT_PATIENTS = 5;

type Stage = "landing" | "upload" | "configure";

type AgentProgressStatus = "pending" | "success" | "error";

type AgentProgressMap = Record<
  string,
  { status: AgentProgressStatus; message?: string }
>;

const envApiBase = (() => {
  const raw = import.meta.env.VITE_API_URL;
  if (typeof raw !== "string") {
    return "";
  }
  const trimmed = raw.trim();
  return trimmed ? trimmed.replace(/\/$/, "") : "";
})();

const buildApiUrl = (path: string) => {
  const normalized = path.startsWith("/") ? path : `/${path}`;
  return envApiBase ? `${envApiBase}${normalized}` : normalized;
};

const normalizePatientIdentifier = (
  value: string | number | boolean | undefined | null
): string => {
  if (value === undefined || value === null) {
    return "";
  }
  return String(value).trim();
};

const getPatientIdFromRecord = (record: PatientRecord): string => {
  return (
    normalizePatientIdentifier(record["Patient#"]) ||
    normalizePatientIdentifier(record["patient_id"]) ||
    normalizePatientIdentifier(record["PatientID"])
  );
};

const mergeAgentPatientResults = (
  existing: AgentPatientResult[],
  additions: AgentPatientResult[]
): AgentPatientResult[] => {
  if (!additions.length) {
    return existing;
  }
  const map = new Map<string, AgentPatientResult>();
  existing.forEach((entry, index) => {
    const key =
      normalizePatientIdentifier(entry.patient_id) || `existing-${index}`;
    map.set(key, entry);
  });
  additions.forEach((entry, index) => {
    const key =
      normalizePatientIdentifier(entry.patient_id) || `incoming-${index}`;
    map.set(key, entry);
  });
  return Array.from(map.values());
};

export default function App() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [stage, setStage] = useState<Stage>("landing");
  const [activeIngestion, setActiveIngestion] = useState<
    "none" | "excel" | "text"
  >("none");
  const [uploading, setUploading] = useState(false);
  const [patients, setPatients] = useState<PatientRecord[]>([]);
  const [groundTruth, setGroundTruth] = useState<GroundTruthRow[]>([]);
  const [rangeStart, setRangeStart] = useState(1);
  const [rangeEnd, setRangeEnd] = useState(1);
  const [instructions, setInstructions] = useState(DEFAULT_INSTRUCTIONS);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [agentResponses, setAgentResponses] = useState<unknown[]>([]);
  const [agentPatients, setAgentPatients] = useState<AgentPatientResult[]>([]);
  const [patientOutcomes, setPatientOutcomes] = useState<PatientOutcomeEntry[]>(
    []
  );
  const [outcomeSummary, setOutcomeSummary] = useState<OutcomeCounters | null>(
    null
  );
  const [outcomeEnvironment, setOutcomeEnvironment] =
    useState<OutcomeEnvironment>(OUTCOME_ENVIRONMENTS.CLINICAL_TRIAL);
  const [outcomeStatuses, setOutcomeStatuses] = useState<
    Record<string, string | null>
  >({});
  const [outcomeSending, setOutcomeSending] = useState<Record<string, boolean>>(
    {}
  );
  const [agentError, setAgentError] = useState<string | null>(null);
  const [agentProgress, setAgentProgress] = useState<AgentProgressMap>({});
  const [runningAgent, setRunningAgent] = useState(false);

  const hasPatients = patients.length > 0;
  const agentProgressEntries = Object.entries(agentProgress);
  const completedRuns = agentProgressEntries.filter(
    ([, entry]) => entry.status === "success"
  ).length;

  const stageIndex = useMemo(() => {
    if (stage === "configure") return 3;
    if (stage === "upload") return 2;
    return 1;
  }, [stage]);

  useEffect(() => {
    if (!agentPatients.length) {
      setPatientOutcomes([]);
      setOutcomeSummary(null);
      return;
    }
    const derived = derivePatientOutcomes(
      agentPatients,
      groundTruth,
      outcomeEnvironment
    );
    setPatientOutcomes(derived.entries);
    setOutcomeSummary(derived.summary);
  }, [agentPatients, groundTruth, outcomeEnvironment]);

  const selectedRange = useMemo(() => {
    if (!patients.length) return [];
    const from = Math.max(0, Math.min(rangeStart - 1, patients.length - 1));
    const to = Math.max(from + 1, Math.min(rangeEnd, patients.length));
    return patients.slice(from, to);
  }, [patients, rangeStart, rangeEnd]);

  const payloadPreview = useMemo(() => {
    if (!selectedRange.length) return "[]";
    const trimmed = selectedRange.slice(0, MAX_AGENT_PATIENTS);
    return JSON.stringify(trimmed, null, 2);
  }, [selectedRange]);

  const dynamicColumns = useMemo(() => {
    const visible = TABLE_COLUMNS.filter((column) =>
      patients.some((entry) => entry[column] !== undefined)
    );
    const extras = new Set<string>();
    patients.forEach((entry) => {
      Object.keys(entry).forEach((key) => {
        if (
          !visible.includes(key) &&
          ![
            "patient_id",
            "PatientID",
            "originalRowIndex",
            "Note",
            "Symptoms",
          ].includes(key)
        ) {
          extras.add(key);
        }
      });
    });
    return [...visible, ...Array.from(extras)];
  }, [patients]);

  const resetToLanding = () => {
    setStage("landing");
    setActiveIngestion("none");
    setStatusMessage(null);
  };

  const handleFiles = async (files: FileList | null) => {
    if (!files?.length) return;
    setUploading(true);
    setStatusMessage(null);
    setAgentResponses([]);
    setAgentError(null);
    setAgentProgress({});
    try {
      const file = files[0];
      const body = new FormData();
      body.append("file", file);
      const response = await fetch(buildApiUrl("/api/patient-data"), {
        method: "POST",
        body,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(
          payload.detail ?? payload.error ?? "Unable to parse workbook"
        );
      }
      const parsedPatients: PatientRecord[] = payload.patients ?? [];
      setPatients(parsedPatients);
      setGroundTruth(payload.groundTruth ?? []);
      setRangeStart(1);
      setRangeEnd(Math.max(parsedPatients.length, 1));
      setAgentPatients([]);
      setStage("configure");
      setStatusMessage(
        `Parsed ${parsedPatients.length} patient rows from ${file.name}`
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : "Upload failed";
      setStatusMessage(message);
      setStage("upload");
    } finally {
      setUploading(false);
    }
  };

  const handleFileInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    handleFiles(event.target.files);
    event.target.value = "";
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    handleFiles(event.dataTransfer.files);
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const handleDropzoneKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInputRef.current?.click();
    }
  };

  const clampRangeStart = (value: number) => {
    const maxPatientsCount = Math.max(patients.length, 1);
    const normalized = Math.max(1, Math.min(value, maxPatientsCount));
    setRangeStart(normalized);
    setRangeEnd((prev) => {
      const minEnd = normalized;
      const maxEnd = Math.min(
        normalized + MAX_AGENT_PATIENTS - 1,
        maxPatientsCount
      );
      const ensuredMin = Math.max(prev, minEnd);
      return Math.min(ensuredMin, maxEnd);
    });
  };

  const clampRangeEnd = (value: number) => {
    const maxPatientsCount = Math.max(patients.length, 1);
    const maxEnd = Math.min(
      rangeStart + MAX_AGENT_PATIENTS - 1,
      maxPatientsCount
    );
    const normalized = Math.max(rangeStart, Math.min(value, maxEnd));
    setRangeEnd(normalized);
  };

  const handleOutcomePromptChange = (patientId: string, value: string) => {
    setPatientOutcomes((prev) =>
      prev.map((entry) =>
        entry.patientId === patientId ? { ...entry, prompt: value } : entry
      )
    );
  };

  const sendOutcomePrompt = async (entry: PatientOutcomeEntry) => {
    setOutcomeSending((prev) => ({ ...prev, [entry.patientId]: true }));
    setOutcomeStatuses((prev) => ({ ...prev, [entry.patientId]: null }));
    try {
      const response = await fetch(buildApiUrl("/api/outcome-prompt"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          patientId: entry.patientId,
          prompt: entry.prompt,
          scenario: entry.scenario,
          environment: outcomeEnvironment,
          diagnosis: entry.diagnosis,
          predictedStroke: entry.predictedStroke,
          truthStroke: entry.truthStroke,
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(
          payload.detail ?? payload.error ?? "Outcome prompt failed"
        );
      }
      setOutcomeStatuses((prev) => ({
        ...prev,
        [entry.patientId]: "Sent to agent",
      }));
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "Outcome prompt failed";
      setOutcomeStatuses((prev) => ({ ...prev, [entry.patientId]: message }));
    } finally {
      setOutcomeSending((prev) => ({ ...prev, [entry.patientId]: false }));
    }
  };

  const runAgent = async () => {
    if (!selectedRange.length) return;
    const limitedPatients = selectedRange.slice(0, MAX_AGENT_PATIENTS);
    if (!limitedPatients.length) return;
    const patientIdSet = new Set(
      limitedPatients
        .map((entry) => getPatientIdFromRecord(entry))
        .filter((value) => value.length > 0)
    );
    const filteredGroundTruth = patientIdSet.size
      ? groundTruth.filter((row) =>
          patientIdSet.has(normalizePatientIdentifier(row["Patient#"]))
        )
      : groundTruth;
    const descriptors = limitedPatients.map((patient, index) => {
      const fallbackId = String(rangeStart + index);
      const patientId = getPatientIdFromRecord(patient) || fallbackId;
      return { patient, progressKey: patientId };
    });

    const initialProgress = descriptors.reduce<AgentProgressMap>(
      (acc, descriptor) => {
        acc[descriptor.progressKey] = { status: "pending" };
        return acc;
      },
      {}
    );

    setRunningAgent(true);
    setAgentError(null);
    setAgentResponses([]);
    setAgentPatients([]);
    setPatientOutcomes([]);
    setOutcomeSummary(null);
    setOutcomeStatuses({});
    setOutcomeSending({});
    setAgentProgress(initialProgress);

    try {
      const tasks = descriptors.map(({ patient, progressKey }) =>
        (async () => {
          try {
            const response = await fetch(buildApiUrl("/api/run-agent"), {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                patients: [patient],
                groundTruth: filteredGroundTruth,
                instructions,
                limit: 1,
              }),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
              throw new Error(
                payload.detail ?? payload.error ?? "Agent run failed"
              );
            }
            const responsePatients: AgentPatientResult[] =
              payload.output?.patients ?? [];
            if (responsePatients.length) {
              setAgentPatients((prev) =>
                mergeAgentPatientResults(prev, responsePatients)
              );
            }
            setAgentResponses((prev) => [...prev, payload]);
            setAgentProgress((prev) => ({
              ...prev,
              [progressKey]: { status: "success" },
            }));
          } catch (error) {
            const message =
              error instanceof Error ? error.message : "Agent run failed";
            setAgentProgress((prev) => ({
              ...prev,
              [progressKey]: { status: "error", message },
            }));
            setAgentError(message);
          }
        })()
      );

      await Promise.all(tasks);
    } finally {
      setRunningAgent(false);
    }
  };

  return (
    <div className="app-shell">
      <header className="jhu-header">
        <div className="brand-mark">
          <div className="brand-icon" aria-hidden="true">
            <svg viewBox="0 0 64 64" role="img" focusable="false">
              <path
                d="M20 8v18a12 12 0 0 0 24 0V8"
                fill="none"
                stroke="currentColor"
                strokeWidth="4"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M32 40v8a10 10 0 0 0 20 0v-6"
                fill="none"
                stroke="currentColor"
                strokeWidth="4"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <circle
                cx="52"
                cy="34"
                r="6"
                fill="none"
                stroke="currentColor"
                strokeWidth="4"
              />
              <circle
                cx="32"
                cy="56"
                r="5"
                fill="none"
                stroke="currentColor"
                strokeWidth="4"
              />
            </svg>
          </div>
          <div>
            <p className="brand-title">Johns Hopkins Medicine</p>
            <p className="brand-subtitle">Clinical Research Platform</p>
          </div>
        </div>
      </header>
      <main>
        <section className="hero-card">
          <div>
            <p className="badge">Vertigo Agent Workbench</p>
            <h1>Johns Hopkins vertigo triage assistant</h1>
            <p className="subtitle">
              Parse clinician spreadsheets locally, tune prompts, call the
              backend agent, and reconcile against real ground truth without
              Base44 dependencies.
            </p>
          </div>
          <ol className="stepper">
            {["Prepare Data", "Configure Prompt", "Run Agent", "Reconcile"].map(
              (label, index) => {
                const stepNumber = index + 1;
                const stateClass =
                  stepNumber < stageIndex
                    ? "complete"
                    : stepNumber === stageIndex
                    ? "active"
                    : "upcoming";
                const descriptions = [
                  "Upload Excel/CSV or paste patient rows",
                  "Inspect parsed data & tune the agent",
                  "Review structured results",
                  "Compare with ground truth & insights",
                ];
                return (
                  <li key={label} className={stateClass}>
                    <span className="step-number">{stepNumber}</span>
                    <div>
                      <p className="step-label">{label}</p>
                      <p className="step-description">{descriptions[index]}</p>
                    </div>
                  </li>
                );
              }
            )}
          </ol>
        </section>

        <section className="ingestion-card">
          <div className="ingestion-header">
            <div>
              <h2>Choose how to ingest data</h2>
              <p>
                Upload Excel/CSV straight from the browser or paste rows
                manually.
              </p>
            </div>
          </div>
          <div className="ingestion-options">
            <button
              type="button"
              className={`ingestion-tile ${
                activeIngestion === "excel" ? "selected" : ""
              }`}
              onClick={() => {
                setActiveIngestion("excel");
                setStage("upload");
              }}
            >
              <span className="tile-icon" aria-hidden />
              <div>
                <p className="tile-title">Excel/CSV File</p>
                <p className="tile-description">
                  Upload .xlsx, .xls, or .csv file
                </p>
              </div>
            </button>
          </div>
        </section>

        {activeIngestion === "excel" && stage === "upload" && (
          <section className="dropzone-card">
            <button
              type="button"
              className="back-link"
              onClick={resetToLanding}
            >
              &larr; Choose another ingestion option
            </button>
            <div
              className={`dropzone ${uploading ? "loading" : ""}`}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onKeyDown={handleDropzoneKeyDown}
              role="button"
              tabIndex={0}
            >
              <div className="dropzone-inner">
                <span className="upload-icon" aria-hidden />
                <p className="tile-title">Drop your Excel file here</p>
                <p className="tile-description">
                  or{" "}
                  <button
                    type="button"
                    className="link"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    browse
                  </button>{" "}
                  to upload
                </p>
                <p className="helper-text">Supports .xls, .xlsx, .csv files</p>
              </div>
            </div>
            {statusMessage && <p className="status-message">{statusMessage}</p>}
          </section>
        )}

        {stage === "configure" && (
          <section className="workspace-grid">
            <div className="data-panel">
              <header>
                <div>
                  <p className="badge subtle">Parsed Patient Rows</p>
                  <h3>{patients.length} records</h3>
                </div>
                <div className="range-inputs">
                  <label>
                    Range Start
                    <input
                      type="number"
                      min={1}
                      max={Math.max(patients.length, 1)}
                      value={rangeStart}
                      onChange={(event) => {
                        const parsed = Number(event.target.value);
                        clampRangeStart(Number.isNaN(parsed) ? 1 : parsed);
                      }}
                    />
                  </label>
                  <label>
                    Range End
                    <input
                      type="number"
                      min={rangeStart}
                      max={Math.min(
                        rangeStart + MAX_AGENT_PATIENTS - 1,
                        Math.max(patients.length, 1)
                      )}
                      value={rangeEnd}
                      onChange={(event) => {
                        const parsed = Number(event.target.value);
                        clampRangeEnd(
                          Number.isNaN(parsed) ? rangeStart : parsed
                        );
                      }}
                    />
                  </label>
                </div>
              </header>
              {statusMessage && (
                <p className="status-message subtle">{statusMessage}</p>
              )}

              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      {dynamicColumns.map((column) => (
                        <th key={column}>{column}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {patients.map((row, index) => (
                      <tr key={`${row["Patient#"] ?? index}-${index}`}>
                        {dynamicColumns.map((column) => (
                          <td key={column}>
                            {String(row[column] ?? "").slice(0, 60)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="helper-text">Showing {patients.length} records</p>
            </div>

            <div className="prompt-panel">
              <header>
                <p className="badge subtle">Agent Prompt</p>
                <h3>Customize the clinical instructions</h3>
              </header>
              <textarea
                value={instructions}
                onChange={(event) => setInstructions(event.target.value)}
                rows={8}
              />
              <p className="helper-text">
                Include structured schema instructions so the backend agent can
                return tabular results.
              </p>

              <div className="range-send-controls">
                <p className="helper-text">
                  Send this patient range (max {MAX_AGENT_PATIENTS})
                </p>
                <div className="range-fields">
                  <label>
                    Start
                    <input
                      type="number"
                      min={1}
                      max={Math.max(patients.length, 1)}
                      value={rangeStart}
                      onChange={(event) => {
                        const parsed = Number(event.target.value);
                        clampRangeStart(Number.isNaN(parsed) ? 1 : parsed);
                      }}
                    />
                  </label>
                  <span>to</span>
                  <label>
                    End
                    <input
                      type="number"
                      min={rangeStart}
                      max={Math.min(
                        rangeStart + MAX_AGENT_PATIENTS - 1,
                        Math.max(patients.length, 1)
                      )}
                      value={rangeEnd}
                      onChange={(event) => {
                        const parsed = Number(event.target.value);
                        clampRangeEnd(
                          Number.isNaN(parsed) ? rangeStart : parsed
                        );
                      }}
                    />
                  </label>
                </div>
              </div>

              <div className="range-note">
                Sending patients{" "}
                {selectedRange.length ? `${rangeStart}-${rangeEnd}` : "-"} (up
                to {MAX_AGENT_PATIENTS} per run).
                <br />
                Range selection is capped at {MAX_AGENT_PATIENTS} patients while
                we address timeout limits.
              </div>

              <button
                type="button"
                className="primary"
                disabled={!hasPatients || runningAgent}
                onClick={runAgent}
              >
                {runningAgent
                  ? "Running agent..."
                  : "Run agent on parsed patients"}
              </button>
              {agentError && (
                <p className="status-message error">{agentError}</p>
              )}
              {agentProgressEntries.length > 0 && (
                <div className="agent-progress">
                  <div className="preview-header">
                    <p>Run progress</p>
                    <span className="badge subtle">
                      {completedRuns}/{agentProgressEntries.length} completed
                    </span>
                  </div>
                  <ul className="agent-progress-list">
                    {agentProgressEntries.map(([patientId, progress]) => {
                      const label = /^patient/i.test(patientId)
                        ? patientId
                        : `Patient ${patientId}`;
                      let statusLabel = "Running...";
                      if (progress.status === "success") {
                        statusLabel = "Completed";
                      } else if (progress.status === "error") {
                        statusLabel = progress.message ?? "Failed";
                      }
                      return (
                        <li
                          key={patientId}
                          className={`agent-progress-item ${progress.status}`}
                        >
                          <span>{label}</span>
                          <span>{statusLabel}</span>
                        </li>
                      );
                    })}
                  </ul>
                </div>
              )}

              <div className="payload-preview">
                <div className="preview-header">
                  <p>Payload Preview</p>
                  <span className="badge subtle">Preview only</span>
                </div>
                <pre>{payloadPreview}</pre>
              </div>

              <div className="ground-truth">
                <div className="preview-header">
                  <p>Ground Truth</p>
                  <span className="badge subtle">
                    {groundTruth.length} rows
                  </span>
                </div>
                <div className="truth-scroll">
                  <div className="truth-grid">
                    {groundTruth.map((row) => (
                      <div key={row["Patient#"]} className="truth-card">
                        <p className="truth-label">Patient {row["Patient#"]}</p>
                        <p className="truth-value">
                          True Stroke: {row["True Stroke?"] ?? "-"}
                        </p>
                        <p className="truth-metric">
                          Stroke Risk: {row["Stroke Risk"] ?? "-"}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="response-panel">
                <div className="preview-header">
                  <p>Agent Response</p>
                  {agentResponses.length > 0 && (
                    <span className="badge subtle">
                      {agentResponses.length} payload
                      {agentResponses.length > 1 ? "s" : ""}
                    </span>
                  )}
                </div>
                <pre>
                  {agentResponses.length
                    ? JSON.stringify(agentResponses, null, 2)
                    : "Results will appear here after the agent run completes."}
                </pre>
                <div className="outcome-panel">
                  <div className="outcome-controls">
                    <label>
                      Outcome environment
                      <select
                        value={outcomeEnvironment}
                        onChange={(event) =>
                          setOutcomeEnvironment(
                            event.target.value as OutcomeEnvironment
                          )
                        }
                      >
                        <option value={OUTCOME_ENVIRONMENTS.CLINICAL_TRIAL}>
                          Clinical trial
                        </option>
                        <option value={OUTCOME_ENVIRONMENTS.ROUTINE_CARE}>
                          Routine care
                        </option>
                      </select>
                    </label>
                    <div className="outcome-match-summary">
                      <div>
                        <p>Matches</p>
                        <strong>{outcomeSummary?.correct ?? 0}</strong>
                      </div>
                      <div>
                        <p>Mismatch</p>
                        <strong>{outcomeSummary?.incorrect ?? 0}</strong>
                      </div>
                      <div>
                        <p>Strokes detected</p>
                        <strong>{outcomeSummary?.detected ?? 0}</strong>
                      </div>
                      <div>
                        <p>Strokes missed</p>
                        <strong>{outcomeSummary?.missed ?? 0}</strong>
                      </div>
                    </div>
                  </div>
                  {patientOutcomes.length === 0 ? (
                    <p className="helper-text">
                      Run the agent to compare predictions against ground truth
                      and craft patient-level outcome prompts.
                    </p>
                  ) : (
                    <div className="outcome-cards">
                      {patientOutcomes.map((entry) => {
                        const status = outcomeStatuses[entry.patientId] ?? null;
                        const isSending =
                          outcomeSending[entry.patientId] ?? false;
                        const predictionLabel = entry.predictedStroke
                          ? "Stroke"
                          : "Non-stroke";
                        const truthLabel = entry.truthStroke
                          ? "Stroke"
                          : "Non-stroke";
                        return (
                          <div className="outcome-card" key={entry.patientId}>
                            <div className="outcome-card-header">
                              <div>
                                <p className="patient-label">
                                  Patient {entry.patientId}
                                </p>
                                <p className="scenario-label">
                                  {entry.narrative}
                                </p>
                                <p className="scenario-meta helper-text">
                                  Prediction: {predictionLabel} | Truth:{" "}
                                  {truthLabel}
                                </p>
                              </div>
                              <span
                                className={`scenario-pill ${entry.scenario}`}
                              >
                                {scenarioLabel(entry.scenario)}
                              </span>
                            </div>
                            <textarea
                              value={entry.prompt}
                              onChange={(event) =>
                                handleOutcomePromptChange(
                                  entry.patientId,
                                  event.target.value
                                )
                              }
                              rows={8}
                            />
                            <div className="outcome-actions">
                              <button
                                type="button"
                                className="secondary"
                                disabled={isSending}
                                onClick={() => sendOutcomePrompt(entry)}
                              >
                                {isSending
                                  ? "Sending..."
                                  : "Send outcome prompt"}
                              </button>
                              {status && (
                                <span className="outcome-status">{status}</span>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </section>
        )}
      </main>

      <input
        ref={fileInputRef}
        type="file"
        accept=".csv,.xlsx,.xls"
        hidden
        onChange={handleFileInputChange}
      />
    </div>
  );
}
