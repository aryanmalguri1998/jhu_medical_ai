"""FastAPI entrypoint for ingesting clinical workbooks and running agents."""

from __future__ import annotations

import io
import json
import os
import uuid
from typing import Annotated, Any, Mapping

import httpx
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from dotenv import load_dotenv

from .sample_data import SAMPLE_GROUND_TRUTH, SAMPLE_PATIENTS
from .workflow_runner import WorkflowAgentError, execute_workflow

DEFAULT_CHATKIT_BASE = "https://api.openai.com"
SESSION_COOKIE_NAME = "chatkit_session_id"
SESSION_COOKIE_MAX_AGE_SECONDS = 60 * 60 * 24 * 30  # 30 days

app = FastAPI(title="Managed ChatKit Session API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYMPTOM_FIELDS = [
    "Suden Onset Vertigo",
    "Positional Vertigo",
    "Dizziness that is reproducible with standing",
]
GROUND_TRUTH_FIELDS = ["True Stroke?", "Stroke Risk"]
MAX_AGENT_BATCH = 5

SINGLE_SEGMENT_FIELDS = {
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
}

load_dotenv()


class RunAgentRequest(BaseModel):
    """Payload schema for invoking the backend agent."""

    patients: list[dict[str, Any]]
    ground_truth: Annotated[
        list[dict[str, Any]] | None,
        Field(alias="groundTruth"),
    ] = None
    instructions: str | None = None
    workflow_id: Annotated[
        str | None,
        Field(alias="workflowId"),
    ] = None
    limit: int | None = None

    model_config = ConfigDict(populate_by_name=True)


class OutcomePromptRequest(BaseModel):
    patient_id: Annotated[str, Field(alias="patientId")]
    prompt: str
    environment: str | None = None
    scenario: str | None = None
    diagnosis: str | None = None
    predicted_stroke: Annotated[bool | None, Field(alias="predictedStroke")]=None
    truth_stroke: Annotated[bool | None, Field(alias="truthStroke")]=None

    model_config = ConfigDict(populate_by_name=True)


@app.get("/health")
async def health() -> Mapping[str, str]:
    return {"status": "ok"}


@app.post("/api/create-session")
async def create_session(request: Request) -> JSONResponse:
    """Exchange a workflow id for a ChatKit client secret."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return respond({"error": "Missing OPENAI_API_KEY environment variable"}, 500)

    body = await read_json_body(request)
    workflow_id = resolve_workflow_id(body)
    if not workflow_id:
        return respond({"error": "Missing workflow id"}, 400)

    user_id, cookie_value = resolve_user(request.cookies)
    api_base = chatkit_api_base()

    try:
        async with httpx.AsyncClient(base_url=api_base, timeout=10.0) as client:
            upstream = await client.post(
                "/v1/chatkit/sessions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "OpenAI-Beta": "chatkit_beta=v1",
                    "Content-Type": "application/json",
                },
                json={"workflow": {"id": workflow_id}, "user": user_id},
            )
    except httpx.RequestError as error:
        return respond(
            {"error": f"Failed to reach ChatKit API: {error}"},
            502,
            cookie_value,
        )

    payload = parse_json(upstream)
    if not upstream.is_success:
        message = None
        if isinstance(payload, Mapping):
            message = payload.get("error")
        message = message or upstream.reason_phrase or "Failed to create session"
        return respond({"error": message}, upstream.status_code, cookie_value)

    client_secret = None
    expires_after = None
    if isinstance(payload, Mapping):
        client_secret = payload.get("client_secret")
        expires_after = payload.get("expires_after")

    if not client_secret:
        return respond(
            {"error": "Missing client secret in response"},
            502,
            cookie_value,
        )

    return respond(
        {"client_secret": client_secret, "expires_after": expires_after},
        200,
        cookie_value,
    )


@app.post("/api/patient-data")
async def ingest_patient_data(file: UploadFile = File(...)) -> Mapping[str, Any]:
    """Parse uploaded spreadsheets into structured patient payloads."""

    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename on upload")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    dataframe = load_dataframe(file.filename, raw_bytes)
    patients, ground_truth = serialize_dataframe(dataframe)
    return {"patients": patients, "groundTruth": ground_truth}


@app.get("/api/sample-data")
async def sample_data() -> Mapping[str, Any]:
    """Return a curated workbook payload for demos or testing."""

    return {"patients": SAMPLE_PATIENTS, "groundTruth": SAMPLE_GROUND_TRUTH}


@app.post("/api/run-agent")
async def run_agent(
    payload: RunAgentRequest,
    request: Request,
    response: Response,
) -> Mapping[str, Any]:
    """Invoke the configured workflow with the prepared patient payload."""

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    patients = payload.patients or []
    if not patients:
        raise HTTPException(status_code=400, detail="Patient payload is required")

    limit = payload.limit or MAX_AGENT_BATCH
    limit = min(limit, MAX_AGENT_BATCH)
    trimmed_patients = patients[:limit]

    user_id, cookie_value = resolve_user(request.cookies)
    set_session_cookie(response, cookie_value)

    workflow_input = {
        "patients": trimmed_patients,
        "ground_truth": payload.ground_truth or [],
        "instructions": (payload.instructions or "").strip(),
    }
    try:
        agent_output = await execute_workflow(workflow_input, session_id=user_id)
    except WorkflowAgentError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except Exception as error:  # pragma: no cover - bubble up to FastAPI handler
        raise HTTPException(
            status_code=500,
            detail=f"Workflow execution failed: {error}",
        ) from error

    return {"output": agent_output}


@app.post("/api/outcome-prompt")
async def send_outcome_prompt(
    payload: OutcomePromptRequest,
    request: Request,
    response: Response,
) -> Mapping[str, Any]:
    prompt = (payload.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Outcome prompt must not be empty")

    user_id, cookie_value = resolve_user(request.cookies)
    set_session_cookie(response, cookie_value)

    body = {
        "patient_id": payload.patient_id,
        "outcome_prompt": prompt,
        "environment": payload.environment,
        "scenario": payload.scenario,
        "diagnosis": payload.diagnosis,
        "predicted_stroke": payload.predicted_stroke,
        "truth_stroke": payload.truth_stroke,
    }

    try:
        agent_output = await execute_workflow(body, session_id=user_id)
    except WorkflowAgentError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    except Exception as error:  # pragma: no cover - surface unexpected failures
        raise HTTPException(status_code=500, detail=f"Outcome prompt failed: {error}") from error

    return {"status": "sent", "agentResponse": agent_output}


def set_session_cookie(response: Response, cookie_value: str | None = None) -> None:
    if not cookie_value:
        return
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=cookie_value,
        max_age=SESSION_COOKIE_MAX_AGE_SECONDS,
        httponly=True,
        samesite="lax",
        secure=is_prod(),
        path="/",
    )


def respond(
    payload: Mapping[str, Any], status_code: int, cookie_value: str | None = None
) -> JSONResponse:
    response = JSONResponse(payload, status_code=status_code)
    set_session_cookie(response, cookie_value)
    return response


def is_prod() -> bool:
    env = (os.getenv("ENVIRONMENT") or os.getenv("NODE_ENV") or "").lower()
    return env == "production"


async def read_json_body(request: Request) -> Mapping[str, Any]:
    raw = await request.body()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, Mapping) else {}


def resolve_workflow_id(body: Mapping[str, Any] | None = None) -> str | None:
    payload = body or {}
    workflow = payload.get("workflow", {})
    workflow_id = None
    if isinstance(workflow, Mapping):
        workflow_id = workflow.get("id")
    workflow_id = workflow_id or payload.get("workflowId")
    env_workflow = (
        os.getenv("CHATKIT_WORKFLOW_ID")
        or os.getenv("VITE_CHATKIT_WORKFLOW_ID")
        or os.getenv("NEXT_PUBLIC_CHATKIT_WORKFLOW_ID")
    )
    if not workflow_id and env_workflow:
        workflow_id = env_workflow
    if workflow_id and isinstance(workflow_id, str) and workflow_id.strip():
        return workflow_id.strip()
    return None


def resolve_user(cookies: Mapping[str, str]) -> tuple[str, str | None]:
    existing = cookies.get(SESSION_COOKIE_NAME)
    if existing:
        return existing, None
    user_id = str(uuid.uuid4())
    return user_id, user_id


def load_dataframe(filename: str, payload: bytes) -> pd.DataFrame:
    """Create a pandas dataframe from CSV or Excel content."""

    destination = io.BytesIO(payload)
    normalized_name = filename.lower()
    try:
        if normalized_name.endswith(".csv"):
            frame = pd.read_csv(destination)
        elif normalized_name.endswith((".xlsx", ".xls")):
            frame = pd.read_excel(destination)
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file type. Upload .csv, .xls, or .xlsx files.",
            )
    except Exception as error:  # pragma: no cover - handled by FastAPI
        raise HTTPException(
            status_code=400,
            detail=f"Unable to parse workbook: {error}",
        ) from error

    if frame.empty:
        raise HTTPException(status_code=400, detail="Workbook does not contain any rows")

    return frame.fillna("")


def serialize_dataframe(
    dataframe: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Turn the user workbook into structured patient and ground truth payloads."""

    frame = dataframe.copy()
    frame = frame.dropna(how="all")
    frame = frame.dropna(axis=1, how="all")

    if looks_like_transposed_matrix(frame):
        return parse_transposed_matrix(frame)

    return parse_rowwise_matrix(frame)


SECTION_LABELS = {
    "history",
    "hpi",
    "medical history",
    "exam",
    "type",
    "percent chance",
    "true stroke? percent chance type",
    "rt risk%",
}

LABEL_ALIASES = {
    "sudden onset vertigo": "Suden Onset Vertigo",
    "skew deviation?": "Skew Devaition?",
    "skew deviation": "Skew Devaition?",
    "true stroke": "True Stroke?",
    "true stroke?": "True Stroke?",
    "stroke risk": "Stroke Risk",
}


def looks_like_transposed_matrix(dataframe: pd.DataFrame) -> bool:
    if dataframe.empty or dataframe.shape[1] < 2:
        return False
    first_column = dataframe.iloc[:, 0].astype(str).str.lower()
    return any("patient#" in value for value in first_column)


def parse_transposed_matrix(dataframe: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    label_column = dataframe.columns[0]
    labels_raw = [normalize_value(value) for value in dataframe[label_column]]
    normalized_labels = [normalize_label_key(label) for label in labels_raw]

    records: list[dict[str, Any]] = []
    for column_index, column in enumerate(dataframe.columns[1:]):
        values = dataframe[column].tolist()
        record: dict[str, Any] = {}
        for normalized_label, raw_value in zip(normalized_labels, values):
            if not normalized_label:
                continue
            normalized_value = normalize_value(raw_value)
            if not normalized_value:
                continue
            append_record_value(record, normalized_label, normalized_value)
        if record:
            record["_row_index"] = column_index
            records.append(record)

    return finalize_patient_records(records)


def parse_rowwise_matrix(dataframe: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for index, row in dataframe.iterrows():
        record: dict[str, Any] = {}
        for column, value in row.items():
            key = normalize_label_key(str(column))
            normalized_value = normalize_value(value)
            if not key or not normalized_value:
                continue
            append_record_value(record, key, normalized_value)
        if record:
            record["_row_index"] = int(index)
            records.append(record)

    return finalize_patient_records(records)


def finalize_patient_records(
    records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    patients: list[dict[str, Any]] = []
    ground_truth: list[dict[str, Any]] = []

    for index, source in enumerate(records):
        record = dict(source)
        row_index = int(record.pop("_row_index", index))
        patient_id = str(record.get("Patient#") or len(patients) + 1)
        record.setdefault("Patient#", patient_id)
        record.setdefault("PatientID", patient_id)
        record["patient_id"] = patient_id
        record["originalRowIndex"] = row_index
        record["Symptoms"] = build_symptom_summary(record)
        record["Note"] = build_note(record, patient_id)
        clean_single_segment_fields(record)

        true_stroke = record.pop("True Stroke?", None)
        stroke_risk = record.pop("Stroke Risk", None) or record.pop(
            "Percent Chance", None
        )
        if true_stroke or stroke_risk:
            ground_truth.append(
                {
                    "Patient#": patient_id,
                    "True Stroke?": true_stroke or "",
                    "Stroke Risk": stroke_risk or "",
                }
            )

        patients.append(record)

    return patients, ground_truth


def clean_single_segment_fields(record: dict[str, Any]) -> None:
    for field in SINGLE_SEGMENT_FIELDS:
        value = record.get(field)
        if not value:
            continue
        record[field] = extract_primary_segment(value)


def normalize_label_key(label: str) -> str:
    text = (label or "").strip()
    if not text:
        return ""
    lower = text.lower()
    if lower.startswith("unnamed:"):
        return ""
    if lower in SECTION_LABELS:
        return ""
    if text.isdigit():
        return ""
    stripped_numeric = text.replace(".", "", 1)
    if stripped_numeric.isdigit():
        return ""
    return LABEL_ALIASES.get(lower, text)


def normalize_value(value: Any) -> str:
    """Convert dataframe values into trimmed strings."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = str(value).strip()
    return text


def append_record_value(record: dict[str, Any], key: str, value: str) -> None:
    """Merge duplicate columns by concatenating their contents."""

    if not value:
        return
    existing = record.get(key)
    if existing:
        record[key] = f"{existing} | {value}"
    else:
        record[key] = value


def build_symptom_summary(record: Mapping[str, Any]) -> str:
    """Aggregate vertigo related columns into a single string."""

    parts: list[str] = []
    for field in SYMPTOM_FIELDS:
        value = record.get(field)
        if value:
            parts.append(str(value))
    return "; ".join(parts)


def build_note(record: Mapping[str, Any], patient_id: str) -> str:
    """Generate a narrative summary for the payload preview."""

    age = extract_primary_segment(record.get("Age")) or "unknown"
    race = extract_primary_segment(record.get("Race")) or "patient"
    sex = extract_primary_segment(record.get("Sex")) or ""
    dizziness = extract_narrative_segment(
        record.get("Dizziness that is reproducible with standing")
    )
    diabetes = extract_narrative_segment(record.get("Years of Diabetes")) or "no documented diabetes"
    smoker = extract_narrative_segment(record.get("Smoker?")) or "unknown smoking status"
    prior_stroke = extract_narrative_segment(record.get("Prior Stroke?")) or "no history of prior stroke"
    atrial_fib = extract_narrative_segment(record.get("Atrial Fibrillation?")) or "no known atrial fibrillation"
    bmi = extract_primary_segment(record.get("BMI")) or "unknown"
    ataxia = extract_narrative_segment(record.get("Ataxia on finger-nose-finger?")) or "no ataxia noted"
    nystagmus = extract_narrative_segment(record.get("Direction-changing nystagmus?")) or "no direction changing nystagmus noted"
    skew = extract_narrative_segment(record.get("Skew Devaition?")) or "no skew deviation"
    head_impulse = extract_narrative_segment(record.get("Head Impulse Test?")) or "no head impulse findings"

    history = (
        f"[Patient {patient_id}] History: A {age} year old {race} {sex}".strip()
        + f" presents with {dizziness or 'dizziness'}."
    )
    medical_history = (
        "They have a past medical history of: "
        f"{diabetes}, {smoker}, {prior_stroke}, {atrial_fib}, and a BMI of {bmi}."
    )
    exam = (
        "On bedside exam they have "
        f"{ataxia}, {nystagmus}, {skew}, and {head_impulse}."
    )
    return " ".join([history, medical_history, exam]).strip()


def extract_primary_segment(value: Any) -> str:
    text = normalize_value(value)
    if not text:
        return ""
    return text.split("|")[0].strip()


def extract_narrative_segment(value: Any) -> str:
    text = normalize_value(value)
    if not text:
        return ""
    segments = [segment.strip().strip(",") for segment in text.split("|") if segment.strip()]
    if not segments:
        return ""
    if len(segments) == 1:
        return segments[0]
    for candidate in segments[1:]:
        if any(character.isalpha() for character in candidate):
            return candidate
    return segments[0]


def chatkit_api_base() -> str:
    return (
        os.getenv("CHATKIT_API_BASE")
        or os.getenv("VITE_CHATKIT_API_BASE")
        or DEFAULT_CHATKIT_BASE
    )


def parse_json(response: httpx.Response) -> Mapping[str, Any]:
    try:
        parsed = response.json()
        return parsed if isinstance(parsed, Mapping) else {}
    except (json.JSONDecodeError, httpx.DecodingError):
        return {}
