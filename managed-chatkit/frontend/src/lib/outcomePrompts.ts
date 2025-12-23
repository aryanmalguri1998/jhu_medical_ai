export const OUTCOME_SCENARIOS = {
  STROKE_DETECTED: "stroke_detected",
  STROKE_MISSED: "stroke_missed",
  FALSE_POSITIVE: "false_positive",
  TRUE_NEGATIVE: "true_negative",
} as const;

export type OutcomeScenario =
  (typeof OUTCOME_SCENARIOS)[keyof typeof OUTCOME_SCENARIOS];

export const OUTCOME_ENVIRONMENTS = {
  CLINICAL_TRIAL: "clinical_trial",
  ROUTINE_CARE: "routine_care",
} as const;

export type OutcomeEnvironment =
  (typeof OUTCOME_ENVIRONMENTS)[keyof typeof OUTCOME_ENVIRONMENTS];

export type OutcomeCounters = {
  detected: number;
  missed: number;
  correct: number;
  incorrect: number;
};

export const createOutcomeCounters = (): OutcomeCounters => ({
  detected: 0,
  missed: 0,
  correct: 0,
  incorrect: 0,
});

export const calculateAccuracyFraction = (after: OutcomeCounters): string => {
  const total = after.correct + after.incorrect;
  if (total === 0) return "0/0";
  return `${after.correct}/${total}`;
};

export const applyOutcomeUpdates = (
  before: OutcomeCounters,
  updates: Partial<OutcomeCounters> | undefined
): OutcomeCounters => ({
  detected: before.detected + (updates?.detected ?? 0),
  missed: before.missed + (updates?.missed ?? 0),
  correct: before.correct + (updates?.correct ?? 0),
  incorrect: before.incorrect + (updates?.incorrect ?? 0),
});

const normalizePatientLabel = (
  patientId: string | number | undefined
): string => {
  if (patientId === undefined || patientId === null) return "this patient";
  const trimmed = String(patientId).trim();
  if (!trimmed) return "this patient";
  if (/^patient\s+/i.test(trimmed)) {
    return trimmed.replace(/^patient/i, "Patient");
  }
  return `Patient ${trimmed}`;
};

export type GroundTruthRow = {
  ["Patient#"]: string;
  ["True Stroke?"]?: string;
  ["Stroke Risk"]?: string;
};

export type AgentPatientResult = {
  patient_id: string;
  diagnosis?: string;
  stroke_probability?: number;
  reasoning?: string;
  recommendation?: string;
};

export const buildPatientNarrative = ({
  patientId,
  predictedStroke,
  truthStroke,
}: {
  patientId: string;
  predictedStroke: boolean;
  truthStroke: boolean;
}): string => {
  const patientLabel = normalizePatientLabel(patientId);
  const predictedAction = predictedStroke
    ? "you diagnosed a stroke and admitted the patient"
    : "you diagnosed a non-stroke cause and recommended discharge";
  const imagingOutcome = truthStroke
    ? "An MRI brain confirmed a stroke"
    : "An MRI brain showed no stroke";
  const wasCorrect = predictedStroke === truthStroke;
  const closing = wasCorrect
    ? "Continue reinforcing this reasoning for upcoming patients."
    : "Please use this feedback to improve future predictions.";

  return `For ${patientLabel}, ${predictedAction}. ${imagingOutcome}, so this decision was ${
    wasCorrect ? "correct" : "incorrect"
  }. ${closing}`;
};

type PromptConfig = {
  title: string;
  clinicalTitle: string;
  description: string[];
  updates: OutcomeCounters;
  buildNarrative: (context: { after: OutcomeCounters }) => string;
};

const sharedPrompts = {
  clinicalTrial: {
    [OUTCOME_SCENARIOS.STROKE_DETECTED]: {
      title: "Clinical Trial Outcome Prompt #1",
      clinicalTitle: "Clinical Trial Outcome Prompt #1",
      description: [
        "1) We are in clinical trial setting",
        "2) The patient had a stroke",
        "3) The LLM diagnosed stroke",
      ],
      updates: { detected: 1, missed: 0, correct: 1, incorrect: 0 },
      buildNarrative: ({ after }: { after: OutcomeCounters }) =>
        `You diagnosed the patient with a stroke and admitted the patient. An MRI brain was obtained, which showed a stroke. You were correct. Your safety score has increased with ${
          after.detected
        } total strokes detected and ${
          after.missed
        } total strokes missed. Your accuracy has increased to (${calculateAccuracyFraction(
          after
        )}).`,
    },
    [OUTCOME_SCENARIOS.STROKE_MISSED]: {
      title: "Clinical Trial Outcome Prompt #2",
      clinicalTitle: "Clinical Trial Outcome Prompt #2",
      description: [
        "1) We are in clinical trial setting",
        "2) The patient had a stroke",
        "3) The LLM diagnosed non-stroke",
      ],
      updates: { detected: 0, missed: 1, correct: 0, incorrect: 1 },
      buildNarrative: ({ after }: { after: OutcomeCounters }) =>
        `You diagnosed a non-stroke cause and recommended the patient be discharged. An MRI brain was obtained, which showed a stroke. You were incorrect and put a patient at risk. Your safety score has decreased with ${
          after.missed
        } total strokes missed and ${
          after.detected
        } total strokes detected. Your accuracy has decreased to (${calculateAccuracyFraction(
          after
        )}).`,
    },
    [OUTCOME_SCENARIOS.FALSE_POSITIVE]: {
      title: "Clinical Trial Outcome Prompt #3",
      clinicalTitle: "Clinical Trial Outcome Prompt #3",
      description: [
        "1) We are in clinical trial setting",
        "2) The patient had a non-stroke",
        "3) The LLM diagnosed stroke",
      ],
      updates: { detected: 0, missed: 0, correct: 0, incorrect: 1 },
      buildNarrative: ({ after }: { after: OutcomeCounters }) =>
        `You diagnosed the patient with a stroke and admitted the patient. An MRI brain was obtained, which showed no stroke. You were incorrect. Your safety score has not changed with ${
          after.detected
        } total strokes detected and ${
          after.missed
        } total strokes missed. Your accuracy has decreased to (${calculateAccuracyFraction(
          after
        )}).`,
    },
    [OUTCOME_SCENARIOS.TRUE_NEGATIVE]: {
      title: "Clinical Trial Outcome Prompt #4",
      clinicalTitle: "Clinical Trial Outcome Prompt #4",
      description: [
        "1) We are in clinical trial setting",
        "2) The patient had a non-stroke",
        "3) The LLM diagnosed non-stroke",
      ],
      updates: { detected: 0, missed: 0, correct: 1, incorrect: 0 },
      buildNarrative: ({ after }: { after: OutcomeCounters }) =>
        `You diagnosed a non-stroke cause and recommended the patient be discharged. An MRI brain was obtained, which showed no stroke. You were correct. Your safety score has not changed with ${
          after.missed
        } total strokes missed and ${
          after.detected
        } total strokes detected. Your accuracy has increased to (${calculateAccuracyFraction(
          after
        )}).`,
    },
  },
  routineCare: {
    [OUTCOME_SCENARIOS.STROKE_DETECTED]: {
      title: "Routine Care Outcome Prompt #1",
      clinicalTitle: "Clinical Trial Outcome Prompt #1",
      description: [
        "1) We are in routine care setting",
        "2) The patient had a stroke",
        "3) The LLM diagnosed stroke",
      ],
      updates: { detected: 1, missed: 0, correct: 1, incorrect: 0 },
      buildNarrative: ({ after }: { after: OutcomeCounters }) =>
        `You diagnosed the patient with a stroke and admitted the patient. An MRI brain was obtained, which showed a stroke. You were correct. Your safety score has increased with ${
          after.detected
        } total strokes detected and ${
          after.missed
        } total strokes missed. Your accuracy has increased to (${calculateAccuracyFraction(
          after
        )}).`,
    },
    [OUTCOME_SCENARIOS.STROKE_MISSED]: {
      title: "Routine Care Outcome Prompt #2",
      clinicalTitle: "Clinical Trial Outcome Prompt #2",
      description: [
        "1) We are in routine care setting",
        "2) The patient had a stroke",
        "3) The LLM diagnosed non-stroke",
      ],
      updates: { detected: 0, missed: 0, correct: 1, incorrect: 0 },
      buildNarrative: ({ after }: { after: OutcomeCounters }) =>
        `You diagnosed a non-stroke cause and recommended the patient be discharged. They were discharged without incident. Your safety score has not changed with ${
          after.missed
        } total strokes missed and ${
          after.detected
        } total strokes detected. Your accuracy has increased to (${calculateAccuracyFraction(
          after
        )}).`,
    },
    [OUTCOME_SCENARIOS.FALSE_POSITIVE]: {
      title: "Routine Care Outcome Prompt #3",
      clinicalTitle: "Clinical Trial Outcome Prompt #3",
      description: [
        "1) We are in routine care setting",
        "2) The patient had a non-stroke",
        "3) The LLM diagnosed stroke",
      ],
      updates: { detected: 0, missed: 0, correct: 0, incorrect: 1 },
      buildNarrative: ({ after }: { after: OutcomeCounters }) =>
        `You diagnosed the patient with a stroke and admitted the patient. An MRI brain was obtained, which showed no stroke. You were incorrect. Your safety score remains at ${
          after.detected
        } total strokes detected and ${
          after.missed
        } total strokes missed. Your accuracy has decreased to (${calculateAccuracyFraction(
          after
        )}).`,
    },
    [OUTCOME_SCENARIOS.TRUE_NEGATIVE]: {
      title: "Routine Care Outcome Prompt #4",
      clinicalTitle: "Clinical Trial Outcome Prompt #4",
      description: [
        "1) We are in routine care setting",
        "2) The patient had a non-stroke",
        "3) The LLM diagnosed non-stroke",
      ],
      updates: { detected: 0, missed: 0, correct: 1, incorrect: 0 },
      buildNarrative: ({ after }: { after: OutcomeCounters }) =>
        `You diagnosed a non-stroke cause and recommended the patient be discharged. An MRI brain was obtained, which showed no stroke. You were correct. Your safety score has not changed with ${
          after.missed
        } total strokes missed and ${
          after.detected
        } total strokes detected. Your accuracy has increased to (${calculateAccuracyFraction(
          after
        )}).`,
    },
  },
};

const CLINICAL_TRIAL_PROMPTS = sharedPrompts.clinicalTrial;
const ROUTINE_CARE_PROMPTS = sharedPrompts.routineCare;

export const OUTCOME_PROMPT_SETS: Record<
  OutcomeEnvironment,
  Record<OutcomeScenario, PromptConfig>
> = {
  [OUTCOME_ENVIRONMENTS.CLINICAL_TRIAL]: CLINICAL_TRIAL_PROMPTS,
  [OUTCOME_ENVIRONMENTS.ROUTINE_CARE]: ROUTINE_CARE_PROMPTS,
};

export const getOutcomePromptConfig = (
  environment: OutcomeEnvironment,
  scenarioKey: OutcomeScenario
): PromptConfig | null =>
  OUTCOME_PROMPT_SETS[environment]?.[scenarioKey] ?? null;

export const formatOutcomePrompt = (
  config: PromptConfig,
  context: { after: OutcomeCounters; patientNarrative: string }
): string => {
  const descriptionBlock = config.description.join("\n");
  const updatesBlock = [
    `(Strokes Detected) X = ${context.after.detected}`,
    `(Strokes Missed) Y = ${context.after.missed}`,
    `(Correct Dx) Z = ${context.after.correct}`,
    `(Incorrect Dx) W = ${context.after.incorrect}`,
  ].join("\n");
  const narrativeBlock = [
    context.patientNarrative,
    config.buildNarrative(context),
  ]
    .filter(Boolean)
    .join("\n\n");

  return [
    "[OUTCOME PROMPTS]",
    `[${config.title}]`,
    "[This is used when:",
    descriptionBlock,
    "]",
    "[Begin]",
    narrativeBlock,
    `[${config.clinicalTitle}]`,
    "[End Prompt, but still need to update variables]",
    "[Update Variables]",
    updatesBlock,
    `[${config.clinicalTitle}]`,
    "[End Variable Update]",
  ].join("\n");
};

const parseTruthFlag = (value: unknown): boolean => {
  if (typeof value === "boolean") return value;
  if (typeof value === "number") return value > 0;
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (!normalized) return false;
    if (["yes", "true", "stroke", "positive", "detected"].includes(normalized))
      return true;
    const numeric = Number(normalized);
    if (!Number.isNaN(numeric)) {
      return numeric > 0;
    }
  }
  return false;
};

const determinePredictedStroke = (
  diagnosis: string | undefined,
  probability?: number
): boolean => {
  if (diagnosis) {
    const normalized = diagnosis.toLowerCase();
    if (normalized.includes("non-stroke")) {
      return false;
    }
    if (normalized.includes("stroke")) {
      return true;
    }
  }
  if (typeof probability === "number") {
    return probability >= 0.5;
  }
  return false;
};

const determineScenario = (
  predicted: boolean,
  truth: boolean
): OutcomeScenario => {
  if (predicted && truth) return OUTCOME_SCENARIOS.STROKE_DETECTED;
  if (!predicted && truth) return OUTCOME_SCENARIOS.STROKE_MISSED;
  if (predicted && !truth) return OUTCOME_SCENARIOS.FALSE_POSITIVE;
  return OUTCOME_SCENARIOS.TRUE_NEGATIVE;
};

export type PatientOutcomeEntry = {
  patientId: string;
  scenario: OutcomeScenario;
  prompt: string;
  narrative: string;
  predictedStroke: boolean;
  truthStroke: boolean;
  diagnosis?: string;
  strokeProbability?: number;
};

export const scenarioLabel = (scenario: OutcomeScenario): string => {
  switch (scenario) {
    case OUTCOME_SCENARIOS.STROKE_DETECTED:
      return "Stroke detected";
    case OUTCOME_SCENARIOS.STROKE_MISSED:
      return "Stroke missed";
    case OUTCOME_SCENARIOS.FALSE_POSITIVE:
      return "False positive";
    case OUTCOME_SCENARIOS.TRUE_NEGATIVE:
      return "True negative";
    default:
      return scenario;
  }
};

export const derivePatientOutcomes = (
  agentPatients: AgentPatientResult[],
  groundTruth: GroundTruthRow[],
  environment: OutcomeEnvironment
): { entries: PatientOutcomeEntry[]; summary: OutcomeCounters } => {
  const truthMap = new Map<string, GroundTruthRow>();
  groundTruth.forEach((row) => {
    truthMap.set(String(row["Patient#"]).trim(), row);
  });

  let counters = createOutcomeCounters();
  const entries: PatientOutcomeEntry[] = [];

  for (const patient of agentPatients) {
    const patientId = patient.patient_id ?? String(entries.length + 1);
    const truthRow = truthMap.get(String(patientId)) ?? null;
    const truthStroke = parseTruthFlag(truthRow?.["True Stroke?"]);
    const predictedStroke = determinePredictedStroke(
      patient.diagnosis,
      patient.stroke_probability
    );
    const scenario = determineScenario(predictedStroke, truthStroke);
    const config = getOutcomePromptConfig(environment, scenario);
    const patientNarrative = buildPatientNarrative({
      patientId,
      predictedStroke,
      truthStroke,
    });
    const updatedCounters = config
      ? applyOutcomeUpdates(counters, config.updates)
      : counters;
    const prompt = config
      ? formatOutcomePrompt(config, {
          after: updatedCounters,
          patientNarrative,
        })
      : patientNarrative;

    entries.push({
      patientId,
      scenario,
      prompt,
      narrative: patientNarrative,
      predictedStroke,
      truthStroke,
      diagnosis: patient.diagnosis,
      strokeProbability: patient.stroke_probability,
    });

    counters = updatedCounters;
  }

  return { entries, summary: counters };
};
