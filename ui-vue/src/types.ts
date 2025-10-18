export type DatasetName = string;

export interface DatasetSummary {
  slug?: string;
  name?: string;
  status?: string;
  message?: string | null;
  path?: string;
  progress?: number;
  downloaded?: number;
  total?: number;
  [key: string]: unknown;
}

export interface DatasetListPayload {
  datasets?: Array<DatasetSummary | DatasetName>;
  available?: DatasetName[];
  installed?: DatasetName[];
}

export type TrainingStatus = 'Idle' | 'Initializing' | 'Training' | 'Stopped';

export interface TrainingConfig {
  dataset: DatasetName;
  mode: 'tstep' | 'fpt';
  network_size: number;
  layers: number;
  lr: number;
  K: number;
  tol: number;
  T?: number;
}

export interface MetricPayload {
  epoch: number;
  step: number;
  loss: number;
  acc?: number;
  throughput?: number;
  lr?: number;
  residual?: number;
  k?: number;
}

export interface MetricEntry extends MetricPayload {
  at: number;
}

export interface SpikePayload {
  layer: number;
  t: number;
  neurons: number[];
  edges?: [number, number][];
}

export interface SpikeEntry extends SpikePayload {
  at: number;
}

export interface MessageEntry {
  subject: string;
  type?: string;
  at: number;
  payload?: unknown;
}

export interface LayerLayout {
  layer: number;
  count: number;
  startIndex: number;
  positions: Float32Array;
}

export interface LogPayload {
  ts: number;
  level: 'DEBUG' | 'INFO' | 'WARNING' | 'ERROR';
  message: string;
  metric?: Record<string, unknown>;
}

export interface LogEntry extends LogPayload {
  at: number;
}

export type SocketEnvelope =
  | { type: 'metrics'; data: MetricPayload }
  | { type: 'spikes'; data: SpikePayload }
  | { type: 'log'; data: LogPayload };
