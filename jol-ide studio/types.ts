export enum CommandKind {
  SEARCH = 'Search',           // Command<Search>
  TRY = 'Try',                 // Command<Try>
  JUDGE = 'Judge',             // Command<Judge>
  COMMUNICATE = 'Communicate' // Command<Communicate>
}

export type OrderType = CommandKind;

export interface TeamWorker {
  id: string;
  name: string;
  commandKind: CommandKind;
  size: number;
  distribution: 'round_robin' | 'weighted' | 'priority';
  connector?: string;
}

export interface MicroCheckResult {
  index: number;
  item: string;
  status: 'passed' | 'failed';
  workerId: number;
  error?: string;
}

export interface MicroCheckpoint {
  id: string;
  name: string;
  assignedTeam: string;
  batchItems: string[];
  strategy: 'round_robin' | 'parallel';
  threshold: number;
  passedCount: number;
  totalCount: number;
  results: MicroCheckResult[];
}

export interface Order {
  id: string;
  type: CommandKind;
  content: string;
  status: 'pending' | 'active' | 'completed' | 'failed';
  assignedWorker?: string;
  result?: any;
  monolithDialogue?: { question: string; answer: string }[];
}

export interface Checkpoint {
  id: string;
  name: string;
  report?: string;
  retainedVars?: string[];
  prunedVars?: string[];
  microCheckpoints?: MicroCheckpoint[];
}

export interface Flow {
  id: string;
  name: string;
  usingTeams: string[];
  teams: Record<string, TeamWorker>;
  checkpoints: Checkpoint[];
  currentCheckpointIndex: number;
  mergePolicy: 'last_wins' | 'deep_merge' | 'crdt';
  historyLog: Order[];
}

export interface SystemChainNode {
  id: string;
  name: string;
  order?: Order;
  nextId?: string;
  prevId?: string;
  impactLevel: number; // Sensitivity Damping intensity (0.0 to 1.0)
  decayFactor?: number;
  echoAnalysis?: string;
}

export interface ProcessTreeNode {
  id: string;
  name: string;
  geneticCode?: string; // Binary path (e.g. 0101)
  type: 'root' | 'branch' | 'leaf';
  status: 'healthy' | 'atrophied' | 'expanded' | 'completed';
  children?: ProcessTreeNode[];
}

export interface SimulationFile {
  name: string;
  content: string;
  status: 'healthy' | 'atrophied' | 'verified' | 'deployed';
}

export interface EvalContextState {
  currentStage: string;
  variables: Record<string, any>;
  reports: string[];
  checkpointIndex: number;
}

export type ProfessionalDomain = 'digital' | 'economic' | 'cyber' | 'mechanical' | 'electro' | 'clinical';

export interface McpToolCall {
  id: string;
  tool: string;
  arguments: Record<string, any>;
  timestamp: string;
  status: 'success' | 'pending' | 'error';
  result?: any;
}

export interface SimulationState {
  flow: Flow | null;
  chain: SystemChainNode[];
  tree: ProcessTreeNode | null;
  files: SimulationFile[];
  resources: Record<string, any>;
  contextState?: EvalContextState;
  lastUpdate: string;
  isSimulating: boolean;
  mcpConnected?: boolean;
  mcpCalls?: McpToolCall[];
}