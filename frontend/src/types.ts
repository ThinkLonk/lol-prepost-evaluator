export type Side = 'blue' | 'red'
export type Role = 'TOP' | 'JUNGLE' | 'MID' | 'BOT' | 'SUPPORT'
export type Entity = { id: string; name: string; image?: string | null; history_count?: number | null }
export type Slot = { role: Role; player_id: string }
export type Metric = { brier_score: number; log_loss: number; roc_auc: number | null; sample_count: number }
export type Bootstrap = {
  teams: Entity[]; players: Entity[]; champions: Entity[]; roles: Role[]
  rosters: Record<string, { slots: Slot[]; source_ended_at: string }>
  metrics: { partition: string; count: number; selected_family: string; dataset_id: string;
    split_id: string; model_set_id: string; scores: { family: string; pre: Metric; post: Metric }[] }
  history_latest: string; load_ms: number
}
export type Context = {
  blue_team_id: string; red_team_id: string; blue_roster: Slot[]; red_roster: Slot[]
  patch: string; context_label: string; user_confirmed: boolean
}
export type Probability = { blue_win_probability: number; red_win_probability: number }
export type ModelRow = {
  family: string; label: string; pre?: Probability; post?: Probability; delta?: Record<Side, number>
}
export type Coverage = {
  pre_used: { game_count: number; latest_ended_at: string | null; age_days_at_pre: number | null }
  teams: { team_id: string; pre_used: { game_count: number; latest_ended_at: string | null } }[]
  patch_observed_in_history: boolean; patch_scope_status: string
}
export type Stat = { value: number | null; sample_count: number; win_count: number | null; missing: boolean }
export type Warning = { warning_code: string; message: string; warning_group?: string }
export type Result = {
  phase: 'PRE' | 'POST'; active: boolean; teams: Record<Side, Entity>; patch: string
  models: ModelRow[]; coverage: Coverage | null; warnings: Warning[]; history_cutoff_at: string
  pre_evaluation_id?: number; post_evaluation_id?: number | null; evaluation_id?: number
  context_label?: string | null
  roster?: Record<Side, { role: Role; player: Entity; champion: Entity | null }[]>
  statistics?: Record<Side, Record<'recent_form' | 'side_win_rate' | 'roster_continuity', Stat>> & { head_to_head: Stat }
  player_champion?: { side: 'BLUE' | 'RED'; role: Role; games_count: number; wins_count: number; win_rate: number | null }[]
  metadata?: Record<string, string>
  timing_ms?: { analysis: number; details: number; save: number }
}
export type Session = { session_id: string; pre: Result | null; post: Result | null;
  pending_pre?: { operation_id: string; context: Context } | null;
  pending_post?: { operation_id: string; pre_evaluation_id: number; champion_ids: string[] } | null }
