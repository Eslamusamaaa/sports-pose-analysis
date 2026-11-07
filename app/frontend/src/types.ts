export type StageState = 'pending' | 'running' | 'done' | 'failed' | 'skipped'
export type JobStatus = 'queued' | 'running' | 'completed' | 'failed' | 'partial'

export interface StageInfo {
  name: string
  state: StageState
  duration_s: number | null
  error: string | null
}

export interface JobStatusResponse {
  job_id: string
  status: JobStatus
  stage: string
  filename: string
  sport?: string
  pose_engine?: string
  size_bytes: number
  error: string | null
  created_at: string
  updated_at: string
  stages: StageInfo[]
}

export interface VideoMeta {
  width: number
  height: number
  fps: number
  duration_s: number | null
  frame_count: number | null
  codec: string
  size_bytes: number
  analysis_frame_limit?: number
}

export interface OpenPoseInfo {
  command: string[]
  cwd: string
  exit_code: number
  duration_s: number
  json_frame_count: number
  json_dir: string
  video_written: boolean
  mp4_written?: boolean
  net_resolution: string
  model_pose: string
  device: string
  seconds_per_frame: number
  openpose_version: string | null
  engine?: string
  engine_label?: string
  keypoint_source?: string
  overlay?: string
  players?: { labels: string[]; track_ids: number[]; frames_present: Record<string, number> }
}

export interface TrackInfo {
  track_id: number
  frames_present: number
  mean_confidence: number
  mean_valid_keypoints: number
  score: number
}

export interface SkeletonInfo {
  keypoint_format: string
  num_keypoints: number
  frame_count: number
  total_person_detections: number
  avg_people_per_frame: number
  max_people_in_a_frame: number
  track_count: number
  tracks: TrackInfo[]
  tracker?: string
  boxes_available?: boolean
  json_versions: string[]
  per_person_array_shape: string
  sample_frames: unknown[]
  skeleton_json_path?: string
  openpose_json_dir?: string
}

export interface ConversionInfo {
  source_format: string
  source_num_keypoints: number
  target_format: string
  target_num_keypoints: number
  target_layout: string
  num_frames: number
  num_persons: number
  track_ids_used: number[]
  slots_filled: number
  slots_zero_padded: number
  coordinate_space: string
  tensor_shape_note: string
  dropped_count: number
  mapping: { coco_index: number; coco_name: string; body25_index: number; body25_name: string }[]
  dropped_body25_joints: { body25_index: number; body25_name: string }[]
}

export interface Prediction {
  rank: number
  class_index: number
  label: string
  score: number
}

export interface MMActionInfo {
  top1: Prediction | null
  topk: Prediction[]
  num_classes: number
  model_name: string
  dataset: string
  graph_layout: string | null
  graph_num_node: number | null
  num_keypoints_sent: number
  num_frames_sent: number
  num_persons_sent: number
  img_shape: number[]
  device_requested: string
  device_used: string
  torch_version: string
  cuda_available: boolean
  model_load_s: number
  inference_s: number
  domain_warning: string
  note: string | null
  analysis_mode?: string
  timeline?: import('./components/TimelinePanel').TimelineData | null
  swimming?: SwimAnalysis
}

export interface SwimWindow {
  start_s: number
  end_s: number
  start_frame: number
  end_frame: number
  stroke: string
  confidence: number
  scores: Record<string, number>
}

export interface SwimSegment {
  stroke: string
  start_s: number
  end_s: number
  start_frame: number
  end_frame: number
  mean_confidence: number
  num_windows: number
}

export interface CoachingTip {
  metric: string
  message: string
  severity: 'info' | 'warn'
  start_s?: number
  end_s?: number
}

export interface SwimAnalysis {
  dominant_stroke: string | null
  distribution: { stroke: string; windows: number; share: number }[]
  uncertain_windows: number
  not_visible_windows?: number
  visible_frames?: number
  total_frames?: number
  num_windows: number
  windows: SwimWindow[]
  segments: SwimSegment[]
  min_confidence: number
  window_seconds: number
  stride_seconds: number
  input_fps: number
  model_fps: number
  frame_step: number
  duration_s: number
  model: { name: string; dataset: string; classes: string[] }
  domain_warning: string
  coaching: { status: string; message: string; tips: CoachingTip[] }
}

export interface OutputEntry {
  path: string
  exists: boolean
  type: string
  size_bytes: number
  file_count?: number
}

export interface ResultResponse {
  job_id: string
  status: JobStatus
  stage: string
  filename: string
  sport?: string
  pose_engine?: string
  size_bytes: number
  error: string | null
  video_meta: VideoMeta | null
  openpose: OpenPoseInfo | null
  skeleton: SkeletonInfo | null
  conversion: ConversionInfo | null
  mmaction: MMActionInfo | null
  model_info: {
    model_name: string
    dataset: string
    keypoint_layout: string
    config: string
    checkpoint: string
    status_banner: string
    disclaimer: string
  }
  stages: StageInfo[]
  outputs: Record<string, OutputEntry>
}

export interface LogEntry {
  ts: string
  level: string
  stage: string | null
  message: string
}

export interface HealthResponse {
  status: string
  openpose_ready: boolean
  openpose_problems: string[]
  mmaction_ready: boolean
  mmaction_problems: string[]
  openpose_version: string | null
  pose_engines?: import('./components/UploadPanel').PoseEngine[]
  swimming?: {
    model_ready: boolean
    checkpoint: string
    window_frames: number
    stride_frames: number
    min_confidence: number
  }
  environment: {
    openpose: Record<string, unknown>
    mmaction: Record<string, unknown>
    ffmpeg: Record<string, unknown>
  }
}
