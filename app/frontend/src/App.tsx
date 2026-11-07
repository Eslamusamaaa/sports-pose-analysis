import { useCallback, useEffect, useRef, useState } from 'react'
import { getHealth, getJob, getLogs, getResult, uploadVideo } from './api'
import { JobPicker } from './components/JobPicker'
import { LiveView } from './components/LiveView'
import { LogPanel } from './components/LogPanel'
import { MMActionResult } from './components/MMActionResult'
import { OpenPoseResult } from './components/OpenPoseResult'
import { PipelineStatus } from './components/PipelineStatus'
import { ModelSetup } from './components/ModelSetup'
import { SportNav } from './components/SportNav'
import { TimelinePanel } from './components/TimelinePanel'
import { UploadPanel } from './components/UploadPanel'
import type { HealthResponse, JobStatusResponse, LogEntry, ResultResponse } from './types'

const TERMINAL = new Set(['completed', 'failed', 'partial'])

function EnvBanner({ health }: { health: HealthResponse | null }) {
  if (!health) return null
  const problems = [...health.openpose_problems, ...health.mmaction_problems]
  const mm = health.environment.mmaction as Record<string, unknown>
  const op = health.environment.openpose as Record<string, unknown>
  return (
    <section className="panel">
      <h2>Environment</h2>
      {problems.length > 0 ? (
        <div className="notice err">
          <strong>Not ready:</strong>
          <pre>{problems.join('\n')}</pre>
        </div>
      ) : (
        <div className="notice ok">
          Pose models ready:{' '}
          {(health.pose_engines ?? []).filter((e) => e.ready).map((e) => e.label).join(' · ') || 'OpenPose BODY_25'}
          {' '}· MMAction2 ready.
        </div>
      )}
      <div className="grid2" style={{ marginTop: 14 }}>
        <dl className="stats">
          <dt>OpenPose build</dt>
          <dd>{health.openpose_version ?? 'unknown'}</dd>
          <dt>OpenPose device</dt>
          <dd>{String(op.device)}</dd>
          <dt>BODY_25 weights</dt>
          <dd>{op.body25_weights_found ? 'found' : 'MISSING'}</dd>
          <dt>Frame cap</dt>
          <dd>{String(op.max_frames)}</dd>
        </dl>
        <dl className="stats">
          <dt>Model</dt>
          <dd>
            {String(mm.model_name)} · {String(mm.dataset)}
          </dd>
          <dt>torch</dt>
          <dd>{String(mm.torch)}</dd>
          <dt>CUDA available</dt>
          <dd>{mm.cuda_available ? 'yes' : 'no'}</dd>
          <dt>Device (effective)</dt>
          <dd>
            {String(mm.device_effective)}
            {mm.device_requested !== mm.device_effective && (
              <span className="muted"> (requested {String(mm.device_requested)})</span>
            )}
          </dd>
        </dl>
      </div>
    </section>
  )
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<JobStatusResponse | null>(null)
  const [result, setResult] = useState<ResultResponse | null>(null)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [error, setError] = useState<string | null>(null)
  const [uploadPct, setUploadPct] = useState(0)
  const [busy, setBusy] = useState(false)
  // Step 1: the pose model is chosen explicitly before live or upload; one
  // choice drives both. Nothing is preselected.
  const [engine, setEngine] = useState<string | null>(null)
  const [liveBusy, setLiveBusy] = useState(false)
  const timer = useRef<number | null>(null)

  useEffect(() => {
    getHealth().then(setHealth).catch((e) => setError(`Cannot reach backend: ${e.message}`))
  }, [])

  const stopPolling = () => {
    if (timer.current !== null) {
      window.clearInterval(timer.current)
      timer.current = null
    }
  }

  // Poll status + logs while the job runs; fetch the full result when terminal.
  useEffect(() => {
    if (!jobId) return
    const tick = async () => {
      try {
        const [s, l] = await Promise.all([getJob(jobId), getLogs(jobId)])
        setJob(s)
        setLogs(l.logs)
        if (TERMINAL.has(s.status)) {
          stopPolling()
          setBusy(false)
          setResult(await getResult(jobId))
        }
      } catch (e) {
        setError((e as Error).message)
        stopPolling()
        setBusy(false)
      }
    }
    void tick()
    timer.current = window.setInterval(tick, 2000)
    return stopPolling
  }, [jobId])

  const onOpenJob = useCallback((id: string) => {
    setError(null)
    setResult(null)
    setJob(null)
    setLogs([])
    setBusy(false)
    setJobId(id)
  }, [])

  const onAnalyze = useCallback(async (file: File) => {
    setError(null)
    setResult(null)
    setJob(null)
    setLogs([])
    setUploadPct(0)
    setBusy(true)
    try {
      if (!engine) throw new Error('Choose a pose model first')
      const { job_id } = await uploadVideo(file, setUploadPct, 'karate', engine)
      setJobId(job_id)
    } catch (e) {
      setError((e as Error).message)
      setBusy(false)
    }
  }, [engine])

  const mmStage = job?.stages.find((s) => s.name === 'mmaction')
  const showResults = result !== null

  return (
    <div className="wrap">
      <header className="top">
        <h1>Karate AI — pipeline proof of concept</h1>
        <p>
          Browser → FastAPI → pose model (OpenPose / RTMPose / YOLO11) → skeleton → COCO-17 → MMAction2
          ST-GCN → results
        </p>
        <SportNav active="karate" />
      </header>

      <EnvBanner health={health} />

      <ModelSetup
        engines={health?.pose_engines}
        value={engine}
        onChange={setEngine}
        locked={busy || liveBusy}
        actionModel={{
          name: 'MMAction2 ST-GCN (NTU RGB+D 60)',
          detail: '60 everyday actions, not karate-trained',
        }}
      />

      <JobPicker currentJobId={jobId} onOpen={onOpenJob} sport="karate" />

      <LiveView engine={engine} onRunningChange={setLiveBusy} />

      <UploadPanel
        busy={busy}
        uploadPct={uploadPct}
        onAnalyze={onAnalyze}
        blockedReason={!engine ? 'Choose a pose model first (step 1)' : undefined}
      />

      {error && (
        <section className="panel">
          <div className="notice err">
            <strong>Error</strong>
            <pre>{error}</pre>
          </div>
        </section>
      )}

      {job && (
        <>
          <PipelineStatus stages={job.stages} poseEngine={job.pose_engine} />
          {job.status === 'partial' && (
            <section className="panel">
              <div className="notice warn">
                <strong>Partial success.</strong> Earlier stages completed; a later stage
                failed. Each stage is reported independently below so it can be debugged
                on its own.
                {job.error && <pre>{job.error}</pre>}
              </div>
            </section>
          )}
          {job.status === 'failed' && (
            <section className="panel">
              <div className="notice err">
                <strong>Pipeline failed at stage: {job.stage}</strong>
                {job.error && <pre>{job.error}</pre>}
              </div>
            </section>
          )}
        </>
      )}

      {showResults && result.openpose && <OpenPoseResult result={result} />}
      {showResults && <MMActionResult result={result} mmStage={mmStage} />}
      {showResults && result.mmaction?.timeline && (
        <TimelinePanel timeline={result.mmaction.timeline} />
      )}

      {(logs.length > 0 || result) && (
        <LogPanel logs={logs} outputs={result?.outputs ?? null} poseEngine={job?.pose_engine ?? result?.pose_engine} />
      )}

      {job && (
        <div className="small muted" style={{ marginTop: 16 }}>
          job_id <code>{job.job_id}</code> · status <code>{job.status}</code> · stage{' '}
          <code>{job.stage}</code>
        </div>
      )}
    </div>
  )
}
