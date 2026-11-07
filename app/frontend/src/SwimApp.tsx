import { useCallback, useEffect, useRef, useState } from 'react'
import { getHealth, getJob, getLogs, getResult, uploadVideo } from './api'
import { JobPicker } from './components/JobPicker'
import { LogPanel } from './components/LogPanel'
import { PipelineStatus } from './components/PipelineStatus'
import { ModelSetup } from './components/ModelSetup'
import { SportNav } from './components/SportNav'
import { SwimLiveView } from './components/SwimLiveView'
import { SwimResult } from './components/SwimResult'
import { UploadPanel } from './components/UploadPanel'
import type { HealthResponse, JobStatusResponse, LogEntry, ResultResponse } from './types'

const TERMINAL = new Set(['completed', 'failed', 'partial'])

function SwimEnv({ health }: { health: HealthResponse | null }) {
  if (!health) return null
  const sw = health.swimming
  const problems = [...health.openpose_problems, ...health.mmaction_problems]
  return (
    <section className="panel">
      <h2>Environment</h2>
      {problems.length > 0 && (
        <div className="notice err" style={{ marginBottom: 10 }}>
          <strong>Not ready:</strong>
          <pre>{problems.join('\n')}</pre>
        </div>
      )}
      {sw?.model_ready ? (
        <div className="notice ok">
          Swimming model ready: ST-GCN fine-tuned on SwimXYZ · 4 strokes · {sw.window_frames}
          -frame windows @ 30 fps.
        </div>
      ) : (
        <div className="notice warn">
          The swimming model is still being trained. Uploads are disabled until it is
          installed in <code>models/swim/</code>.
        </div>
      )}
      <div className="small muted" style={{ marginTop: 10 }}>
        Best results: one swimmer, side view above water, the whole body in frame. The
        model knows 4 strokes only - dives, turns and standing show as “uncertain”.
      </div>
    </section>
  )
}

export default function SwimApp() {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<JobStatusResponse | null>(null)
  const [result, setResult] = useState<ResultResponse | null>(null)
  const [logs, setLogs] = useState<LogEntry[]>([])
  const [error, setError] = useState<string | null>(null)
  const [uploadPct, setUploadPct] = useState(0)
  const [busy, setBusy] = useState(false)
  // Step 1: pose model chosen explicitly before live or upload (one choice for
  // both). For pool video RTMPose found the swimmer most often in our tests
  // (69% of frames vs YOLO11 34%, OpenPose 25%); YOLO11 is fastest live.
  const [engine, setEngine] = useState<string | null>(null)
  const [liveBusy, setLiveBusy] = useState(false)
  const timer = useRef<number | null>(null)

  useEffect(() => {
    document.title = 'Swimming AI'
    // Retry until the backend answers: a page opened while the backend was
    // restarting used to keep its buttons disabled until a manual refresh.
    let cancelled = false
    let timer: number | undefined
    const load = () =>
      getHealth()
        .then((h) => {
          if (!cancelled) {
            setHealth(h)
            setError(null)
          }
        })
        .catch((e) => {
          if (cancelled) return
          setError(`Cannot reach backend (retrying): ${e.message}`)
          timer = window.setTimeout(load, 3000)
        })
    void load()
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [])

  const stopPolling = () => {
    if (timer.current !== null) {
      window.clearInterval(timer.current)
      timer.current = null
    }
  }

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
      const { job_id } = await uploadVideo(file, setUploadPct, 'swimming', engine)
      setJobId(job_id)
    } catch (e) {
      setError((e as Error).message)
      setBusy(false)
    }
  }, [engine])

  const ready = !!health?.swimming?.model_ready

  return (
    <div className="wrap">
      <header className="top">
        <h1>Swimming AI — stroke analysis</h1>
        <p>Video → pose model (OpenPose / RTMPose / YOLO11) → one swimmer → COCO-17 → ST-GCN (SwimXYZ) → stroke timeline</p>
        <SportNav active="swimming" />
      </header>

      <SwimEnv health={health} />

      <ModelSetup
        engines={health?.pose_engines}
        value={engine}
        onChange={setEngine}
        locked={busy || liveBusy}
        actionModel={{
          name: 'ST-GCN fine-tuned on SwimXYZ',
          detail: 'our swimming model: freestyle / backstroke / breaststroke / butterfly',
        }}
      />

      <JobPicker currentJobId={jobId} onOpen={onOpenJob} sport="swimming" />

      <SwimLiveView
        modelReady={health ? ready : null}
        engine={engine}
        onRunningChange={setLiveBusy}
      />

      <UploadPanel
        busy={busy}
        uploadPct={uploadPct}
        onAnalyze={onAnalyze}
        blockedReason={
          !engine
            ? 'Choose a pose model first (step 1)'
            : health && !ready
              ? 'Swimming model not installed yet'
              : undefined
        }
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
          <PipelineStatus stages={job.stages} poseEngine={job.pose_engine} actionLabel="Swimming stroke model (ST-GCN, SwimXYZ)" />
          {job.status !== 'completed' && job.status !== 'running' && job.status !== 'queued' && (
            <section className="panel">
              <div className={`notice ${job.status === 'partial' ? 'warn' : 'err'}`}>
                <strong>
                  {job.status === 'partial' ? 'Partial success' : 'Failed'} at stage: {job.stage}
                </strong>
                {job.error && <pre>{job.error}</pre>}
              </div>
            </section>
          )}
        </>
      )}

      {result && <SwimResult result={result} />}

      {(logs.length > 0 || result) && <LogPanel logs={logs} outputs={result?.outputs ?? null} poseEngine={job?.pose_engine ?? result?.pose_engine} />}
    </div>
  )
}
