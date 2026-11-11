import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { useSettingsStore } from '@/stores/settings'
import { backendBaseUrl } from '@/lib/constants'
import Button from '@/components/ui/Button'
import { cn } from '@/lib/utils'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow
} from '@/components/ui/Table'
import { Card } from '@/components/ui/Card'
import PaginationControls from '@/components/ui/PaginationControls'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription
} from '@/components/ui/Dialog'
import Badge from '@/components/ui/Badge'
import {
  getDocumentsPaginated,
  getAllWorkspacesPipelineStageRuns,
  reprocessFailedDocuments,
  getSystemResources,
  getWorkspaces,
  type SystemResourcesResponse,
  type WorkspaceItem,
  type DocStatusResponse,
  type PipelineStageRunResponse
} from '@/api/lightrag'
import { toast } from 'sonner'
import {
  RefreshCwIcon,
  RotateCcwIcon,
  AlertTriangleIcon,
  SearchIcon,
  XIcon,
  CopyIcon,
  CheckIcon,
  FileTextIcon,
  CpuIcon,
  HardDriveIcon,
  DatabaseIcon,
  ScaleIcon,
  CheckCircle2Icon,
  ClockIcon,
  ActivityIcon,
  ShieldAlertIcon,
  FileCode2Icon
} from 'lucide-react'

// Core pipeline stages for progress tracking
const PIPELINE_STAGES = [
  { id: 'extraction', label: 'Extract' },
  { id: 'chunking', label: 'Chunk' },
  { id: 'chunk_embedding', label: 'Chunk Embed' },
  { id: 'entity_extraction', label: 'KG Extract' },
  { id: 'kg_merge', label: 'KG Merge' },
  { id: 'kg_embedding', label: 'KG Embed' },
  { id: 'finalize', label: 'Finalize' }
]

const STATUS_CONFIGS: Record<string, { label: string; color: string; bg: string; border: string }> = {
  UPLOADING: { label: 'Uploading', color: 'text-sky-400', bg: 'bg-sky-500/10', border: 'border-sky-500/20' },
  EXTRACTING: { label: 'Extracting', color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/20' },
  EXTRACTED: { label: 'Extracted', color: 'text-blue-400', bg: 'bg-blue-500/10', border: 'border-blue-500/20' },
  CHUNKING: { label: 'Chunking / KG', color: 'text-cyan-400', bg: 'bg-cyan-500/10', border: 'border-cyan-500/20' },
  PROCESSED: { label: 'Processed', color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/20' },
  FAILED: { label: 'Failed', color: 'text-rose-400', bg: 'bg-rose-500/10', border: 'border-rose-500/20' }
}

export default function ImportMonitor() {
  const selectedWorkspaceId = useSettingsStore.use.selectedWorkspaceId()

  // State: System Telemetry
  const [telemetry, setTelemetry] = useState<SystemResourcesResponse | null>(null)

  // State: Workspaces
  const [workspaces, setWorkspaces] = useState<WorkspaceItem[]>([])
  const [filterWorkspace, setFilterWorkspace] = useState<string>('all')

  // State: Document List & Filter
  const [documents, setDocuments] = useState<DocStatusResponse[]>([])
  const [stageRuns, setStageRuns] = useState<PipelineStageRunResponse[]>([])
  const [statusCounts, setStatusCounts] = useState<Record<string, number>>({})
  const [loading, setLoading] = useState(false)
  const [reprocessing, setReprocessing] = useState(false)

  // Filters
  const [searchQuery, setSearchQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [docTypeFilter, setDocTypeFilter] = useState<'all' | 'legal' | 'general'>('all')

  // Pagination
  const [currentPage, setCurrentPage] = useState(1)
  const [pageSize, setPageSize] = useState(15)
  const [totalCount, setTotalCount] = useState(0)

  // SSE Push Stream State (Zero Polling)
  const [streamActive, setStreamActive] = useState<boolean>(true)
  const [streamStatus, setStreamStatus] = useState<'connected' | 'connecting' | 'paused' | 'error'>('connecting')
  const prevStatusCountsRef = useRef<Record<string, number>>({})

  // Detail Modal
  const [selectedDoc, setSelectedDoc] = useState<DocStatusResponse | null>(null)
  const [copiedId, setCopiedId] = useState<string | null>(null)

  // Load Workspaces
  useEffect(() => {
    getWorkspaces()
      .then((res) => {
        const items = res.workspace_items || []
        setWorkspaces(items)
      })
      .catch((err) => console.error('Failed to load workspaces:', err))
  }, [])

  // Fetch System Telemetry
  const fetchTelemetry = useCallback(async () => {
    try {
      const data = await getSystemResources()
      setTelemetry(data)
    } catch (err) {
      console.error('Failed to fetch system resources telemetry:', err)
    }
  }, [])

  // Fetch Documents & Active Stage Runs
  const fetchDocuments = useCallback(async () => {
    try {
      setLoading(true)
      const targetWorkspace = filterWorkspace === 'all' ? 'all' : filterWorkspace

      // 1. Fetch paginated documents
      const docsPromise = getDocumentsPaginated({
        status_filter: statusFilter === 'all' ? undefined : (statusFilter as any),
        page: currentPage,
        page_size: pageSize,
        sort_field: 'updated_at',
        sort_direction: 'desc',
        workspace_id: targetWorkspace,
        doc_type: docTypeFilter === 'all' ? undefined : docTypeFilter,
      })

      // 2. Fetch stage runs for active stage tracking
      const stagesPromise = getAllWorkspacesPipelineStageRuns({
        page: 1,
        page_size: 50,
        workspace_id: targetWorkspace === 'all' ? undefined : targetWorkspace
      }).catch(() => null)

      const [docsRes, stageRes] = await Promise.all([docsPromise, stagesPromise])

      setDocuments(docsRes.documents || [])
      setTotalCount(docsRes.pagination?.total_count || 0)
      setStatusCounts(docsRes.status_counts || {})

      if (stageRes && stageRes.runs) {
        setStageRuns(stageRes.runs)
      }
    } catch (err: any) {
      console.error('Failed to fetch documents for import monitor:', err)
    } finally {
      setLoading(false)
    }
  }, [filterWorkspace, statusFilter, currentPage, pageSize, docTypeFilter])

  // Silent document reload when stage progress updates (avoids table flashing)
  const fetchDocumentsSilently = useCallback(async () => {
    try {
      const targetWorkspace = filterWorkspace === 'all' ? 'all' : filterWorkspace
      const docsRes = await getDocumentsPaginated({
        status_filter: statusFilter === 'all' ? undefined : (statusFilter as any),
        page: currentPage,
        page_size: pageSize,
        sort_field: 'updated_at',
        sort_direction: 'desc',
        workspace_id: targetWorkspace,
        doc_type: docTypeFilter === 'all' ? undefined : docTypeFilter,
      })
      if (docsRes?.documents) {
        setDocuments(docsRes.documents)
        setTotalCount(docsRes.pagination?.total_count || 0)
      }
    } catch {
      // silent catch for background live refresh
    }
  }, [filterWorkspace, statusFilter, currentPage, pageSize, docTypeFilter])

  // Server-Sent Events (SSE) Live Push Stream - ZERO CLIENT POLLING
  useEffect(() => {
    // 1. Initial snapshot fetch
    fetchTelemetry()
    fetchDocuments()

    if (!streamActive) {
      setStreamStatus('paused')
      return
    }

    setStreamStatus('connecting')
    const abortController = new AbortController()
    let retryTimer: ReturnType<typeof setTimeout> | null = null

    const targetWorkspace = filterWorkspace === 'all' ? 'all' : filterWorkspace
    const streamUrl = `${backendBaseUrl}/pipeline/stream?workspace_id=${encodeURIComponent(targetWorkspace)}&interval=2.0`

    const startStreaming = async () => {
      try {
        const token = localStorage.getItem('LIGHTRAG-API-TOKEN')
        const apiKey = useSettingsStore.getState().apiKey
        const headers: Record<string, string> = { Accept: 'text/event-stream' }
        if (token) headers['Authorization'] = `Bearer ${token}`
        if (apiKey) headers['X-API-Key'] = apiKey

        const response = await fetch(streamUrl, {
          headers,
          signal: abortController.signal
        })

        if (!response.ok) {
          throw new Error(`SSE stream HTTP error: ${response.status}`)
        }

        setStreamStatus('connected')

        const reader = response.body?.getReader()
        if (!reader) return

        const decoder = new TextDecoder()
        let buffer = ''

        while (!abortController.signal.aborted) {
          const { done, value } = await reader.read()
          if (done) break

          buffer += decoder.decode(value, { stream: true })
          const parts = buffer.split('\n\n')
          buffer = parts.pop() || ''

          for (const part of parts) {
            if (!part.trim()) continue
            let eventType = 'message'
            let dataText = ''

            for (const line of part.split('\n')) {
              if (line.startsWith('event:')) {
                eventType = line.slice(6).trim()
              } else if (line.startsWith('data:')) {
                dataText = line.slice(5).trim()
              }
            }

            if (!dataText) continue

            try {
              const payload = JSON.parse(dataText)
              if (eventType === 'telemetry') {
                setTelemetry(payload)
              } else if (eventType === 'pipeline') {
                const newCounts = payload.status_counts || {}
                setStatusCounts(newCounts)

                // If documents in flight changed or active documents exist, update document table silently
                const prevCounts = prevStatusCountsRef.current
                const countsChanged = JSON.stringify(prevCounts) !== JSON.stringify(newCounts)
                const inProgress = (newCounts['EXTRACTING'] || 0) + (newCounts['CHUNKING'] || 0) + (newCounts['UPLOADING'] || 0)

                if (countsChanged || inProgress > 0) {
                  fetchDocumentsSilently()
                }
                prevStatusCountsRef.current = newCounts
              }
            } catch (err) {
              console.warn('Error parsing SSE event payload:', err)
            }
          }
        }
      } catch (err: any) {
        if (!abortController.signal.aborted) {
          console.warn('Pipeline SSE Stream disconnected, reconnecting in 5s...', err)
          setStreamStatus('error')
          retryTimer = setTimeout(() => {
            if (!abortController.signal.aborted && streamActive) {
              startStreaming()
            }
          }, 5000)
        }
      }
    }

    startStreaming()

    // Handle visibility: disconnect when hidden to conserve server resources, reconnect when visible
    const handleVisibility = () => {
      if (document.visibilityState === 'hidden') {
        abortController.abort()
      } else if (document.visibilityState === 'visible' && streamActive) {
        fetchDocuments()
        startStreaming()
      }
    }
    document.addEventListener('visibilitychange', handleVisibility)

    return () => {
      abortController.abort()
      if (retryTimer) clearTimeout(retryTimer)
      document.removeEventListener('visibilitychange', handleVisibility)
    }
  }, [streamActive, filterWorkspace, fetchTelemetry, fetchDocuments, fetchDocumentsSilently])

  // Reprocess failed documents action
  const handleReprocessFailed = async () => {
    try {
      setReprocessing(true)
      const res = await reprocessFailedDocuments(
        undefined,
        docTypeFilter === 'all' ? undefined : docTypeFilter
      )
      toast.success(res.message || 'Triggered reprocessing for failed documents')
      fetchDocuments()
    } catch (err: any) {
      toast.error(`Reprocess error: ${err.message || err}`)
    } finally {
      setReprocessing(false)
    }
  }

  // Copy helper
  const handleCopy = (text: string, id: string) => {
    navigator.clipboard.writeText(text)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 2000)
    toast.success('Copied to clipboard')
  }

  // Helper to determine if a document is a legal document
  const checkIsLegal = (doc: any): boolean => {
    if (!doc) return false
    if (doc.metadata?.doc_type === 'legal' || doc.doc_type === 'legal') return true
    if (doc.metadata?.document_code || doc.metadata?.legal_document_type) return true
    if (typeof doc.id === 'string') {
      if (doc.id.startsWith('legal-')) return true
      if (/^\d+$/.test(doc.id)) return true
    }
    if (doc.file_path?.includes('legal_docs/')) return true
    return false
  }

  // Filtered documents by search query and docType
  const filteredDocuments = useMemo(() => {
    return documents.filter((doc) => {
      // Doc type filter
      const isLegal = checkIsLegal(doc)
      if (docTypeFilter === 'legal' && !isLegal) return false
      if (docTypeFilter === 'general' && isLegal) return false

      // Search query
      if (!searchQuery.trim()) return true
      const q = searchQuery.toLowerCase()
      const name = (doc.file_path || doc.id || '').toLowerCase()
      const errorMsg = (doc.error_msg || '').toLowerCase()
      return name.includes(q) || doc.id.toLowerCase().includes(q) || errorMsg.includes(q)
    })
  }, [documents, searchQuery, docTypeFilter])

  // Compute status totals
  const totalDocs = useMemo(() => {
    return Object.values(statusCounts).reduce((acc, curr) => acc + curr, 0)
  }, [statusCounts])

  const processingCount = useMemo(() => {
    return (
      (statusCounts.UPLOADING || 0) +
      (statusCounts.EXTRACTING || 0) +
      (statusCounts.EXTRACTED || 0) +
      (statusCounts.CHUNKING || 0)
    )
  }, [statusCounts])

  // Find active stage run for a doc_id
  const getDocStageInfo = (docId: string) => {
    const cleanId = docId.replace(/^legal-/, '')
    return stageRuns.find((r) => r.doc_id === docId || r.doc_id === cleanId || r.run_id === docId)
  }

  return (
    <div className="flex flex-col h-full bg-background text-foreground overflow-y-auto px-4 py-4 md:px-8 space-y-5">
      {/* 1. Header Section: Title + Controls */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-border/40 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
            <h1 className="text-xl md:text-2xl font-bold tracking-tight flex items-center gap-2 font-sans">
              Document Import & Pipeline Telemetry
            </h1>
            <Badge variant="outline" className="text-xs font-mono border-emerald-500/30 text-emerald-400 bg-emerald-500/10">
              LIVE
            </Badge>
          </div>
          <p className="text-xs md:text-sm text-muted-foreground mt-0.5">
            Real-time pipeline monitoring, ingestion lifecycle status, and system overload protection.
          </p>
        </div>

        {/* Global Controls: Auto-refresh, Manual trigger, Reprocess */}
        <div className="flex flex-wrap items-center gap-2">
          {/* SSE Stream Status Indicator & Toggle (Zero Polling) */}
          <div className="flex items-center gap-2 bg-card/90 border border-border/70 rounded-lg px-2.5 py-1 text-xs">
            {streamActive && streamStatus === 'connected' ? (
              <div className="flex items-center gap-1.5">
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                <span className="font-medium text-emerald-400">SSE Push Stream</span>
                <span className="text-[10px] text-muted-foreground hidden sm:inline">(Zero Polling)</span>
              </div>
            ) : streamActive && streamStatus === 'connecting' ? (
              <div className="flex items-center gap-1.5">
                <RefreshCwIcon className="size-3 text-amber-400 animate-spin" />
                <span className="text-amber-400">Connecting Stream...</span>
              </div>
            ) : streamActive && streamStatus === 'error' ? (
              <div className="flex items-center gap-1.5">
                <AlertTriangleIcon className="size-3 text-rose-400" />
                <span className="text-rose-400">Stream Offline</span>
              </div>
            ) : (
              <div className="flex items-center gap-1.5">
                <span className="h-2 w-2 rounded-full bg-zinc-500"></span>
                <span className="text-muted-foreground">Stream Paused</span>
              </div>
            )}

            <button
              type="button"
              onClick={() => setStreamActive((prev) => !prev)}
              className="ml-1 pl-1.5 border-l border-border text-[11px] font-medium text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
              title={streamActive ? 'Pause real-time SSE stream' : 'Resume real-time SSE stream'}
            >
              {streamActive ? 'Pause' : 'Resume'}
            </button>
          </div>

          {/* Manual Refresh */}
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              fetchTelemetry()
              fetchDocuments()
            }}
            disabled={loading}
            className="h-8 gap-1.5 text-xs transition-all active:scale-[0.98]"
          >
            <RefreshCwIcon className={cn('size-3.5', loading && 'animate-spin text-emerald-400')} />
            Refresh
          </Button>

          {/* Reprocess Failed Button */}
          {(statusCounts.FAILED || 0) > 0 && (
            <Button
              variant="destructive"
              size="sm"
              onClick={handleReprocessFailed}
              disabled={reprocessing}
              className="h-8 gap-1.5 text-xs shadow-sm shadow-rose-950 transition-all active:scale-[0.98]"
            >
              <RotateCcwIcon className={cn('size-3.5', reprocessing && 'animate-spin')} />
              Reprocess Failed ({statusCounts.FAILED})
            </Button>
          )}
        </div>
      </div>

      {/* 2. System Hardware & Overload Protection Telemetry Cards */}
      {telemetry && (
        <div className="space-y-3">
          {/* Overload Alert Banner */}
          {telemetry.system_status !== 'HEALTHY' && (
            <div
              className={cn(
                'rounded-lg p-3 flex items-start gap-3 border shadow-sm transition-all',
                telemetry.system_status === 'CRITICAL'
                  ? 'bg-rose-500/10 border-rose-500/30 text-rose-300'
                  : 'bg-amber-500/10 border-amber-500/30 text-amber-300'
              )}
            >
              <ShieldAlertIcon className="size-5 shrink-0 mt-0.5 text-rose-400" />
              <div className="flex-1 text-xs">
                <span className="font-semibold text-sm block">
                  System Load Notice: {telemetry.system_status}
                </span>
                {telemetry.message} CPU is at{' '}
                <span className="font-mono font-bold">{telemetry.cpu.percent}%</span>, RAM at{' '}
                <span className="font-mono font-bold">{telemetry.memory.percent}%</span>. To prevent pipeline crashes, reduce parallel batch insertion or pause non-critical stages.
              </div>
            </div>
          )}

          {/* Hardware Gauge Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {/* CPU Card */}
            <Card className="bg-card/90 border-border/60 p-3.5 relative overflow-hidden">
              <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
                <span className="flex items-center gap-1.5 font-medium">
                  <CpuIcon className="size-3.5 text-cyan-400" />
                  CPU Utilization
                </span>
                <span className="font-mono text-muted-foreground">{telemetry.cpu.cores} Cores</span>
              </div>
              <div className="flex items-baseline justify-between">
                <span
                  className={cn(
                    'text-2xl font-bold font-mono tracking-tight',
                    telemetry.cpu.percent > 85 ? 'text-rose-400' : telemetry.cpu.percent > 70 ? 'text-amber-400' : 'text-emerald-400'
                  )}
                >
                  {telemetry.cpu.percent}%
                </span>
                <span className="text-[11px] font-mono text-muted-foreground">
                  Load: {telemetry.cpu.load_avg.join(', ')}
                </span>
              </div>
              {/* Progress bar */}
              <div className="w-full bg-secondary/60 h-1.5 rounded-full overflow-hidden mt-2">
                <div
                  className={cn(
                    'h-full transition-all duration-500 rounded-full',
                    telemetry.cpu.percent > 85 ? 'bg-rose-500' : telemetry.cpu.percent > 70 ? 'bg-amber-500' : 'bg-cyan-500'
                  )}
                  style={{ width: `${Math.min(telemetry.cpu.percent, 100)}%` }}
                />
              </div>
            </Card>

            {/* RAM Memory Card */}
            <Card className="bg-card/90 border-border/60 p-3.5 relative overflow-hidden">
              <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
                <span className="flex items-center gap-1.5 font-medium">
                  <HardDriveIcon className="size-3.5 text-violet-400" />
                  RAM Memory
                </span>
                <span className="font-mono text-muted-foreground">{telemetry.memory.total_gb} GB Total</span>
              </div>
              <div className="flex items-baseline justify-between">
                <span
                  className={cn(
                    'text-2xl font-bold font-mono tracking-tight',
                    telemetry.memory.percent > 85 ? 'text-rose-400' : telemetry.memory.percent > 75 ? 'text-amber-400' : 'text-violet-400'
                  )}
                >
                  {telemetry.memory.percent}%
                </span>
                <span className="text-[11px] font-mono text-muted-foreground">
                  {telemetry.memory.used_gb} GB used / {telemetry.memory.free_gb} GB free
                </span>
              </div>
              <div className="w-full bg-secondary/60 h-1.5 rounded-full overflow-hidden mt-2">
                <div
                  className={cn(
                    'h-full transition-all duration-500 rounded-full',
                    telemetry.memory.percent > 85 ? 'bg-rose-500' : 'bg-violet-500'
                  )}
                  style={{ width: `${Math.min(telemetry.memory.percent, 100)}%` }}
                />
              </div>
            </Card>

            {/* PostgreSQL Shared Pool Card */}
            <Card className="bg-card/90 border-border/60 p-3.5 relative overflow-hidden">
              <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
                <span className="flex items-center gap-1.5 font-medium">
                  <DatabaseIcon className="size-3.5 text-blue-400" />
                  PostgreSQL Pool
                </span>
                <span className="font-mono text-xs text-emerald-400 font-semibold">
                  {telemetry.postgresql_pool?.status || 'Active'}
                </span>
              </div>
              <div className="flex items-baseline justify-between">
                <span className="text-2xl font-bold font-mono tracking-tight text-blue-400">
                  {telemetry.postgresql_pool?.size || 0}
                  <span className="text-xs text-muted-foreground font-normal">
                    /{telemetry.postgresql_pool?.max_size || 400}
                  </span>
                </span>
                <span className="text-[11px] font-mono text-muted-foreground">
                  Idle: {telemetry.postgresql_pool?.idle_size ?? '-'} conn
                </span>
              </div>
              <div className="w-full bg-secondary/60 h-1.5 rounded-full overflow-hidden mt-2">
                <div
                  className="h-full bg-blue-500 transition-all duration-500 rounded-full"
                  style={{
                    width: `${Math.min(
                      ((telemetry.postgresql_pool?.size || 0) / (telemetry.postgresql_pool?.max_size || 400)) * 100,
                      100
                    )}%`
                  }}
                />
              </div>
            </Card>

            {/* System Status & Pipeline Health */}
            <Card className="bg-card/90 border-border/60 p-3.5 relative overflow-hidden">
              <div className="flex items-center justify-between text-xs text-muted-foreground mb-1">
                <span className="flex items-center gap-1.5 font-medium">
                  <ActivityIcon className="size-3.5 text-emerald-400" />
                  Server Health
                </span>
                <span className="text-xs font-mono text-muted-foreground">
                  Disk: {telemetry.disk.percent}%
                </span>
              </div>
              <div className="flex items-center gap-2 mt-1">
                <span
                  className={cn(
                    'px-2 py-0.5 rounded text-xs font-mono font-bold uppercase tracking-wider',
                    telemetry.system_status === 'HEALTHY'
                      ? 'bg-emerald-500/15 text-emerald-400 border border-emerald-500/30'
                      : telemetry.system_status === 'WARNING'
                      ? 'bg-amber-500/15 text-amber-400 border border-amber-500/30'
                      : 'bg-rose-500/15 text-rose-400 border border-rose-500/30 animate-pulse'
                  )}
                >
                  {telemetry.system_status}
                </span>
                <span className="text-xs text-muted-foreground truncate">
                  {telemetry.system_status === 'HEALTHY' ? 'Normal Operating Range' : 'High Resource Load'}
                </span>
              </div>
              <p className="text-[11px] text-muted-foreground font-mono mt-2 truncate">
                Updated at {new Date(telemetry.timestamp * 1000).toLocaleTimeString()}
              </p>
            </Card>
          </div>
        </div>
      )}

      {/* 3. Ingestion KPI Overview Banner */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
        {/* Total Documents */}
        <div
          onClick={() => setStatusFilter('all')}
          className={cn(
            'p-3 rounded-lg border cursor-pointer transition-all active:scale-[0.98]',
            statusFilter === 'all'
              ? 'bg-zinc-800/80 border-emerald-500/50 shadow-sm'
              : 'bg-card/50 border-border/50 hover:bg-card/80'
          )}
        >
          <div className="text-xs text-muted-foreground font-medium">Total Docs</div>
          <div className="text-xl font-bold font-mono mt-0.5">{totalDocs.toLocaleString()}</div>
        </div>

        {/* In-Progress (Uploading + Extracting + Chunking) */}
        <div
          onClick={() => setStatusFilter('CHUNKING')}
          className={cn(
            'p-3 rounded-lg border cursor-pointer transition-all active:scale-[0.98]',
            statusFilter === 'CHUNKING'
              ? 'bg-cyan-950/40 border-cyan-500/50 shadow-sm'
              : 'bg-card/50 border-border/50 hover:bg-card/80'
          )}
        >
          <div className="text-xs text-cyan-400 font-medium flex items-center gap-1">
            <span className="h-1.5 w-1.5 rounded-full bg-cyan-400 animate-ping" />
            In Pipeline
          </div>
          <div className="text-xl font-bold font-mono text-cyan-300 mt-0.5">
            {processingCount.toLocaleString()}
          </div>
        </div>

        {/* Extracted */}
        <div
          onClick={() => setStatusFilter('EXTRACTED')}
          className={cn(
            'p-3 rounded-lg border cursor-pointer transition-all active:scale-[0.98]',
            statusFilter === 'EXTRACTED'
              ? 'bg-blue-950/40 border-blue-500/50 shadow-sm'
              : 'bg-card/50 border-border/50 hover:bg-card/80'
          )}
        >
          <div className="text-xs text-blue-400 font-medium">Extracted</div>
          <div className="text-xl font-bold font-mono text-blue-300 mt-0.5">
            {(statusCounts.EXTRACTED || 0).toLocaleString()}
          </div>
        </div>

        {/* Processed (Completed) */}
        <div
          onClick={() => setStatusFilter('PROCESSED')}
          className={cn(
            'p-3 rounded-lg border cursor-pointer transition-all active:scale-[0.98]',
            statusFilter === 'PROCESSED'
              ? 'bg-emerald-950/40 border-emerald-500/50 shadow-sm'
              : 'bg-card/50 border-border/50 hover:bg-card/80'
          )}
        >
          <div className="text-xs text-emerald-400 font-medium flex items-center gap-1">
            <CheckCircle2Icon className="size-3 text-emerald-400" />
            Processed
          </div>
          <div className="text-xl font-bold font-mono text-emerald-300 mt-0.5">
            {(statusCounts.PROCESSED || 0).toLocaleString()}
          </div>
        </div>

        {/* Failed */}
        <div
          onClick={() => setStatusFilter('FAILED')}
          className={cn(
            'p-3 rounded-lg border cursor-pointer transition-all active:scale-[0.98]',
            statusFilter === 'FAILED'
              ? 'bg-rose-950/40 border-rose-500/50 shadow-sm'
              : 'bg-card/50 border-border/50 hover:bg-card/80'
          )}
        >
          <div className="text-xs text-rose-400 font-medium flex items-center gap-1">
            <AlertTriangleIcon className="size-3 text-rose-400" />
            Failed
          </div>
          <div className="text-xl font-bold font-mono text-rose-300 mt-0.5">
            {(statusCounts.FAILED || 0).toLocaleString()}
          </div>
        </div>

        {/* Quick Filter: Legal Docs Only */}
        <div
          onClick={() => setDocTypeFilter((prev) => (prev === 'legal' ? 'all' : 'legal'))}
          className={cn(
            'p-3 rounded-lg border cursor-pointer transition-all active:scale-[0.98]',
            docTypeFilter === 'legal'
              ? 'bg-violet-950/40 border-violet-500/50 shadow-sm'
              : 'bg-card/50 border-border/50 hover:bg-card/80'
          )}
        >
          <div className="text-xs text-violet-400 font-medium flex items-center gap-1">
            <ScaleIcon className="size-3 text-violet-400" />
            Legal Docs
          </div>
          <div className="text-xs text-muted-foreground mt-1">
            {docTypeFilter === 'legal' ? 'Filtering Legal' : 'Click to filter'}
          </div>
        </div>
      </div>

      {/* 4. Filter Toolbar: Search, Doc Type, Workspace */}
      <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-between gap-3 bg-card border border-border/60 rounded-xl p-3 shadow-sm">
        {/* Search input with quick clear */}
        <div className="relative flex-1 min-w-[200px]">
          <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search by filename, document ID, or error message..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-background border border-border/60 rounded-lg pl-9 pr-8 py-1.5 text-xs focus:outline-none focus:ring-1 focus:ring-emerald-500 text-foreground placeholder:text-muted-foreground"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            >
              <XIcon className="size-3.5" />
            </button>
          )}
        </div>

        {/* Document Type Dropdown */}
        <div className="flex items-center gap-2">
          <label className="text-xs text-muted-foreground whitespace-nowrap">Doc Type:</label>
          <select
            value={docTypeFilter}
            onChange={(e) => setDocTypeFilter(e.target.value as any)}
            className="bg-background border border-border/60 rounded-lg px-2.5 py-1.5 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-emerald-500 cursor-pointer"
          >
            <option value="all">All Types</option>
            <option value="legal">Legal (Văn bản pháp luật)</option>
            <option value="general">General Documents</option>
          </select>
        </div>

        {/* Workspace Dropdown */}
        <div className="flex items-center gap-2">
          <label className="text-xs text-muted-foreground whitespace-nowrap">Workspace:</label>
          <select
            value={filterWorkspace}
            onChange={(e) => {
              setFilterWorkspace(e.target.value)
              setCurrentPage(1)
            }}
            className="bg-background border border-border/60 rounded-lg px-2.5 py-1.5 text-xs text-foreground focus:outline-none focus:ring-1 focus:ring-emerald-500 cursor-pointer max-w-[180px] truncate"
          >
            <option value="all">All Workspaces</option>
            {workspaces.map((ws) => (
              <option key={ws.workspace_id} value={ws.workspace_id}>
                {ws.workspace_name || ws.workspace_id}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* 5. Document List Table */}
      <Card className="bg-card border-border/60 overflow-hidden shadow-sm flex-1 flex flex-col">
        <div className="overflow-x-auto">
          <Table>
            <TableHeader className="bg-secondary/40">
              <TableRow className="border-border/60 hover:bg-transparent">
                <TableHead className="w-[40%] text-xs font-semibold">Document / Title</TableHead>
                <TableHead className="w-[12%] text-xs font-semibold">Type</TableHead>
                <TableHead className="w-[15%] text-xs font-semibold">Status & Stage</TableHead>
                <TableHead className="w-[15%] text-xs font-semibold">Workspace</TableHead>
                <TableHead className="w-[10%] text-xs font-semibold">Updated At</TableHead>
                <TableHead className="w-[8%] text-xs font-semibold text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {loading && filteredDocuments.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="h-48 text-center text-muted-foreground">
                    <div className="flex flex-col items-center justify-center gap-2">
                      <RefreshCwIcon className="size-6 animate-spin text-emerald-400" />
                      <span className="text-xs">Loading document pipeline telemetry...</span>
                    </div>
                  </TableCell>
                </TableRow>
              ) : filteredDocuments.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={6} className="h-48 text-center">
                    <div className="flex flex-col items-center justify-center gap-1.5 text-muted-foreground">
                      <FileTextIcon className="size-8 text-muted-foreground/50" />
                      <p className="text-xs font-medium">No documents matching current filters</p>
                      {(searchQuery || statusFilter !== 'all' || docTypeFilter !== 'all') && (
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => {
                            setSearchQuery('')
                            setStatusFilter('all')
                            setDocTypeFilter('all')
                          }}
                          className="text-xs text-emerald-400 hover:text-emerald-300 h-7"
                        >
                          Clear all filters
                        </Button>
                      )}
                    </div>
                  </TableCell>
                </TableRow>
              ) : (
                filteredDocuments.map((doc) => {
                  const isLegal = checkIsLegal(doc)
                  const statusConf = STATUS_CONFIGS[doc.status] || {
                    label: doc.status,
                    color: 'text-zinc-400',
                    bg: 'bg-zinc-500/10',
                    border: 'border-zinc-500/20'
                  }

                  const activeStage = getDocStageInfo(doc.id)

                  return (
                    <TableRow
                      key={doc.id}
                      className="border-border/40 hover:bg-secondary/30 transition-colors group cursor-pointer"
                      onClick={() => setSelectedDoc(doc)}
                    >
                      {/* Document Name / Title */}
                      <TableCell className="py-2.5">
                        <div className="flex flex-col">
                          <span className="font-medium text-xs text-foreground group-hover:text-emerald-400 transition-colors truncate max-w-[420px]">
                            {doc.file_path ? doc.file_path.split('/').pop() : doc.id}
                          </span>
                          <div className="flex items-center gap-1.5 mt-0.5">
                            <span className="text-[11px] font-mono text-muted-foreground truncate max-w-[240px]">
                              {doc.id}
                            </span>
                            <button
                              onClick={(e) => {
                                e.stopPropagation()
                                handleCopy(doc.id, doc.id)
                              }}
                              className="text-muted-foreground hover:text-foreground p-0.5"
                              title="Copy ID"
                            >
                              {copiedId === doc.id ? (
                                <CheckIcon className="size-3 text-emerald-400" />
                              ) : (
                                <CopyIcon className="size-3" />
                              )}
                            </button>
                            {doc.error_msg && (
                              <span className="text-[11px] text-rose-400 font-mono truncate max-w-[200px]">
                                • {doc.error_msg}
                              </span>
                            )}
                          </div>
                        </div>
                      </TableCell>

                      {/* Type Badge */}
                      <TableCell className="py-2.5">
                        {isLegal ? (
                          <Badge
                            variant="outline"
                            className="text-[10px] bg-violet-500/10 text-violet-400 border-violet-500/30 gap-1 py-0 px-1.5 font-normal"
                          >
                            <ScaleIcon className="size-2.5" />
                            Legal
                          </Badge>
                        ) : (
                          <Badge
                            variant="outline"
                            className="text-[10px] bg-secondary text-muted-foreground border-border/60 py-0 px-1.5 font-normal"
                          >
                            General
                          </Badge>
                        )}
                      </TableCell>

                      {/* Status & Live Pipeline Stage */}
                      <TableCell className="py-2.5">
                        <div className="flex flex-col gap-1">
                          <div className="flex items-center gap-1.5">
                            <span
                              className={cn(
                                'text-[11px] font-mono font-medium px-2 py-0.5 rounded-full border',
                                statusConf.color,
                                statusConf.bg,
                                statusConf.border
                              )}
                            >
                              {statusConf.label}
                            </span>
                            {activeStage && (
                              <span className="text-[10px] font-mono text-cyan-400 animate-pulse bg-cyan-500/10 px-1.5 py-0.5 rounded border border-cyan-500/30">
                                {activeStage.stage} ({activeStage.status})
                              </span>
                            )}
                          </div>
                        </div>
                      </TableCell>

                      {/* Workspace */}
                      <TableCell className="py-2.5 text-xs text-muted-foreground">
                        <span className="truncate max-w-[120px] block font-mono text-[11px]" title={doc.workspace_id || (doc as any).metadata?.workspace_id}>
                          {doc.workspace_id || (doc as any).metadata?.workspace_id || selectedWorkspaceId || 'default'}
                        </span>
                      </TableCell>

                      {/* Updated Time */}
                      <TableCell className="py-2.5 text-xs text-muted-foreground font-mono">
                        {doc.updated_at ? new Date(doc.updated_at).toLocaleTimeString() : '-'}
                      </TableCell>

                      {/* Actions */}
                      <TableCell className="py-2.5 text-right">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation()
                            setSelectedDoc(doc)
                          }}
                          className="h-7 text-xs text-muted-foreground hover:text-emerald-400 px-2"
                        >
                          View
                        </Button>
                      </TableCell>
                    </TableRow>
                  )
                })
              )}
            </TableBody>
          </Table>
        </div>

        {/* Pagination bar */}
        <div className="p-3 border-t border-border/40 flex items-center justify-between bg-card">
          <span className="text-xs text-muted-foreground font-mono">
            Showing {filteredDocuments.length} of {totalCount.toLocaleString()} items
          </span>
          <PaginationControls
            currentPage={currentPage}
            totalPages={Math.max(1, Math.ceil(totalCount / pageSize))}
            totalCount={totalCount}
            onPageChange={setCurrentPage}
            pageSize={pageSize}
            onPageSizeChange={(newSize) => {
              setPageSize(newSize)
              setCurrentPage(1)
            }}
          />
        </div>
      </Card>

      {/* 6. Document Inspection Modal */}
      {selectedDoc && (
        <Dialog open={!!selectedDoc} onOpenChange={(open) => !open && setSelectedDoc(null)}>
          <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto bg-card border-border/80">
            <DialogHeader className="border-b border-border/40 pb-3">
              <div className="flex items-center justify-between">
                <DialogTitle className="text-base font-bold flex items-center gap-2">
                  <FileCode2Icon className="size-4 text-emerald-400" />
                  Document Detail Inspector
                </DialogTitle>
                <Badge
                  variant="outline"
                  className={cn(
                    'text-xs font-mono',
                    STATUS_CONFIGS[selectedDoc.status]?.color,
                    STATUS_CONFIGS[selectedDoc.status]?.bg,
                    STATUS_CONFIGS[selectedDoc.status]?.border
                  )}
                >
                  {selectedDoc.status}
                </Badge>
              </div>
              <DialogDescription className="text-xs text-muted-foreground font-mono truncate">
                ID: {selectedDoc.id}
              </DialogDescription>
            </DialogHeader>

            <div className="space-y-4 py-2 text-xs">
              {/* File details */}
              <div className="grid grid-cols-2 gap-2 bg-secondary/30 p-3 rounded-lg border border-border/40">
                <div>
                  <span className="text-muted-foreground block text-[11px]">Filename / Path:</span>
                  <span className="font-medium text-foreground break-all">{selectedDoc.file_path || selectedDoc.id}</span>
                </div>
                <div>
                  <span className="text-muted-foreground block text-[11px]">Last Updated:</span>
                  <span className="font-mono text-foreground">
                    {selectedDoc.updated_at ? new Date(selectedDoc.updated_at).toLocaleString() : '-'}
                  </span>
                </div>
              </div>

              {/* Pipeline Stage Progression Sequence */}
              <div className="p-3 bg-secondary/30 rounded-lg border border-border/40">
                <span className="font-semibold block mb-2 text-[11px] text-muted-foreground uppercase tracking-wider">
                  Pipeline Stage Sequence:
                </span>
                <div className="flex flex-wrap items-center gap-1.5">
                  {PIPELINE_STAGES.map((stg, idx) => (
                    <div key={stg.id} className="flex items-center gap-1.5">
                      <span
                        className={cn(
                          'text-[10px] font-mono px-2 py-0.5 rounded border',
                          selectedDoc.status === 'PROCESSED'
                            ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                            : selectedDoc.status === 'FAILED'
                            ? 'bg-rose-500/10 text-rose-400 border-rose-500/20'
                            : 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20'
                        )}
                      >
                        {stg.label}
                      </span>
                      {idx < PIPELINE_STAGES.length - 1 && (
                        <span className="text-muted-foreground/60 text-[10px]">→</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Error Message banner if failed */}
              {selectedDoc.error_msg && (
                <div className="p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg text-rose-300">
                  <div className="flex items-center gap-1.5 font-bold mb-1 text-rose-400">
                    <AlertTriangleIcon className="size-4" />
                    Failure Details:
                  </div>
                  <pre className="font-mono text-[11px] whitespace-pre-wrap break-all bg-black/40 p-2 rounded">
                    {selectedDoc.error_msg}
                  </pre>
                </div>
              )}

              {/* Legal specific metadata if present */}
              {(checkIsLegal(selectedDoc) || (selectedDoc as any).legal_info || (selectedDoc as any).metadata?.doc_type === 'legal') && (
                <div className="p-3 bg-violet-500/10 border border-violet-500/30 rounded-lg text-violet-300">
                  <div className="flex items-center gap-1.5 font-semibold mb-2 text-violet-400">
                    <ScaleIcon className="size-3.5" />
                    Legal Document Information:
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-[11px]">
                    <div>
                      <span className="text-muted-foreground">Document Code:</span>{' '}
                      <span className="font-mono font-medium text-foreground">
                        {(selectedDoc as any).metadata?.document_code || (selectedDoc as any).legal_info?.document_number || '-'}
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Document Type:</span>{' '}
                      <span className="font-medium text-foreground">
                        {(selectedDoc as any).metadata?.document_type || (selectedDoc as any).metadata?.legal_document_type || '-'}
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Issuing Authority:</span>{' '}
                      <span className="text-foreground">
                        {(selectedDoc as any).metadata?.issuing_authority || (selectedDoc as any).legal_info?.issuing_authority || '-'}
                      </span>
                    </div>
                    <div>
                      <span className="text-muted-foreground">Legal Status:</span>{' '}
                      <span className="font-medium text-amber-400">
                        {(selectedDoc as any).metadata?.legal_status || '-'}
                      </span>
                    </div>
                    {((selectedDoc as any).metadata?.issuance_date || (selectedDoc as any).metadata?.effective_date) && (
                      <>
                        <div>
                          <span className="text-muted-foreground">Issuance Date:</span>{' '}
                          <span className="font-mono text-foreground">
                            {(selectedDoc as any).metadata?.issuance_date || '-'}
                          </span>
                        </div>
                        <div>
                          <span className="text-muted-foreground">Effective Date:</span>{' '}
                          <span className="font-mono text-foreground">
                            {(selectedDoc as any).metadata?.effective_date || '-'}
                          </span>
                        </div>
                      </>
                    )}
                  </div>
                </div>
              )}

              {/* Raw Document JSON */}
              <div>
                <div className="flex items-center justify-between mb-1">
                  <span className="font-semibold text-muted-foreground text-[11px]">Complete Raw Document Record:</span>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleCopy(JSON.stringify(selectedDoc, null, 2), 'raw_json')}
                    className="h-6 text-[10px] gap-1 px-1.5"
                  >
                    {copiedId === 'raw_json' ? <CheckIcon className="size-3 text-emerald-400" /> : <CopyIcon className="size-3" />}
                    Copy JSON
                  </Button>
                </div>
                <pre className="bg-zinc-950 p-3 rounded-lg border border-border/60 font-mono text-[11px] text-zinc-300 overflow-x-auto max-h-60">
                  {JSON.stringify(selectedDoc, null, 2)}
                </pre>
              </div>
            </div>

            <div className="border-t border-border/40 pt-3 flex justify-end">
              <Button variant="outline" size="sm" onClick={() => setSelectedDoc(null)} className="text-xs">
                Close
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      )}
    </div>
  )
}
